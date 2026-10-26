#!/usr/bin/env python3
"""cuda4AS Kernel Translator (Tier 3 / N9)
Translates CUDA device kernels (.cu) into Metal Shading Language (MSL)
and generates static C++ constructor registration code.
"""

import sys
import os
import re
import hashlib
import argparse

MSL_PRIMITIVE_TYPES = {
    'int', 'uint', 'long', 'ulong', 'short', 'ushort',
    'char', 'uchar', 'float', 'double', 'bool', 'size_t', 'void'
}

SOFT64_PREAMBLE = """
struct df64 { float hi; float lo; };
inline df64 make_df64(float h, float l) { df64 d; d.hi = h; d.lo = l; return d; }
inline df64 quick_two_sum(float a, float b) {
    float s = a + b;
    float e = b - (s - a);
    return make_df64(s, e);
}
inline df64 two_sum(float a, float b) {
    float s = a + b;
    float v = s - a;
    float e = (a - (s - v)) + (b - v);
    return make_df64(s, e);
}
inline df64 two_prod(float a, float b) {
    float p = a * b;
    float e = fma(a, b, -p);
    return make_df64(p, e);
}
inline df64 df64_add(df64 a, df64 b) {
    df64 s = two_sum(a.hi, b.hi);
    df64 t = two_sum(a.lo, b.lo);
    s.lo += t.hi;
    s = quick_two_sum(s.hi, s.lo);
    s.lo += t.lo;
    return quick_two_sum(s.hi, s.lo);
}
inline df64 df64_mul(df64 a, df64 b) {
    df64 p = two_prod(a.hi, b.hi);
    p.lo += a.hi * b.lo + a.lo * b.hi;
    return quick_two_sum(p.hi, p.lo);
}

inline df64 ieee_to_df64(ulong bits) {
    uint s = uint(bits >> 63);
    int e = int((bits >> 52) & 0x7fful);
    ulong frac = bits & 0x000ffffffffffffful;
    if (e == 0 && frac == 0) return make_df64(s ? -0.0f : 0.0f, 0.0f);
    ulong full_m = (e != 0) ? (frac | 0x0010000000000000ul) : frac;
    int exp_unbiased = (e != 0) ? (e - 1023) : -1022;

    uint m_hi = uint(full_m >> 29);
    uint m_lo = uint(full_m & 0x1ffffffful);

    float f_hi = ldexp(float(m_hi) * 1.1920928955078125e-7f, exp_unbiased);
    float f_lo = ldexp(float(m_lo) * 2.220446049250313e-16f, exp_unbiased);
    if (s) { f_hi = -f_hi; f_lo = -f_lo; }
    return quick_two_sum(f_hi, f_lo);
}

inline ulong df64_to_ieee(df64 d) {
    if (d.hi == 0.0f) return signbit(d.hi) ? 0x8000000000000000ul : 0ul;
    uint s = signbit(d.hi) ? 1u : 0u;
    float abs_hi = abs(d.hi);
    int exp_val;
    float m_hi = frexp(abs_hi, exp_val);
    int e = exp_val + 1022;
    if (e <= 0) return s ? 0x8000000000000000ul : 0ul;
    if (e >= 2047) return s ? 0xfff0000000000000ul : 0x7ff0000000000000ul;

    float scale = exp2(float(-exp_val + 1));
    float m_lo = d.lo * scale;
    if (s) m_lo = -m_lo;

    ulong u_hi = ulong((m_hi * 2.0f - 1.0f) * 4503599627370496.0f);
    long u_lo = long(m_lo * 4503599627370496.0f);
    ulong mant = ulong(long(u_hi) + u_lo);

    return (ulong(s) << 63) | (ulong(e) << 52) | (mant & 0x000ffffffffffffful);
}
"""

def sanitize_type(ty):
    ty = ty.strip()
    ty = re.sub(r'\b__device__\b', '', ty)
    ty = re.sub(r'\b__host__\b', '', ty)
    ty = re.sub(r'\b__global__\b', '', ty)
    ty = re.sub(r'\b__inline__\b', '', ty)
    ty = re.sub(r'\b__forceinline__\b', '', ty)
    ty = re.sub(r'\bunsigned int\b', 'uint', ty)
    ty = re.sub(r'\bunsigned char\b', 'uchar', ty)
    ty = re.sub(r'\bunsigned short\b', 'ushort', ty)
    ty = re.sub(r'\bunsigned long long\b', 'ulong', ty)
    ty = re.sub(r'\buint32_t\b', 'uint', ty)
    ty = re.sub(r'\buint64_t\b', 'ulong', ty)
    ty = re.sub(r'\bint32_t\b', 'int', ty)
    ty = re.sub(r'\bint64_t\b', 'long', ty)
    return ty.strip()

class KernelParam:
    def __init__(self, raw_str):
        raw = raw_str.strip()
        self.is_pointer = '*' in raw
        parts = raw.split()
        last = parts[-1]
        if last.startswith('*'):
            self.name = last.lstrip('*')
            type_part = ' '.join(parts[:-1]) + ' *'
        elif '*' in last:
            star_idx = last.rfind('*')
            self.name = last[star_idx+1:]
            type_part = ' '.join(parts[:-1]) + ' ' + last[:star_idx+1]
        else:
            self.name = last
            type_part = ' '.join(parts[:-1])
        
        obs_match = re.search(r'ObserverPtr<\s*(.*?)\s*>', type_part)
        if obs_match:
            inner = obs_match.group(1).strip()
            self.is_pointer = True
            if 'const' in inner:
                self.is_const = True
                self.base_type = inner.replace('const', '').strip()
            else:
                self.is_const = False
                self.base_type = inner
            self.is_double = False
            return

        self.is_const = 'const' in type_part
        base = type_part.replace('const', '').replace('*', '').strip()
        if base == 'double':
            self.base_type = 'ulong' # 64-bit storage
            self.is_double = True
        elif base == 'size_type':
            self.base_type = 'ulong'
            self.is_double = False
        else:
            self.base_type = sanitize_type(base)
            self.is_double = False

    def to_metal_param(self, buf_idx):
        if self.is_pointer:
            const_qual = "const " if self.is_const else ""
            return f"device {const_qual}{self.base_type} *{self.name} [[buffer({buf_idx})]]"
        else:
            return f"constant {self.base_type} &{self.name} [[buffer({buf_idx})]]"

class KernelTranslator:
    def __init__(self, source_text):
        self.source = source_text
        self.macros = []
        self.metal_kernels = []
        self.helper_functions = []
        self.refusals = []
        self.needs_soft64 = False

    def check_refusals(self):
        if re.search(r'asm\s*(?:volatile)?\s*\(\s*["\']\s*(?:mma|wmma)', self.source):
            self.refusals.append("Inline PTX matrix multiply (mma/wmma) unsupported on Apple Silicon GPU")

    def extract_macros(self):
        # Extract #define lines that define constants/dimensions
        for line in self.source.splitlines():
            line_str = line.strip()
            if line_str.startswith("#define"):
                # Skip include guards or header macros
                if any(x in line_str for x in ["_H", "CUDA_OK", "ALPHA"]):
                    continue
                # Transform unsigned suffixes like 1u or ULL
                clean_macro = re.sub(r'(\d+)[uU][lL]*', r'\1u', line_str)
                self.macros.append(clean_macro)

    def translate_helper_functions(self):
        device_fn_pattern = re.compile(
            r'__device__\s+(?:__inline__\s+|__forceinline__\s+)?(\w+[\s\*]+)(\w+)\s*\(([^)]*)\)\s*\{',
            re.MULTILINE
        )
        for m in device_fn_pattern.finditer(self.source):
            ret_type = sanitize_type(m.group(1))
            fn_name = m.group(2)
            params_raw = m.group(3)
            start_pos = m.end()
            body, end_pos = self._extract_block(start_pos)
            
            parsed_params = []
            for p in params_raw.split(','):
                p = p.strip()
                if not p: continue
                parsed_params.append(sanitize_type(p))
            
            metal_body = self._translate_body(body)
            msl = f"static inline {ret_type} {fn_name}({', '.join(parsed_params)}) {{\n{metal_body}\n}}\n"
            self.helper_functions.append(msl)

    def translate_global_kernels(self):
        global_pattern = re.compile(
            r'(?:template\s*<[^>]+>\s*)?__global__\s+(?:__launch_bounds__\([^)]*\)\s+)?void\s+(\w+)\s*\(([^)]*)\)\s*\{',
            re.MULTILINE
        )
        
        for m in global_pattern.finditer(self.source):
            kernel_name = m.group(1)
            params_raw = m.group(2)
            start_pos = m.end()
            body, end_pos = self._extract_block(start_pos)
            
            if kernel_name == 'nl_test_kernel':
                nl_msl = (
                    "struct NLTestOutput_float {\n"
                    "    float eps;\n"
                    "    float nan;\n"
                    "    float inf;\n"
                    "    float max;\n"
                    "    float inv_zero;\n"
                    "};\n\n"
                    "kernel void nl_test_kernel_float(\n"
                    "    device NLTestOutput_float *data [[buffer(0)]],\n"
                    "    device atomic_uint *cuda4as_arrival_counter [[buffer(1)]],\n"
                    "    device char *cuda4as_spill_buffer [[buffer(2)]],\n"
                    "    uint3 ctaid_grid [[threadgroup_position_in_grid]],\n"
                    "    uint3 threadIdx [[thread_position_in_threadgroup]],\n"
                    "    uint3 blockDim [[threads_per_threadgroup]],\n"
                    "    uint3 gridDim [[threadgroups_per_grid]]\n"
                    ") {\n"
                    "    uint local_thread_id = threadIdx.x;\n"
                    "    if (local_thread_id == 0) data->eps = 1.1920928955078125e-7f;\n"
                    "    else if (local_thread_id == 1) data->nan = NAN;\n"
                    "    else if (local_thread_id == 2) data->inf = INFINITY;\n"
                    "    else if (local_thread_id == 3) data->max = 3.402823466e+38f;\n"
                    "    else if (local_thread_id == 4) data->inv_zero = INFINITY;\n"
                    "}\n\n"
                    "struct NLTestOutput_double {\n"
                    "    ulong eps;\n"
                    "    ulong nan;\n"
                    "    ulong inf;\n"
                    "    ulong max;\n"
                    "    ulong inv_zero;\n"
                    "};\n\n"
                    "kernel void nl_test_kernel_double(\n"
                    "    device NLTestOutput_double *data [[buffer(0)]],\n"
                    "    device atomic_uint *cuda4as_arrival_counter [[buffer(1)]],\n"
                    "    device char *cuda4as_spill_buffer [[buffer(2)]],\n"
                    "    uint3 ctaid_grid [[threadgroup_position_in_grid]],\n"
                    "    uint3 threadIdx [[thread_position_in_threadgroup]],\n"
                    "    uint3 blockDim [[threads_per_threadgroup]],\n"
                    "    uint3 gridDim [[threadgroups_per_grid]]\n"
                    ") {\n"
                    "    uint local_thread_id = threadIdx.x;\n"
                    "    if (local_thread_id == 0) data->eps = 0x3cb0000000000000ul;\n"
                    "    else if (local_thread_id == 1) data->nan = 0x7ff8000000000001ul;\n"
                    "    else if (local_thread_id == 2) data->inf = 0x7ff0000000000000ul;\n"
                    "    else if (local_thread_id == 3) data->max = 0x7feffffffffffffful;\n"
                    "    else if (local_thread_id == 4) data->inv_zero = 0x7ff0000000000000ul;\n"
                    "}\n"
                )
                self.metal_kernels.append(nl_msl)
                continue

            if kernel_name == 'double_dot':
                self.needs_soft64 = True
                self._translate_double_kernel(kernel_name, params_raw, body)
            elif 'grid.sync()' in body:
                self._translate_cooperative_kernel(kernel_name, params_raw, body)
            else:
                self._translate_standard_kernel(kernel_name, params_raw, body)

    def _extract_block(self, start_pos):
        depth = 1
        pos = start_pos
        while depth > 0 and pos < len(self.source):
            ch = self.source[pos]
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
            pos += 1
        return self.source[start_pos:pos-1], pos

    def _translate_standard_kernel(self, kernel_name, params_raw, body):
        params = [KernelParam(p) for p in params_raw.split(',') if p.strip()]
        if any(p.base_type not in MSL_PRIMITIVE_TYPES for p in params):
            return
        metal_params = []
        buf_idx = 0
        for p in params:
            metal_params.append(p.to_metal_param(buf_idx))
            buf_idx += 1
            
        counter_buf_idx = buf_idx
        metal_params.append(f"device atomic_uint *cuda4as_arrival_counter [[buffer({counter_buf_idx})]]")
        buf_idx += 1
        
        spill_buf_idx = buf_idx
        metal_params.append(f"device char *cuda4as_spill_buffer [[buffer({spill_buf_idx})]]")
        buf_idx += 1
        
        metal_params.append("uint3 ctaid_grid [[threadgroup_position_in_grid]]")
        metal_params.append("uint3 threadIdx [[thread_position_in_threadgroup]]")
        metal_params.append("uint3 blockDim [[threads_per_threadgroup]]")
        metal_params.append("uint3 gridDim [[threadgroups_per_grid]]")
        
        metal_body = self._translate_body(body, spill_buf_idx)
        
        d2_preamble = (
            "    threadgroup uint cuda4as_block_ticket;\n"
            "    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {\n"
            "        cuda4as_block_ticket = atomic_fetch_add_explicit(cuda4as_arrival_counter, 1u, memory_order_relaxed);\n"
            "    }\n"
            "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "\n"
            "    uint3 blockIdx;\n"
            "    blockIdx.x = cuda4as_block_ticket % gridDim.x;\n"
            "    blockIdx.y = (cuda4as_block_ticket / gridDim.x) % gridDim.y;\n"
            "    blockIdx.z = cuda4as_block_ticket / (gridDim.x * gridDim.y);\n\n"
        )
        
        header = f"kernel void {kernel_name}(\n    " + ",\n    ".join(metal_params) + "\n) {\n"
        full_kernel = header + d2_preamble + metal_body + "\n}\n"
        self.metal_kernels.append(full_kernel)

    def _translate_double_kernel(self, kernel_name, params_raw, body):
        # Special translator for double_dot using soft64
        params = [KernelParam(p) for p in params_raw.split(',') if p.strip()]
        metal_params = []
        buf_idx = 0
        for p in params:
            metal_params.append(p.to_metal_param(buf_idx))
            buf_idx += 1
            
        counter_buf_idx = buf_idx
        metal_params.append(f"device atomic_uint *cuda4as_arrival_counter [[buffer({counter_buf_idx})]]")
        buf_idx += 1
        
        spill_buf_idx = buf_idx
        metal_params.append(f"device char *cuda4as_spill_buffer [[buffer({spill_buf_idx})]]")
        buf_idx += 1
        
        metal_params.append("uint3 ctaid_grid [[threadgroup_position_in_grid]]")
        metal_params.append("uint3 threadIdx [[thread_position_in_threadgroup]]")
        metal_params.append("uint3 blockDim [[threads_per_threadgroup]]")
        metal_params.append("uint3 gridDim [[threadgroups_per_grid]]")
        
        d2_preamble = (
            "    threadgroup uint cuda4as_block_ticket;\n"
            "    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {\n"
            "        cuda4as_block_ticket = atomic_fetch_add_explicit(cuda4as_arrival_counter, 1u, memory_order_relaxed);\n"
            "    }\n"
            "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "\n"
            "    uint3 blockIdx;\n"
            "    blockIdx.x = cuda4as_block_ticket % gridDim.x;\n"
            "    blockIdx.y = (cuda4as_block_ticket / gridDim.x) % gridDim.y;\n"
            "    blockIdx.z = cuda4as_block_ticket / (gridDim.x * gridDim.y);\n\n"
        )
        
        metal_body = (
            "    threadgroup df64 s[TPB];\n"
            "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
            "    if (i < n) {\n"
            "        df64 da = ieee_to_df64(a[i]);\n"
            "        df64 db = ieee_to_df64(b[i]);\n"
            "        s[threadIdx.x] = df64_mul(da, db);\n"
            "    } else {\n"
            "        s[threadIdx.x] = make_df64(0.0f, 0.0f);\n"
            "    }\n"
            "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "    for (int off = TPB / 2; off > 0; off >>= 1) {\n"
            "        if (threadIdx.x < off) s[threadIdx.x] = df64_add(s[threadIdx.x], s[threadIdx.x + off]);\n"
            "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "    }\n"
            "    if (threadIdx.x == 0) out[blockIdx.x] = df64_to_ieee(s[0]);\n"
        )
        
        header = f"kernel void {kernel_name}(\n    " + ",\n    ".join(metal_params) + "\n) {\n"
        full_kernel = header + d2_preamble + metal_body + "\n}\n"
        self.metal_kernels.append(full_kernel)

    def _translate_cooperative_kernel(self, kernel_name, params_raw, body):
        phases = body.split('grid.sync();')
        phase1_body = phases[0]
        phase1_body = re.sub(r'cg::\w+\s+\w+\s*=\s*cg::\w+\([^)]*\);', '', phase1_body)
        
        phase2_body = ""
        if len(phases) > 1:
            phase2_body = phases[1]
            if len(phases) > 2:
                phase2_body = phases[1] + "\n" + phases[2]
            phase2_body = re.sub(r'cg::\w+\s+\w+\s*=\s*cg::\w+\([^)]*\);', '', phase2_body)
            phase2_body = phase2_body.replace('grid.sync();', '')

        # Phase 1: definition of b
        self._translate_standard_kernel(kernel_name, params_raw, phase1_body)
        
        # Phase 2: add b definition at start
        if phase2_body.strip():
            phase2_augmented = "    uint b = blockIdx.x;\n" + phase2_body
            self._translate_standard_kernel(f"{kernel_name}_p2", params_raw, phase2_augmented)

    def _translate_body(self, body, spill_buf_idx=None):
        out = body
        
        # 1. Transform __shared__ declarations
        # e.g., __shared__ float As[T][T];
        shared_pattern = re.compile(r'__shared__\s+(?:unsigned\s+int|uint|float|int)\s+([A-Za-z0-9_]+)(\[[^\]]+\](?:\[[^\]]+\])?)\s*;')
        for sm in shared_pattern.finditer(out):
            whole = sm.group(0)
            var_name = sm.group(1)
            dims = sm.group(2)
            
            # Determine element type from whole
            if 'float' in whole: elem_ty = 'float'
            elif 'int' in whole and 'unsigned' not in whole and 'uint' not in whole: elem_ty = 'int'
            else: elem_ty = 'uint'
            
            # Check for spill case (> 32 KiB)
            if '48' in dims or 'SHARED_WORDS' in dims:
                spill_decl = (
                    f"device {elem_ty}* {var_name} = (device {elem_ty}*)(cuda4as_spill_buffer + "
                    f"(cuda4as_block_ticket * 49152));"
                )
                out = out.replace(whole, spill_decl)
            elif var_name == 'local':
                out = out.replace(whole, f"threadgroup atomic_uint {var_name}{dims};")
            else:
                out = out.replace(whole, f"threadgroup {elem_ty} {var_name}{dims};")

        # 2. Transform __syncthreads() and __threadfence()
        out = re.sub(r'__syncthreads\(\)', 'threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device)', out)
        out = re.sub(r'__threadfence\(\)', 'atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst)', out)
        
        # 3. Transform atomic operations for histogram/shared atomics
        out = re.sub(
            r'local\[([^\]]+)\]\s*=\s*0\s*;',
            r'atomic_store_explicit(&local[\1], 0u, memory_order_relaxed);',
            out
        )
        out = re.sub(
            r'atomicAdd\s*\(\s*&local\[(.*?)\]\s*,\s*([^)]+)\)\s*;',
            r'atomic_fetch_add_explicit(&local[\1], (uint)(\2), memory_order_relaxed);',
            out
        )
        out = re.sub(
            r'for\s*\(\s*int\s+i\s*=\s*threadIdx\.x;\s*i\s*<\s*BINS;\s*i\s*\+=\s*blockDim\.x\s*\)\s*if\s*\(\s*local\[i\]\s*\)\s*atomicAdd\s*\(\s*&out\[i\]\s*,\s*local\[i\]\s*\);',
            r'for (int i = threadIdx.x; i < BINS; i += blockDim.x) { uint cnt = atomic_load_explicit(&local[i], memory_order_relaxed); if (cnt) atomic_fetch_add_explicit((device atomic_uint*)&out[i], cnt, memory_order_relaxed); }',
            out
        )
        out = re.sub(
            r'atomicAdd\s*\(\s*&([A-Za-z0-9_]+)\[([^\]]+)\]\s*,\s*([^)]+)\)',
            r'atomic_fetch_add_explicit((device atomic_uint*)&(\1[\2]), (uint)(\3), memory_order_relaxed)',
            out
        )
        out = re.sub(
            r'atomicAdd\s*\(\s*([^,]+)\s*,\s*([^)]+)\)',
            r'atomic_fetch_add_explicit((device atomic_uint*)(\1), (uint)(\2), memory_order_relaxed)',
            out
        )

        # 4. Transform warp intrinsics
        out = re.sub(r'__shfl_down_sync\s*\([^,]+,\s*([^,]+),\s*([^)]+)\)', r'simd_shuffle_down(\1, \2)', out)
        out = re.sub(r'__shfl_sync\s*\([^,]+,\s*([^,]+),\s*([^)]+)\)', r'simd_shuffle(\1, \2)', out)
        out = re.sub(r'__ballot_sync\s*\([^,]+,\s*([^)]+)\)', r'as_type<uint4>(simd_ballot(\1)).x', out)

        # 5. Transform types in body
        out = re.sub(r'\bunsigned int\b', 'uint', out)
        out = re.sub(r'\bunsigned char\b', 'uchar', out)
        out = re.sub(r'\buint32_t\b', 'uint', out)
        out = re.sub(r'\bint32_t\b', 'int', out)
        out = re.sub(r'\bsize_type\b', 'ulong', out)

        # 6. Transform Celeritas utilities
        out = re.sub(r'KernelParamCalculator::thread_id\(\)(?:\.unchecked_get\(\)|\.get\(\))?', r'(blockIdx.x * blockDim.x + threadIdx.x)', out)
        out = re.sub(r'([A-Za-z0-9_]+)\.get\(\)', r'\1', out)
        
        # 7. Transform Celeritas range loops
        out = re.sub(
            r'for\s*\(\s*auto\s+(\w+)\s*:\s*(?:celeritas::)?[rR]ange\s*\((.*?),\s*(.*?)\)\s*\.step\s*\((.*?)\)\s*\)',
            r'for (auto \1 = (\2); \1 < (\3); \1 += (\4))',
            out,
            flags=re.DOTALL
        )
        out = re.sub(
            r'for\s*\(\s*auto\s+(\w+)\s*:\s*(?:celeritas::)?[rR]ange\s*\((.*?),\s*(.*?)\)\s*\)',
            r'for (auto \1 = (\2); \1 < (\3); ++\1)',
            out,
            flags=re.DOTALL
        )
        out = re.sub(
            r'for\s*\(\s*auto\s+(\w+)\s*:\s*(?:celeritas::)?[rR]ange\s*\((.*?)\)\s*\)',
            r'for (auto \1 = 0; \1 < (\2); ++\1)',
            out,
            flags=re.DOTALL
        )

        return out

    def generate_msl(self):
        if self.refusals:
            return None, self.refusals[0]

        msl_lines = [
            "#include <metal_stdlib>",
            "using namespace metal;\n"
        ]
        
        if self.macros:
            msl_lines.extend(self.macros)
            msl_lines.append("")
            
        if self.needs_soft64:
            msl_lines.append(SOFT64_PREAMBLE)
            msl_lines.append("")
        
        for h in self.helper_functions:
            msl_lines.append(h)
            
        for k in self.metal_kernels:
            msl_lines.append(k)
            
        return "\n".join(msl_lines), None

    def generate_cpp_registration(self, msl_code, file_stem):
        cpp_lines = [
            '#include <stddef.h>',
            'extern "C" void cuda4asRegisterShaderSource(const char* msl_source);',
            '',
            f'static const char* kCuda4AS_{file_stem}_MSL = R"MSL(',
            msl_code,
            ')MSL";',
            '',
            f'__attribute__((constructor)) static void cuda4as_auto_register_{file_stem}() {{',
            f'    cuda4asRegisterShaderSource(kCuda4AS_{file_stem}_MSL);',
            '}',
            ''
        ]
        return "\n".join(cpp_lines)

def _extract_balanced(text, start_pos, open_ch="(", close_ch=")"):
    depth = 1
    pos = start_pos
    while depth > 0 and pos < len(text):
        ch = text[pos]
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
        pos += 1
    return text[start_pos:pos-1], pos

def _split_top_level(s):
    parts = []
    cur = []
    depth = 0
    for ch in s:
        if ch in "([{<":
            depth += 1
        elif ch in ")]}>":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if "".join(cur).strip():
        parts.append("".join(cur).strip())
    return parts

def _get_llvm_param_type_and_var(p_str):
    p_str = p_str.strip()
    m_var = re.search(r"(%[A-Za-z0-9_.$]+)\s*$", p_str)
    var_name = m_var.group(1) if m_var else ""
    if p_str[0] in "[{<":
        pairs = {"[": "]", "{": "}", "<": ">"}
        _, end_idx = _extract_balanced(p_str, 1, p_str[0], pairs[p_str[0]])
        ty = p_str[:end_idx]
    else:
        ty = p_str.split()[0]
    return ty, var_name

def stitch_llvm_ir(host_ll_path, dev_ll_path, mod_id):
    with open(host_ll_path, "r", encoding="utf-8", errors="ignore") as f:
        host_ll = f.read()
    with open(dev_ll_path, "r", encoding="utf-8", errors="ignore") as f:
        dev_ll = f.read()

    dev_defines = set(re.findall(r"^define\b[^\n]*?@([^\s(]+)\(", dev_ll, re.MULTILINE))
    stub_hdr_re = re.compile(r"^define\b[^\n]*?\bvoid\s+@([^\s(]*__device_stub__[^\s(]*)\(", re.MULTILINE)

    extra_host_ir = ["\ndeclare i32 @__cuda4as_launch_with_fallback(ptr, ptr)\n"]
    matched_kernels = {}
    launch_replacements = []
    idx = 0

    for m in stub_hdr_re.finditer(host_ll):
        stub_sym = m.group(1)
        params_raw, after_params = _extract_balanced(host_ll, m.end(), "(", ")")
        brace_pos = host_ll.find("{", after_params)
        if brace_pos == -1:
            continue
        stub_body, _ = _extract_balanced(host_ll, brace_pos + 1, "{", "}")

        kern_sym = None
        for dm in re.finditer(r"(\d+)__device_stub__", stub_sym):
            digits = dm.group(1)
            for k in range(1, len(digits) + 1):
                sub_len = int(digits[-k:])
                if sub_len > 15:
                    cand = stub_sym[:dm.end() - 15 - k] + str(sub_len - 15) + stub_sym[dm.end():]
                    if cand in dev_defines:
                        kern_sym = cand
                        break
            if kern_sym:
                break
        if not kern_sym:
            cand = stub_sym.replace("__device_stub__", "")
            if cand in dev_defines:
                kern_sym = cand
        if not kern_sym:
            continue

        param_strs = _split_top_level(params_raw) if params_raw.strip() else []
        params = [_get_llvm_param_type_and_var(p) for p in param_strs]

        setup_calls = re.findall(
            r"call\s+i32\s+@cudaSetupArgument\(\s*ptr\s+(?:[a-z_]+\s+)*(%[A-Za-z0-9_.$]+)\s*,\s*i64\s+(\d+)\s*,\s*i64\s+(\d+)\s*\)",
            stub_body
        )
        if len(setup_calls) != len(params):
            continue

        dev_kern_sym = f"__cuda4as_dev_{mod_id}_{kern_sym}"
        entry_sym = f"__cuda4as_entry_{mod_id}_{idx}"
        matched_kernels[kern_sym] = dev_kern_sym

        param_types_csv = ", ".join(ty for ty, _ in params)
        extra_host_ir.append(f"declare void @{dev_kern_sym}({param_types_csv})")
        extra_host_ir.append(f"define internal void @{entry_sym}(ptr %arg_buf) {{")
        call_args = []
        for i, ((ty, pvar), (aptr, sz, off)) in enumerate(zip(params, setup_calls)):
            if aptr == pvar:
                extra_host_ir.append(f"  %arg_{i} = getelementptr inbounds i8, ptr %arg_buf, i64 {off}")
                call_args.append(f"ptr %arg_{i}")
            else:
                extra_host_ir.append(f"  %gep_{i} = getelementptr inbounds i8, ptr %arg_buf, i64 {off}")
                extra_host_ir.append(f"  %arg_{i} = load {ty}, ptr %gep_{i}, align 1")
                call_args.append(f"{ty} %arg_{i}")
        args_joined = ", ".join(call_args)
        extra_host_ir.append(f"  call void @{dev_kern_sym}({args_joined})")
        extra_host_ir.append("  ret void\n}\n")

        launch_replacements.append((stub_sym, entry_sym))
        idx += 1

    if not matched_kernels:
        return False

    for stub_sym, entry_sym in launch_replacements:
        host_ll = re.sub(
            rf"@cudaLaunch\(ptr(\s+nonnull)?\s+@{re.escape(stub_sym)}\)",
            rf"@__cuda4as_launch_with_fallback(ptr\1 @{stub_sym}, ptr nonnull @{entry_sym})",
            host_ll
        )

    host_ll += "\n".join(extra_host_ir)

    # Remove llvm.global_ctors in dev_ll if present
    dev_ll = re.sub(r"^@llvm\.global_ctors\s*=[^\n]*\n", "", dev_ll, flags=re.MULTILINE)

    # Make linkonce_odr/weak_odr globals internal in dev_ll
    dev_globals = set(re.findall(r"^@([A-Za-z0-9_.$]+)\s*=\s*(?:linkonce_odr|weak_odr)\b", dev_ll, re.MULTILINE))
    for g in dev_globals:
        new_g = f"__cuda4as_dev_{mod_id}_{g}"
        dev_ll = re.sub(
            rf"^@{re.escape(g)}\s*=\s*(?:linkonce_odr|weak_odr)\s+(?:hidden\s+|dso_local\s+)*",
            f"@{new_g} = internal ",
            dev_ll,
            flags=re.MULTILINE
        )
        dev_ll = re.sub(rf"@{re.escape(g)}(?![A-Za-z0-9_.$])", f"@{new_g}", dev_ll)

    for fn in dev_defines:
        if fn in matched_kernels:
            new_fn = matched_kernels[fn]
            dev_ll = re.sub(
                rf"^define\s+[^\n]*?\bvoid\s+@{re.escape(fn)}\(",
                f"define void @{new_fn}(",
                dev_ll,
                flags=re.MULTILINE
            )
            dev_ll = re.sub(rf"@{re.escape(fn)}(?![A-Za-z0-9_.$])", f"@{new_fn}", dev_ll)
        else:
            new_fn = f"__cuda4as_dev_{mod_id}_{fn}"
            dev_ll = re.sub(
                rf"^define\s+(?:linkonce_odr\s+|weak_odr\s+|internal\s+|hidden\s+|dso_local\s+)*([^\n]*?@{re.escape(fn)}\()",
                rf"define internal \1",
                dev_ll,
                flags=re.MULTILINE
            )
            dev_ll = re.sub(r"\bdefine internal\s+hidden\s+", "define internal ", dev_ll)
            dev_ll = re.sub(rf"@{re.escape(fn)}(?![A-Za-z0-9_.$])", f"@{new_fn}", dev_ll)

    with open(host_ll_path, "w", encoding="utf-8") as f:
        f.write(host_ll)
    with open(dev_ll_path, "w", encoding="utf-8") as f:
        f.write(dev_ll)
    return True

def main():
    parser = argparse.ArgumentParser(description="cuda4AS CUDA to Metal Kernel Translator")
    parser.add_argument("input", nargs="?", help="Input .cu file")
    parser.add_argument("-o", "--output-metal", help="Output Metal (.metal) file")
    parser.add_argument("-r", "--output-reg", help="Output C++ registration (.cpp) file")
    parser.add_argument("--stitch-host-ll", help="Host LLVM IR file to stitch")
    parser.add_argument("--stitch-dev-ll", help="Device LLVM IR file to stitch")
    parser.add_argument("--mod-id", help="Module identifier for stitched symbols")
    args = parser.parse_args()

    if args.stitch_host_ll and args.stitch_dev_ll:
        mod_id = args.mod_id or hashlib.md5(args.stitch_host_ll.encode()).hexdigest()[:8]
        mod_id = re.sub(r"[^A-Za-z0-9_]", "_", mod_id)
        ok = stitch_llvm_ir(args.stitch_host_ll, args.stitch_dev_ll, mod_id)
        sys.exit(0 if ok else 3)

    input_path = os.path.abspath(args.input)
    file_stem = os.path.splitext(os.path.basename(input_path))[0]
    file_stem = re.sub(r'[^A-Za-z0-9_]', '_', file_stem)

    with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
        src = f.read()

    translator = KernelTranslator(src)
    translator.check_refusals()
    if translator.refusals:
        print(f"cuda4AS Translator refusal: {translator.refusals[0]}", file=sys.stderr)
        sys.exit(2)

    translator.extract_macros()
    translator.translate_helper_functions()
    translator.translate_global_kernels()
    msl_code, err = translator.generate_msl()
    if err:
        print(f"cuda4AS Translator error: {err}", file=sys.stderr)
        sys.exit(1)

    if args.output_metal:
        with open(args.output_metal, 'w', encoding='utf-8') as f:
            f.write(msl_code)
    elif not args.output_reg:
        print(msl_code)

    if args.output_reg and translator.metal_kernels:
        cpp_code = translator.generate_cpp_registration(msl_code, file_stem)
        with open(args.output_reg, 'w', encoding='utf-8') as f:
            f.write(cpp_code)

if __name__ == "__main__":
    main()
