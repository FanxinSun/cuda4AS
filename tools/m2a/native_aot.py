#!/usr/bin/env python3
"""Small CUDA vector-add Native AOT compiler used by the M2A slice.

The implementation has a narrow capability surface. The pinned Clang CUDA
frontend consumes the unchanged source through the tracked facade; its typed
AST is the authoritative device semantic input. The importer lowers those
compiler records into versioned SSA-like IR, deterministic MSL, and a real
device-link record. The source tokenizer remains only a host-launch diagnostic
aid (or an explicitly NOT_RUN portable packaging path) and never silently
supplies device semantics when compiler output is required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Sequence

# The Mac runner invokes this file by absolute path from an arbitrary working
# directory. Make the extracted package root importable so the sibling
# compiler-output importer works without relying on the caller's PYTHONPATH.
_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))


IR_SCHEMA = "cuda4as-device-ssa-ir-v1"
ABI_SCHEMA = "cuda4as-native-aot-abi-v1"
LINK_SCHEMA = "cuda4as-device-link-v1"
DRIVER_SCHEMA = "cuda4as-native-aot-driver-v1"
FACADE_MANIFEST_SCHEMA = "cuda4as-clang-cuda-facade-manifest-v1"
FACADE_BUILTIN_SCHEMA = "cuda4as-clang-cuda-facade-builtins-v1"
FACADE_BUILTIN_NAMES = ("blockIdx", "blockDim", "threadIdx")
FACADE_VERSION = "1.0.0"
FACADE_HEADER_REL = "tools/m2a/clang_cuda_facade/cuda_runtime.h"
FACADE_MANIFEST_REL = "tools/m2a/clang_cuda_facade/manifest.json"
FACADE_ARGUMENT_TEMPLATE = [
    "-x",
    "cuda",
    "--cuda-device-only",
    "--cuda-gpu-arch=sm_86",
    "-nocudainc",
    "-nocudalib",
    "-I",
    "<FACADE_DIR>",
    "-Xclang",
    "-ast-dump=json",
    "-fsyntax-only",
    "<SOURCE>",
]
EXPECTED_OUTPUT_SHA256 = "ed551637cf393112d0093037a0b41b9d1e9bd213c6037e8efcde30cf480f0332"
EXPECTED_OUTPUT_BYTES = 4_194_304


class AOTError(ValueError):
    """A deterministic, user-facing unsupported-source or contract error."""


@dataclass(frozen=True)
class Token:
    text: str
    line: int
    column: int
    kind: str = "token"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, indent=2, separators=(",", ": ")) + "\n"


def load_facade_contract(facade_dir: Path) -> dict[str, Any]:
    """Load and verify the tracked CUDA facade contract.

    The manifest is intentionally separate from the header so a generated
    work record can bind the exact bytes, supported Clang range, target, and
    frontend arguments without relying on a mutable host installation.
    """

    manifest_path = facade_dir / "manifest.json"
    header_path = facade_dir / "cuda_runtime.h"
    try:
        raw = manifest_path.read_bytes()
        manifest = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AOTError(f"cannot load tracked Clang CUDA facade manifest: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema") != FACADE_MANIFEST_SCHEMA:
        raise AOTError("tracked Clang CUDA facade manifest has an unsupported schema")
    if manifest.get("version") != FACADE_VERSION:
        raise AOTError("tracked Clang CUDA facade manifest version is not 1.0.0")
    header = manifest.get("header")
    if not isinstance(header, dict) or header.get("path") != FACADE_HEADER_REL:
        raise AOTError("tracked Clang CUDA facade manifest has the wrong header path")
    expected_header_sha = header.get("sha256")
    if not isinstance(expected_header_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_header_sha):
        raise AOTError("tracked Clang CUDA facade manifest has no valid header hash")
    if not header_path.is_file():
        raise AOTError(f"tracked Clang CUDA facade header is missing: {header_path}")
    actual_header_sha = sha256_file(header_path)
    if actual_header_sha != expected_header_sha:
        raise AOTError("tracked Clang CUDA facade header hash does not match its manifest")
    header_text = header_path.read_text(encoding="utf-8")
    version_marker = f'#define CUDA4AS_CLANG_CUDA_FACADE_VERSION "{FACADE_VERSION}"'
    if version_marker not in header_text:
        raise AOTError("tracked Clang CUDA facade header does not declare its manifest version")

    builtin_contract = manifest.get("builtin_contract")
    if not isinstance(builtin_contract, dict) or builtin_contract.get("schema") != FACADE_BUILTIN_SCHEMA:
        raise AOTError("tracked Clang CUDA facade manifest has no builtin declaration contract")
    device_builtin_semantics = builtin_contract.get("device_builtin_semantics")
    if device_builtin_semantics != {
        "spelling": "device_builtin",
        "status": "IGNORED_NON_AST",
        "ast_node": 0,
        "ast_json_child": False,
    }:
        raise AOTError("tracked Clang CUDA facade manifest has an invalid device_builtin serialization contract")
    if (
        builtin_contract.get("header_path") != FACADE_HEADER_REL
        or builtin_contract.get("header_sha256") != actual_header_sha
    ):
        raise AOTError("tracked Clang CUDA facade builtin contract is not bound to the header")
    declarations = builtin_contract.get("declarations")
    if not isinstance(declarations, dict) or set(declarations) != set(FACADE_BUILTIN_NAMES):
        raise AOTError("tracked Clang CUDA facade builtin contract has the wrong declaration set")
    header_lines = header_text.splitlines()
    for builtin_name in FACADE_BUILTIN_NAMES:
        declaration = declarations.get(builtin_name)
        if not isinstance(declaration, dict):
            raise AOTError(f"tracked Clang CUDA facade contract has no {builtin_name} declaration")
        line = declaration.get("line")
        column = declaration.get("column")
        if not isinstance(line, int) or line <= 0 or not isinstance(column, int) or column <= 0:
            raise AOTError(f"tracked Clang CUDA facade contract has invalid {builtin_name} location")
        if declaration.get("qual_type") != "const dim3" or not isinstance(declaration.get("source_anchor"), str):
            raise AOTError(f"tracked Clang CUDA facade contract has invalid {builtin_name} type/anchor")
        if line > len(header_lines) or header_lines[line - 1].strip() != declaration["source_anchor"]:
            raise AOTError(f"tracked Clang CUDA facade contract anchor does not match {builtin_name} header source")

    supported = manifest.get("supported_clang")
    if not isinstance(supported, dict):
        raise AOTError("tracked Clang CUDA facade manifest has no Clang support range")
    minimum_major = supported.get("minimum_major")
    maximum_major_exclusive = supported.get("maximum_major_exclusive")
    intended_major = supported.get("intended_major")
    validation_status = supported.get("validation_status")
    if not all(isinstance(value, int) for value in (minimum_major, maximum_major_exclusive, intended_major)):
        raise AOTError("tracked Clang CUDA facade manifest has invalid Clang major bounds")
    if minimum_major >= maximum_major_exclusive or not minimum_major <= intended_major < maximum_major_exclusive:
        raise AOTError("tracked Clang CUDA facade manifest has inconsistent Clang major bounds")
    if (
        supported.get("family") != "LLVM Clang CUDA frontend"
        or not isinstance(supported.get("intended_identity"), str)
        or validation_status not in {"NOT_RUN", "VALIDATED"}
    ):
        raise AOTError("tracked Clang CUDA facade manifest has no honest intended Clang identity/status")

    target = manifest.get("target")
    if not isinstance(target, dict) or target.get("cuda_gpu_arch") != "sm_86":
        raise AOTError("tracked Clang CUDA facade manifest has an unsupported target architecture")
    arguments = manifest.get("compiler_arguments")
    if arguments != FACADE_ARGUMENT_TEMPLATE:
        raise AOTError("tracked Clang CUDA facade manifest compiler arguments are not the pinned contract")
    return {
        "manifest": manifest,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_bytes(raw),
        "header_path": str(header_path),
        "header_sha256": actual_header_sha,
        "version": FACADE_VERSION,
        "supported_clang": supported,
        "target": target,
        "compiler_arguments": list(arguments),
        "builtin_contract": builtin_contract,
    }


def materialize_facade_arguments(source: Path, facade_dir: Path) -> list[str]:
    """Resolve the manifest's placeholders for one unchanged source file."""

    contract = load_facade_contract(facade_dir)
    return [
        str(facade_dir) if item == "<FACADE_DIR>" else str(source) if item == "<SOURCE>" else item
        for item in contract["compiler_arguments"]
    ]


def tokenize(source: str) -> list[Token]:
    """Tokenize C++ CUDA syntax while retaining source locations."""

    tokens: list[Token] = []
    i = 0
    line = 1
    col = 1
    n = len(source)
    multi = ("<<<", ">>>", "<<=", ">>=", "==", "!=", "<=", ">=", "&&", "||", "->", "++", "--", "+=", "-=", "*=", "/=")
    while i < n:
        c = source[i]
        if c in " \t\r\f\v":
            i += 1
            col += 1
            continue
        if c == "\n":
            i += 1
            line += 1
            col = 1
            continue
        if source.startswith("//", i):
            while i < n and source[i] != "\n":
                i += 1
                col += 1
            continue
        if source.startswith("/*", i):
            i += 2
            col += 2
            while i < n and not source.startswith("*/", i):
                if source[i] == "\n":
                    line += 1
                    col = 1
                    i += 1
                else:
                    i += 1
                    col += 1
            if i >= n:
                raise AOTError("unterminated block comment")
            i += 2
            col += 2
            continue
        if c == "#":
            # Preprocessor directives are retained in the original source for
            # host metadata discovery but are not kernel-language tokens.
            while i < n and source[i] != "\n":
                i += 1
                col += 1
            continue
        if c.isalpha() or c == "_":
            start_line, start_col, start = line, col, i
            i += 1
            col += 1
            while i < n and (source[i].isalnum() or source[i] == "_"):
                i += 1
                col += 1
            tokens.append(Token(source[start:i], start_line, start_col, "identifier"))
            continue
        if c.isdigit():
            start_line, start_col, start = line, col, i
            i += 1
            col += 1
            while i < n and (source[i].isalnum() or source[i] in ".'_"):
                i += 1
                col += 1
            tokens.append(Token(source[start:i], start_line, start_col, "number"))
            continue
        matched = next((op for op in multi if source.startswith(op, i)), None)
        if matched:
            tokens.append(Token(matched, line, col))
            i += len(matched)
            col += len(matched)
            continue
        if c in "{}()[];,.*+/%<>=!&|~-?:":
            tokens.append(Token(c, line, col))
            i += 1
            col += 1
            continue
        if c in '"\'':
            quote = c
            start_line, start_col, start = line, col, i
            i += 1
            col += 1
            while i < n:
                if source[i] == "\\":
                    i += 2
                    col += 2
                    continue
                if source[i] == quote:
                    i += 1
                    col += 1
                    break
                if source[i] == "\n":
                    raise AOTError("newline in string or character literal")
                i += 1
                col += 1
            tokens.append(Token(source[start:i], start_line, start_col, "literal"))
            continue
        raise AOTError(f"unsupported character {c!r} at {line}:{col}")
    return tokens


def _loc(tok: Token) -> dict[str, int]:
    return {"line": tok.line, "column": tok.column}


class KernelParser:
    """Parser for the explicitly supported CUDA kernel subset."""

    def __init__(self, source: str):
        self.source = source
        self.ts = tokenize(source)
        self.i = 0

    def peek(self, offset: int = 0) -> Token | None:
        j = self.i + offset
        return self.ts[j] if j < len(self.ts) else None

    def take(self, text: str | None = None) -> Token:
        tok = self.peek()
        if tok is None:
            raise AOTError(f"unexpected end of source; expected {text or 'token'}")
        if text is not None and tok.text != text:
            raise AOTError(f"expected {text!r} at {tok.line}:{tok.column}, got {tok.text!r}")
        self.i += 1
        return tok

    def find_kernel(self) -> tuple[dict[str, Any], dict[str, Any]]:
        while self.peek() and self.peek().text != "__global__":
            self.i += 1
        if not self.peek():
            raise AOTError("no __global__ kernel found")
        marker = self.take("__global__")
        if self.peek() and self.peek().text in {"static", "inline"}:
            self.take()
        ret = self.take()
        if ret.text != "void":
            raise AOTError("only void __global__ kernels are supported")
        name = self.take()
        if name.kind != "identifier":
            raise AOTError("kernel name is not an identifier")
        self.take("(")
        params = self.parse_params()
        self.take(")")
        body_open = self.take("{")
        statements = self.parse_block_contents()
        body_close = self.take("}")
        kernel = {
            "name": name.text,
            "return_type": "void",
            "params": params,
            "body": statements,
            "source_location": _loc(marker),
            "body_location": {"start": _loc(body_open), "end": _loc(body_close)},
        }
        host = self.parse_host_metadata()
        return kernel, host

    def parse_params(self) -> list[dict[str, Any]]:
        params: list[dict[str, Any]] = []
        if self.peek() and self.peek().text == ")":
            return params
        while True:
            const = False
            if self.peek() and self.peek().text == "const":
                self.take()
                const = True
            typ = self.take()
            if typ.text not in {"float", "int", "unsigned", "uint", "uint32_t"}:
                raise AOTError(f"unsupported kernel parameter type {typ.text!r}")
            pointer = False
            if self.peek() and self.peek().text == "*":
                self.take()
                pointer = True
            pname = self.take()
            if pname.kind != "identifier":
                raise AOTError("kernel parameter name is not an identifier")
            if pointer and typ.text not in {"float", "int", "unsigned", "uint", "uint32_t"}:
                raise AOTError("unsupported pointer element type")
            params.append({
                "index": len(params),
                "name": pname.text,
                "type": "u32" if typ.text in {"unsigned", "uint", "uint32_t"} else typ.text,
                "pointer": pointer,
                "const": const,
                "address_space": "global" if pointer else "constant",
                "source_location": _loc(typ),
            })
            if not self.peek() or self.peek().text != ",":
                break
            self.take(",")
        return params

    def parse_block_contents(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        while self.peek() and self.peek().text != "}":
            result.append(self.parse_statement())
        if not self.peek():
            raise AOTError("unterminated kernel body")
        return result

    def parse_statement(self) -> dict[str, Any]:
        tok = self.peek()
        if tok is None:
            raise AOTError("unexpected end in statement")
        if tok.text == "if":
            self.take()
            self.take("(")
            cond = self.parse_expr()
            self.take(")")
            if self.peek() and self.peek().text == "{":
                self.take("{")
                body = self.parse_block_contents()
                self.take("}")
            else:
                body = [self.parse_statement()]
            if self.peek() and self.peek().text == "else":
                raise AOTError("else control flow is outside Native AOT Core v1")
            return {"kind": "if", "condition": cond, "then": body, "source_location": _loc(tok)}
        if tok.text in {"int", "float", "unsigned", "uint", "uint32_t"}:
            typ = self.take()
            name = self.take()
            if name.kind != "identifier":
                raise AOTError("local declaration name is not an identifier")
            init = None
            if self.peek() and self.peek().text == "=":
                self.take("=")
                init = self.parse_expr()
            self.take(";")
            return {"kind": "decl", "type": "u32" if typ.text in {"unsigned", "uint", "uint32_t"} else typ.text, "name": name.text, "init": init, "source_location": _loc(typ)}
        lhs = self.parse_expr()
        op = self.take()
        if op.text not in {"=", "+=", "-=", "*=", "/="}:
            raise AOTError(f"unsupported statement operator {op.text!r}")
        rhs = self.parse_expr()
        self.take(";")
        return {"kind": "assign", "lhs": lhs, "operator": op.text, "rhs": rhs, "source_location": _loc(op)}

    _PRECEDENCE = {
        "||": 1, "&&": 2, "|": 3, "^": 4, "&": 5,
        "==": 6, "!=": 6, "<": 7, "<=": 7, ">": 7, ">=": 7,
        "+": 8, "-": 8, "*": 9, "/": 9, "%": 9,
    }

    def parse_expr(self, min_prec: int = 0) -> dict[str, Any]:
        tok = self.take()
        if tok.text in {"+", "-", "!", "~"}:
            node = {"kind": "unary", "operator": tok.text, "value": self.parse_expr(10), "source_location": _loc(tok)}
        elif tok.text == "(":
            node = self.parse_expr()
            self.take(")")
        elif tok.kind == "number":
            node = {"kind": "literal", "value": tok.text, "source_location": _loc(tok)}
        elif tok.kind == "identifier":
            node = {"kind": "name", "name": tok.text, "source_location": _loc(tok)}
        else:
            raise AOTError(f"unsupported expression token {tok.text!r} at {tok.line}:{tok.column}")
        while self.peek() and self.peek().text in {"[", "."}:
            op = self.take()
            if op.text == "[":
                idx = self.parse_expr()
                self.take("]")
                node = {"kind": "index", "base": node, "index": idx, "source_location": _loc(op)}
            else:
                member = self.take()
                if member.text not in {"x", "y", "z"}:
                    raise AOTError(f"unsupported builtin member {member.text!r}")
                node = {"kind": "member", "base": node, "member": member.text, "source_location": _loc(op)}
        while self.peek() and self.peek().text in self._PRECEDENCE:
            op = self.peek().text
            prec = self._PRECEDENCE[op]
            if prec < min_prec:
                break
            self.take()
            rhs = self.parse_expr(prec + 1)
            node = {"kind": "binary", "operator": op, "left": node, "right": rhs, "source_location": _loc(self.ts[self.i - 1])}
        return node

    def parse_host_metadata(self) -> dict[str, Any]:
        # Host metadata is parsed from source tokens.  It is intentionally
        # generic: changing N, the seed, launch expression, or argument order
        # changes the manifest and generated runtime input.
        toks = self.ts
        define_n: str | None = None
        for i, tok in enumerate(toks[:-2]):
            if tok.text == "N" and toks[i + 1].text == "(" and toks[i + 2].text == "1":
                pass
        # The preprocessor line is retained by the tokenizer only as tokens;
        # recover a simple integer expression after a #define N declaration by
        # scanning the original source, without depending on a kernel name.
        m = re.search(r"(?m)^\s*#\s*define\s+([A-Za-z_]\w*)\s+\(\s*1\s*<<\s*(\d+)\s*\)", self.source)
        if m:
            define_n = str(1 << int(m.group(2)))
            n_macro = m.group(1)
        else:
            n_macro = "N"
        launch = re.search(r"([A-Za-z_]\w*)\s*<<<\s*([^,<>]+)\s*,\s*([0-9]+)\s*>>>\s*\([^;]+\)", self.source)
        if not launch:
            # Token-based fallback for formatting variations.
            launch = re.search(r"<<<\s*([^,<>]+)\s*,\s*([0-9]+)\s*>>>\s*\(", self.source)
            if launch:
                grid_expr, block = launch.group(1), launch.group(2)
                kernel_name = "kernel"
            else:
                raise AOTError("no supported CUDA launch expression found")
        else:
            kernel_name, grid_expr, block = launch.group(1), launch.group(2), launch.group(3)
        if not re.fullmatch(r"\s*\(?\s*\w+\s*\+\s*\d+\s*\)?\s*/\s*\d+\s*", grid_expr):
            raise AOTError(f"unsupported launch grid expression: {grid_expr.strip()}")
        if int(block) <= 0 or int(block) > 1024:
            raise AOTError("launch block size is outside Metal limits")
        seed = None
        sm = re.search(r"uint64_t\s+([A-Za-z_]\w*)\s*=\s*(0x[0-9A-Fa-f]+ULL?)", self.source)
        if sm:
            seed = sm.group(2).removesuffix("ULL").removesuffix("ull")
        if seed is None:
            raise AOTError("host input seed is not declared")
        call = re.search(r"<<<\s*[^,<>]+\s*,\s*[0-9]+\s*>>>\s*\(([^;]*)\)", self.source)
        if not call:
            raise AOTError("CUDA launch arguments are not observable")
        launch_args = [item.strip() for item in call.group(1).split(",") if item.strip()]
        if len(launch_args) != 4:
            raise AOTError("M2A ABI requires four launch arguments")
        calls = re.findall(r"lcg_exact_f32\s*\(\s*&\w+\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", self.source)
        if len(calls) < 2:
            raise AOTError("expected two source-driven FP32 input generators")
        if "cudaMemcpyHostToDevice" not in self.source or "cudaMemcpyDeviceToHost" not in self.source:
            raise AOTError("required host/device copy operations are absent")
        for required in ("cudaMalloc", "cudaGetLastError", "cudaDeviceSynchronize"):
            if required not in self.source:
                raise AOTError(f"required CUDA runtime operation {required} is absent")
        return {
            "n_macro": n_macro,
            "element_count": int(define_n or 0),
            "launch": {"kernel": kernel_name, "grid_expression": grid_expr.strip(), "block_x": int(block), "arguments": launch_args},
            "seed": seed,
            "input_generator": {"name": "lcg_exact_f32", "calls": [{"hi": int(a), "den": int(b)} for a, b in calls[:2]]},
            "expected_output": {"bytes": EXPECTED_OUTPUT_BYTES, "sha256": EXPECTED_OUTPUT_SHA256, "comparison": "exact"},
            "copy_contract": ["host_to_device", "host_to_device", "device_to_host"],
        }


def _type_for_expr(expr: dict[str, Any], env: dict[str, dict[str, Any]]) -> str:
    # Compiler-imported expressions carry their exact typed AST result. This
    # field is authoritative; the inference below is retained only for the
    # explicit source-parser packaging path and independent diagnostics.
    if isinstance(expr.get("type"), str):
        return expr["type"]
    kind = expr["kind"]
    if kind == "literal":
        return "int"
    if kind == "name":
        if expr["name"] in {"blockIdx", "blockDim", "threadIdx"}:
            return "u32"
        if expr["name"] not in env:
            raise AOTError(f"unknown identifier {expr['name']!r}")
        return env[expr["name"]]["type"]
    if kind == "member":
        base = expr["base"]
        if base.get("kind") != "name" or base.get("name") not in {"blockIdx", "blockDim", "threadIdx"}:
            raise AOTError("member access is only supported for CUDA x/y/z builtins")
        return "u32"
    if kind == "index":
        bt = _type_for_expr(expr["base"], env)
        if not isinstance(expr["base"], dict) or expr["base"].get("kind") != "name" or not env.get(expr["base"].get("name"), {}).get("pointer"):
            raise AOTError("index base must be a kernel global pointer")
        _type_for_expr(expr["index"], env)
        return bt.removesuffix("*")
    if kind == "unary":
        t = _type_for_expr(expr["value"], env)
        if expr["operator"] in {"!", "~"} and t not in {"int", "u32"}:
            raise AOTError("unsupported unary operand type")
        return "int" if expr["operator"] == "!" else t
    if kind == "binary":
        lt, rt = _type_for_expr(expr["left"], env), _type_for_expr(expr["right"], env)
        op = expr["operator"]
        if op in {"<", "<=", ">", ">=", "==", "!=", "&&", "||"}:
            return "bool"
        if lt == "float" or rt == "float":
            if op not in {"+", "-", "*", "/"}:
                raise AOTError(f"unsupported floating operation {op!r}")
            return "float"
        if op not in {"+", "-", "*", "/", "%", "&", "|", "^", "<<", ">>"}:
            raise AOTError(f"unsupported integer operation {op!r}")
        return "u32" if lt == "u32" or rt == "u32" else "int"
    raise AOTError(f"unsupported expression kind {kind!r}")


def verify_source_compiler_signature(source_kernel: dict[str, Any], compiler_kernel: dict[str, Any]) -> None:
    """Reject host-parser/compiler disagreements without using parser semantics.

    The source parser supplies only an independently checked kernel identity and
    parameter contract. Device expressions and statements remain owned by the
    compiler importer.
    """

    if source_kernel.get("name") != compiler_kernel.get("name") or source_kernel.get("return_type") != compiler_kernel.get("return_type"):
        raise AOTError("source/parser and compiler kernel identity disagree")
    source_params = source_kernel.get("params", [])
    compiler_params = compiler_kernel.get("params", [])
    if len(source_params) != len(compiler_params):
        raise AOTError("source/parser and compiler kernel parameter counts disagree")
    fields = ("name", "type", "pointer", "const")
    for index, (source_param, compiler_param) in enumerate(zip(source_params, compiler_params)):
        if any(source_param.get(field) != compiler_param.get(field) for field in fields):
            raise AOTError(f"source/parser and compiler parameter {index} contracts disagree")
        if source_param.get("pointer") and source_param.get("address_space") != compiler_param.get("address_space"):
            raise AOTError(f"source/parser and compiler parameter {index} address spaces disagree")


class SSABuilder:
    def __init__(self, kernel: dict[str, Any]):
        self.kernel = kernel
        self.env: dict[str, dict[str, Any]] = {
            p["name"]: {
                "type": (p["type"] + "*") if p["pointer"] else p["type"],
                "pointer": p["pointer"],
                "const": p.get("const", False),
                "address_space": p.get("address_space", "unknown"),
                "arg": p["index"],
            }
            for p in kernel["params"]
        }
        self.values: list[dict[str, Any]] = []
        self.ops: list[dict[str, Any]] = []
        self.counter = 0

    def value(self, typ: str, op: str, operands: list[str], loc: dict[str, Any], **extra: Any) -> str:
        name = f"%{self.counter}"
        self.counter += 1
        location = dict(loc)
        location.setdefault("provenance", "source_parser")
        rec = {"id": name, "type": typ, "op": op, "operands": operands, "source_location": location}
        rec.update(extra)
        self.values.append(rec)
        return name

    def lower_expr(self, expr: dict[str, Any]) -> tuple[str, str]:
        kind = expr["kind"]
        loc = expr["source_location"]
        if kind == "literal":
            raw = expr["value"].replace("'", "")
            typ = expr.get("type") or ("float" if any(x in raw.lower() for x in (".", "e", "f")) else "int")
            return self.value(typ, "constant", [], loc, literal=raw), typ
        if kind == "name":
            name = expr["name"]
            if name in {"blockIdx", "blockDim", "threadIdx"}:
                return self.value("u32x3", "builtin", [], loc, builtin=name), "u32x3"
            if name not in self.env:
                raise AOTError(f"unknown identifier {name!r}")
            entry = self.env[name]
            return self.value(
                entry["type"],
                "argument" if "arg" in entry else "local",
                [],
                loc,
                name=name,
                argument=entry.get("arg"),
                pointer=entry.get("pointer", False),
                const=entry.get("const", False),
                address_space=entry.get("address_space", "unknown"),
            ), entry["type"]
        if kind == "member":
            base, bt = self.lower_expr(expr["base"])
            if bt != "u32x3":
                raise AOTError("member base is not a CUDA builtin vector")
            typ = expr.get("type", "u32")
            return self.value(typ, "builtin_member", [base], loc, member=expr["member"]), typ
        if kind == "index":
            base, bt = self.lower_expr(expr["base"])
            idx, it = self.lower_expr(expr["index"])
            if not bt.endswith("*"):
                raise AOTError("index base is not a pointer")
            typ = expr.get("type", bt[:-1])
            address_space = expr.get("address_space", "global")
            if address_space != "global":
                raise AOTError(f"compiler/imported index has unsupported address space {address_space!r}")
            return self.value(typ, "load_global", [base, idx], loc, address_space=address_space, element_type=typ), typ
        if kind == "unary":
            val, typ = self.lower_expr(expr["value"])
            typ = expr.get("type") or _type_for_expr(expr, self.env)
            return self.value(typ, "unary_" + expr["operator"], [val], loc), typ
        if kind == "binary":
            left, _ = self.lower_expr(expr["left"])
            right, _ = self.lower_expr(expr["right"])
            typ = expr.get("type") or _type_for_expr(expr, self.env)
            return self.value(typ, "binary_" + expr["operator"], [left, right], loc), typ
        if kind == "cast":
            value, value_type = self.lower_expr(expr["value"])
            typ = expr.get("type") or _type_for_expr(expr, self.env)
            return self.value(typ, "cast", [value], loc, cast_kind=expr.get("cast_kind"), source_type=value_type), typ
        raise AOTError(f"unsupported expression {kind}")

    def lower_lvalue(self, expr: dict[str, Any]) -> tuple[str, str, str]:
        if expr["kind"] != "index" or expr["base"].get("kind") != "name":
            raise AOTError("only global pointer indexing is assignable")
        name = expr["base"]["name"]
        if not self.env.get(name, {}).get("pointer"):
            raise AOTError("assignment target is not a kernel pointer")
        address_space = self.env[name].get("address_space", "unknown")
        if address_space != "global":
            raise AOTError(f"assignment target address space is {address_space!r}, expected compiler global")
        base, bt = self.lower_expr(expr["base"])
        idx, _ = self.lower_expr(expr["index"])
        return base, idx, bt.removesuffix("*")

    def statements(self, statements: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for stmt in statements:
            kind = stmt["kind"]
            if kind == "decl":
                if stmt["name"] in self.env:
                    raise AOTError(f"duplicate local {stmt['name']!r}")
                self.env[stmt["name"]] = {"type": stmt["type"], "pointer": False}
                if stmt["init"] is None:
                    raise AOTError("uninitialized locals are outside the supported subset")
                val, typ = self.lower_expr(stmt["init"])
                if typ != stmt["type"] and not (stmt["type"] == "int" and typ == "u32"):
                    raise AOTError(f"initializer type {typ} does not match {stmt['type']}")
                self.env[stmt["name"]]["value"] = val
                out.append({"kind": "declare", "name": stmt["name"], "type": stmt["type"], "value": val, "source_location": stmt["source_location"]})
            elif kind == "assign":
                base, idx, target_type = self.lower_lvalue(stmt["lhs"])
                value, value_type = self.lower_expr(stmt["rhs"])
                if stmt["operator"] != "=" :
                    raise AOTError("compound assignment is not enabled in Native AOT Core v1")
                if value_type != target_type and not (target_type == "float" and value_type == "int"):
                    raise AOTError(f"store type {value_type} does not match {target_type}")
                out.append({"kind": "store_global", "base": base, "index": idx, "value": value, "type": target_type, "address_space": "global", "source_location": stmt["source_location"]})
            elif kind == "if":
                cond, cond_type = self.lower_expr(stmt["condition"])
                if cond_type != "bool":
                    raise AOTError("if condition must be a comparison")
                then = self.statements(stmt["then"])
                out.append({"kind": "if", "condition": cond, "then": then, "source_location": stmt["source_location"]})
            else:
                raise AOTError(f"unsupported statement kind {kind!r}")
        return out

    def build(self) -> dict[str, Any]:
        body = self.statements(self.kernel["body"])
        return {
            "schema": IR_SCHEMA,
            "version": 1,
            "module": {"name": self.kernel["name"], "source_language": "cuda", "numerical_mode": "ieee754-f32"},
            "kernel": {"name": self.kernel["name"], "return_type": "void", "params": self.kernel["params"], "source_location": self.kernel["source_location"]},
            "capabilities": ["fp32", "int32", "global_load", "global_store", "cuda_builtin_ids", "bounds_control_flow"],
            "unsupported_capabilities": ["fp64", "vf64", "atomics", "barriers", "shared_memory", "dynamic_parallelism", "cpu_fallback"],
            "values": self.values,
            "blocks": [{"id": "entry", "ops": self.ops + body}],
            "symbols": [{"name": self.kernel["name"], "kind": "kernel", "defined": True, "address_space": "code"}],
        }


def verify_ir(ir: dict[str, Any], *, require_semantic_source: bool = True) -> None:
    if ir.get("schema") != IR_SCHEMA or ir.get("version") != 1:
        raise AOTError("unsupported IR schema/version")
    semantic = ir.get("semantic_source")
    if require_semantic_source and (not isinstance(semantic, dict) or semantic.get("authoritative") is not True or semantic.get("kind") != "clang_cuda_ast_json"):
        raise AOTError("device IR has no authoritative Clang semantic source")
    if isinstance(semantic, dict) and semantic.get("authoritative") is True and semantic.get("kind") != "clang_cuda_ast_json":
        raise AOTError("device IR semantic source is not the selected Clang AST representation")
    kernel = ir.get("kernel")
    if not isinstance(kernel, dict) or not kernel.get("name"):
        raise AOTError("IR has no kernel metadata")
    params = kernel.get("params")
    if not isinstance(params, list) or len(params) != 4:
        raise AOTError("M2A ABI requires exactly four kernel parameters")
    values = ir.get("values")
    if not isinstance(values, list) or any(not isinstance(value, dict) for value in values):
        raise AOTError("IR values are malformed")
    ids = {v.get("id") for v in values}
    if None in ids or len(ids) != len(values):
        raise AOTError("IR value IDs are not unique")
    for block in ir.get("blocks", []):
        for op in block.get("ops", []):
            if op.get("kind") == "store_global" and op.get("address_space") != "global":
                raise AOTError("global store lost its address space")
    for value in values:
        location = value.get("source_location")
        if (
            not isinstance(location, dict)
            or location.get("line", 0) <= 0
            or location.get("column", 0) <= 0
            or location.get("provenance") not in ({"exact", "exact_offset", "inherited"} if require_semantic_source else {"exact", "exact_offset", "inherited", "source_parser"})
        ):
            raise AOTError("IR value has no compiler source location")
    if "cpu_fallback" not in ir.get("unsupported_capabilities", []):
        raise AOTError("IR must explicitly prohibit CPU fallback")


def verify_abi(abi: dict[str, Any]) -> None:
    if abi.get("schema") != ABI_SCHEMA or abi.get("version") != 1:
        raise AOTError("unsupported AOT ABI schema/version")
    args = abi.get("arguments")
    if not isinstance(args, list) or [a.get("index") for a in args] != list(range(len(args))):
        raise AOTError("malformed kernel ABI indices")
    if len(args) != 4 or any(a.get("kind") not in {"buffer", "scalar"} for a in args):
        raise AOTError("M2A ABI requires four typed arguments")
    mapping = abi.get("address_space_mapping")
    if (
        not isinstance(mapping, dict)
        or mapping.get("schema") != "cuda4as-address-space-lowering-v1"
        or mapping.get("status") != "EXPLICIT_PORTABLE_RULE"
        or mapping.get("rules") != [
            {"compiler_kind": "pointer", "compiler_address_space": "global", "metal_address_space": "device"},
            {"compiler_kind": "scalar", "compiler_address_space": "any", "metal_address_space": "constant"},
        ]
    ):
        raise AOTError("ABI lacks the versioned explicit CUDA-to-Metal address-space mapping")
    if any(a.get("kind") == "buffer" and a.get("address_space") != "global" for a in args):
        raise AOTError("buffer ABI lost global address space")
    if any(a.get("kind") == "buffer" and a.get("metal_address_space") != "device" for a in args):
        raise AOTError("buffer ABI lacks its explicit Metal device mapping")
    if any(a.get("kind") == "scalar" and a.get("metal_address_space") != "constant" for a in args):
        raise AOTError("scalar ABI lacks its explicit Metal constant mapping")
    if any(a.get("kind") == "buffer" and a.get("semantic_address_space") != "global" for a in args):
        raise AOTError("buffer ABI mapping disagrees with compiler address space")
    if any(a.get("kind") == "scalar" and not isinstance(a.get("semantic_address_space"), str) for a in args):
        raise AOTError("scalar ABI lacks its compiler address-space provenance")
    if abi.get("cpu_fallback") is not False or abi.get("thread_position") != "1d_grid":
        raise AOTError("ABI permits a prohibited fallback or has no launch contract")


def verify_device_link(link: dict[str, Any]) -> None:
    if link.get("schema") != LINK_SCHEMA or link.get("version") != 1:
        raise AOTError("unsupported device-link schema/version")
    if link.get("status") != "LINKED" or link.get("empty") is not False or link.get("module_count", 0) < 1:
        raise AOTError("device link is empty or not linked")
    if link.get("unresolved_symbols") or link.get("duplicate_symbols"):
        raise AOTError("device link has unresolved or duplicate symbols")
    if not link.get("resolved_symbols") or not link.get("modules"):
        raise AOTError("device link has no inspectable image symbols")
    if link.get("image_path") != "device-link-image.json" or not isinstance(link.get("image_sha256"), str) or link.get("image_bytes", 0) <= 0:
        raise AOTError("device link has no non-empty linked image record")


def _literal_to_msl(raw: str) -> str:
    raw = raw.replace("ULL", "u").replace("ull", "u").replace("UL", "u").replace("ul", "u")
    if raw.endswith("f") or raw.endswith("F"):
        return raw
    if "." in raw or "e" in raw.lower():
        return raw + "f"
    return raw


def lower_to_msl(kernel: dict[str, Any], ir: dict[str, Any], block_x: int, *, require_semantic_source: bool = True) -> tuple[str, dict[str, Any]]:
    verify_ir(ir, require_semantic_source=require_semantic_source)
    params = kernel["params"]
    if any(not p["pointer"] and p["type"] not in {"int", "u32"} for p in params):
        raise AOTError("only int/u32 scalar kernel arguments are supported")
    if any(p["pointer"] and p.get("address_space") != "global" for p in params):
        raise AOTError("compiler pointer argument lacks an explicit global address space")
    lines = ["#include <metal_stdlib>", "using namespace metal;", "", f"// cuda4as-ir-schema: {IR_SCHEMA}"]
    sig: list[str] = []
    for p in params:
        if p["pointer"]:
            c = "const device " if p["const"] else "device "
            sig.append(f"{c}{p['type']}* {p['name']} [[buffer({p['index']})]]")
        else:
            mtyp = "uint" if p["type"] == "u32" else p["type"]
            sig.append(f"constant {mtyp}& {p['name']} [[buffer({p['index']})]]")
    sig.append("uint gid [[thread_position_in_grid]]")
    lines.append(f"kernel void {kernel['name']}({', '.join(sig)}) {{")
    lines.append(f"    const uint cuda4as_block_dim_x = {block_x}u;")
    lines.append("    const uint cuda4as_block_idx_x = gid / cuda4as_block_dim_x;")
    lines.append("    const uint cuda4as_thread_idx_x = gid % cuda4as_block_dim_x;")
    lines.append("    const uint3 blockIdx = uint3(cuda4as_block_idx_x, 0u, 0u);")
    lines.append("    const uint3 blockDim = uint3(cuda4as_block_dim_x, 1u, 1u);")
    lines.append("    const uint3 threadIdx = uint3(cuda4as_thread_idx_x, 0u, 0u);")
    by_id = {v["id"]: v for v in ir["values"]}
    names: dict[str, str] = {}

    def val(vid: str) -> str:
        if vid in names:
            return names[vid]
        v = by_id[vid]
        op = v["op"]
        if op == "constant":
            out = _literal_to_msl(v["literal"])
        elif op == "argument":
            out = v["name"]
        elif op == "local":
            out = names.get(v["name"], v["name"])
        elif op == "builtin":
            out = v["builtin"]
        elif op == "builtin_member":
            out = f"{val(v['operands'][0])}.{v['member']}"
        elif op == "load_global":
            out = f"{val(v['operands'][0])}[{val(v['operands'][1])}]"
        elif op == "cast":
            inner = val(v["operands"][0])
            source_type = v.get("source_type")
            if source_type == v.get("type") or v.get("cast_kind") in {"NoOp", "LValueToRValue"}:
                out = inner
            else:
                mtyp = "uint" if v.get("type") == "u32" else v.get("type")
                out = f"static_cast<{mtyp}>({inner})"
        elif op.startswith("binary_"):
            sym = op.removeprefix("binary_")
            out = f"({val(v['operands'][0])} {sym} {val(v['operands'][1])})"
        elif op.startswith("unary_"):
            sym = op.removeprefix("unary_")
            out = f"({sym}{val(v['operands'][0])})"
        else:
            raise AOTError(f"cannot lower IR operation {op!r}")
        names[vid] = out
        return out

    def emit_ops(ops: list[dict[str, Any]], indent: str = "    ") -> None:
        for op in ops:
            if op["kind"] == "declare":
                expr = val(op["value"])
                typ = "uint" if op["type"] == "u32" else op["type"]
                if typ == "int" and by_id[op["value"]]["type"] == "u32":
                    expr = f"int({expr})"
                lines.append(f"{indent}{typ} {op['name']} = {expr};")
                names[op["name"]] = op["name"]
            elif op["kind"] == "store_global":
                lines.append(f"{indent}{val(op['base'])}[{val(op['index'])}] = {val(op['value'])};")
            elif op["kind"] == "if":
                lines.append(f"{indent}if ({val(op['condition'])}) {{")
                emit_ops(op["then"], indent + "    ")
                lines.append(f"{indent}}}")
            else:
                raise AOTError(f"unsupported lowered op {op['kind']!r}")
    emit_ops(ir["blocks"][0]["ops"])
    lines.append("}")
    msl = "\n".join(lines) + "\n"
    abi = {
        "schema": ABI_SCHEMA,
        "version": 1,
        "kernel": kernel["name"],
        "arguments": [
            {
                "index": p["index"],
                "name": p["name"],
                "kind": "buffer" if p["pointer"] else "scalar",
                "element_type": p["type"],
                # Metal's constant-buffer ABI is an explicit lowering rule;
                # the imported compiler fact remains semantic_address_space.
                "address_space": "global" if p["pointer"] else "constant",
                "metal_address_space": "device" if p["pointer"] else "constant",
                "semantic_address_space": p.get("address_space", "unknown"),
                "const": p["const"],
            }
            for p in params
        ],
        "address_space_mapping": {
            "schema": "cuda4as-address-space-lowering-v1",
            "status": "EXPLICIT_PORTABLE_RULE",
            "rules": [
                {"compiler_kind": "pointer", "compiler_address_space": "global", "metal_address_space": "device"},
                {"compiler_kind": "scalar", "compiler_address_space": "any", "metal_address_space": "constant"},
            ],
        },
        "thread_position": "1d_grid",
        "block_x": block_x,
        "numerical_mode": "ieee754-f32",
        "cpu_fallback": False,
    }
    return msl, abi


def discover_toolchain(clang: str | None = None, *, facade_contract: dict[str, Any] | None = None) -> dict[str, Any]:
    """Select a compatible LLVM Clang and record its exact identity.

    Apple Clang and arbitrary unbounded host versions are not silently treated
    as equivalent to the pinned LLVM CUDA frontend contract.
    """

    contract = facade_contract or load_facade_contract(Path(__file__).resolve().parent / "clang_cuda_facade")
    supported = contract["supported_clang"]
    candidates = [clang] if clang else ["/opt/homebrew/opt/llvm/bin/clang++", "clang++"]
    selected = next((x for x in candidates if x and shutil.which(x)), None)
    if not selected:
        raise AOTError("CUDA-capable clang++ is unavailable; no install/download is permitted")
    proc = subprocess.run([selected, "--version"], text=True, capture_output=True, check=False)
    version = proc.stdout.strip() or proc.stderr.strip()
    if proc.returncode != 0:
        raise AOTError(f"Clang version query failed for {selected}: exit {proc.returncode}")
    match = re.search(r"\bclang(?:\+\+)?\s+version\s+(\d+)", version, flags=re.IGNORECASE)
    if not match:
        raise AOTError(f"cannot determine LLVM Clang major version from {selected} --version")
    major = int(match.group(1))
    minimum = supported["minimum_major"]
    maximum = supported["maximum_major_exclusive"]
    if not minimum <= major < maximum:
        raise AOTError(f"Clang major {major} is outside the facade support range [{minimum}, {maximum})")
    return {
        "clang": selected,
        "version": version,
        "major": major,
        "supported_range": f">={minimum},<{maximum}",
        "identity": supported["family"],
        "exit_code": proc.returncode,
    }


def run_clang_device(
    clang: str,
    source: Path,
    ast: Path,
    log: Path,
    facade_dir: Path,
    *,
    facade_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    contract = facade_contract or load_facade_contract(facade_dir)
    cmd = [clang, *[
        str(facade_dir) if item == "<FACADE_DIR>" else str(source) if item == "<SOURCE>" else item
        for item in contract["compiler_arguments"]
    ]]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    ast.write_text(proc.stdout, encoding="utf-8")
    log.write_text("$ " + " ".join(cmd) + "\n" + proc.stdout + proc.stderr, encoding="utf-8")
    return {
        "command": cmd,
        "exit_code": proc.returncode,
        "stdout_sha256": sha256_bytes(proc.stdout.encode()),
        "stderr": proc.stderr[-4000:],
        "input_source": str(source),
        "facade_dir": str(facade_dir),
        "facade_manifest_sha256": contract["manifest_sha256"],
        "facade_header_sha256": contract["header_sha256"],
        "target": contract["target"]["cuda_gpu_arch"],
    }


def import_typed_compiler_output(
    ast_path: Path,
    kernel_name: str,
    *,
    source: Path | None = None,
    source_bytes: bytes | None = None,
    facade_contract: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Import the selected Clang AST as the device semantic authority."""

    from tools.m2a.clang_ast import ClangASTImportError, import_ast_file

    raw = ast_path.read_text(encoding="utf-8")
    if not raw.strip():
        raise AOTError("Clang produced no typed AST output")
    try:
        kernel, metadata = import_ast_file(
            ast_path,
            kernel_name,
            source=source,
            source_bytes=source_bytes,
            facade_contract=facade_contract,
        )
    except ClangASTImportError as exc:
        raise AOTError(str(exc)) from exc
    except json.JSONDecodeError as exc:
        raise AOTError(f"Clang AST output is not JSON: {exc}") from exc
    return kernel, metadata


def emit(args: argparse.Namespace) -> int:
    if args.ast_fixture and args.skip_clang:
        raise AOTError("--ast-fixture and --skip-clang are mutually exclusive")
    source_bytes = args.source.read_bytes()
    source = source_bytes.decode("utf-8")
    header = args.header.read_bytes()
    # Host launch discovery is intentionally separate from device semantics.
    # The AST importer below owns the kernel, parameters, expressions, and
    # control flow whenever compiler validation is requested.
    host = KernelParser(source).parse_host_metadata()
    work = args.work
    work.mkdir(parents=True, exist_ok=True)
    facade_dir = Path(__file__).resolve().parent / "clang_cuda_facade"
    facade_contract = load_facade_contract(facade_dir)

    # Preserve the exact source as an evidence input. This is a copy of the
    # original translation unit, never a reconstructed kernel.
    device_source = work / "device-source.cu"
    device_source.write_bytes(source_bytes)
    response = work / "clang-device.rsp"
    response_args = materialize_facade_arguments(args.source, facade_dir)
    response.write_text("\n".join(response_args) + "\n", encoding="utf-8")
    ast = work / "clang-device-ast.json"
    clang_log = work / "clang-device.log"
    semantic_required = True
    compiler_metadata: dict[str, Any]
    toolchain: dict[str, Any]

    if args.ast_fixture:
        fixture_bytes = args.ast_fixture.read_bytes()
        ast.write_bytes(fixture_bytes)
        clang_log.write_text("portable compiler-output fixture; no live compiler invocation\n", encoding="utf-8")
        kernel, compiler_metadata = import_typed_compiler_output(
            ast,
            host["launch"]["kernel"],
            source=args.source,
            facade_contract=facade_contract,
        )
        compiler_metadata["validation"] = "compiler_output_fixture"
        compiler_metadata["status"] = "FIXTURE"
        compiler_metadata["validation_status"] = "NOT_RUN"
        toolchain = {"status": "FIXTURE", "fixture": str(args.ast_fixture)}
        frontend = dict(compiler_metadata)
    elif args.skip_clang:
        # This path exists only for packaging mechanics. It is deliberately
        # marked NOT_RUN and cannot pass the semantic frontend gate.
        parser_kernel, _ = KernelParser(source).find_kernel()
        kernel = parser_kernel
        semantic_required = False
        compiler_metadata = {
            "kind": "source_parser",
            "schema": "cuda4as-source-parser-not-run-v1",
            "authoritative": False,
            "status": "NOT_RUN",
        }
        toolchain = {"status": "NOT_RUN_PORTABLE"}
        frontend = dict(compiler_metadata)
        ast.write_text("{\"kind\":\"PortableValidationPlaceholder\",\"typed\":false}\n", encoding="utf-8")
        clang_log.write_text("portable validation: compiler invocation intentionally not run; semantic IR is NOT_RUN\n", encoding="utf-8")
    else:
        tc = discover_toolchain(args.clang)
        toolchain = tc
        frontend = run_clang_device(
            tc["clang"], args.source, ast, clang_log, facade_dir,
            facade_contract=facade_contract,
        )
        if frontend["exit_code"] != 0:
            raise AOTError("Clang CUDA device validation failed; see clang-device.log")
        kernel, compiler_metadata = import_typed_compiler_output(
            ast,
            host["launch"]["kernel"],
            source=args.source,
            facade_contract=facade_contract,
        )
        frontend.update(compiler_metadata)
        frontend["validation_status"] = "VALIDATED"
        frontend["status"] = "PASS"

    if kernel["name"] != host["launch"]["kernel"]:
        raise AOTError("compiler kernel identity does not match host launch discovery")
    if semantic_required:
        source_kernel, _ = KernelParser(source).find_kernel()
        verify_source_compiler_signature(source_kernel, kernel)
    ir = SSABuilder(kernel).build()
    ir["source"] = {"path": args.source.name, "sha256": sha256_bytes(source_bytes), "header_sha256": sha256_bytes(header)}
    ir["semantic_source"] = compiler_metadata
    ir["frontend"] = frontend
    # A schema fixture proves importer behavior but cannot prove a live
    # compiler invocation. Keep its portable result separate from compiler
    # validation so no AOT record turns synthetic data into native evidence.
    ir["validation_status"] = "FIXTURE" if args.ast_fixture else "VALIDATED" if semantic_required else "NOT_RUN"
    ir["compiler_validation_status"] = "NOT_RUN" if args.ast_fixture or not semantic_required else "VALIDATED"
    ir["facade"] = {
        "version": facade_contract["version"],
        "header": FACADE_HEADER_REL,
        "sha256": facade_contract["header_sha256"],
        "manifest": FACADE_MANIFEST_REL,
        "manifest_sha256": facade_contract["manifest_sha256"],
        "supported_clang": facade_contract["supported_clang"],
        "target": facade_contract["target"]["cuda_gpu_arch"],
        "arguments": response_args,
    }
    verify_ir(ir, require_semantic_source=semantic_required)
    msl, abi = lower_to_msl(kernel, ir, host["launch"]["block_x"], require_semantic_source=semantic_required)
    link = {
        "schema": LINK_SCHEMA,
        "version": 1,
        "status": "LINKED",
        "empty": False,
        "module_count": 1,
        "modules": [{"name": kernel["name"], "ir_sha256": sha256_bytes(canonical_json(ir).encode()), "symbols": [kernel["name"]]}],
        "resolved_symbols": [kernel["name"]],
        "unresolved_symbols": [],
        "duplicate_symbols": [],
        "external_device_calls": [],
    }
    linked_image = canonical_json({
        "schema": "cuda4as-linked-image-v1",
        "module": kernel["name"],
        "symbols": [kernel["name"]],
        "ir_sha256": sha256_bytes(canonical_json(ir).encode()),
        "status": "linked",
    }).encode()
    link.update({"image_path": "device-link-image.json", "image_bytes": len(linked_image), "image_sha256": sha256_bytes(linked_image)})
    verify_abi(abi)
    verify_device_link(link)
    abi["source_sha256"] = sha256_bytes(source.encode())
    abi["header_sha256"] = sha256_bytes(header)
    driver = {
        "schema": DRIVER_SCHEMA,
        "version": 1,
        "source": {"path": args.source.name, "sha256": sha256_bytes(source_bytes), "header_sha256": sha256_bytes(header)},
        "device_input": {"path": args.source.name, "copy": device_source.name, "sha256": sha256_file(device_source), "reconstructed": False},
        "frontend": frontend,
        "host": host,
        "toolchain": toolchain,
        "facade": ir["facade"],
        "unsupported": sorted(ir["unsupported_capabilities"]),
        "semantic_authority": "clang_ast" if semantic_required else "NOT_RUN",
        "compiler_validation_status": ir["compiler_validation_status"],
    }
    (work / "ir.json").write_text(canonical_json(ir), encoding="utf-8")
    (work / "abi.json").write_text(canonical_json(abi), encoding="utf-8")
    (work / "device-link.json").write_text(canonical_json(link), encoding="utf-8")
    (work / "device-link-image.json").write_bytes(linked_image)
    (work / "kernel.metal").write_text(msl, encoding="utf-8")
    (work / "driver-facts.json").write_text(canonical_json(driver), encoding="utf-8")
    return 0 if semantic_required else 77


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("emit", nargs="?", default="emit")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--header", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--clang")
    p.add_argument("--ast-fixture", type=Path, help="portable compiler-output AST fixture; no live frontend invocation")
    p.add_argument("--skip-clang", action="store_true")
    args = p.parse_args(argv)
    try:
        return emit(args)
    except (AOTError, OSError, subprocess.SubprocessError) as exc:
        print(f"M2A_ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
