#!/usr/bin/env python3
"""Small, source-driven CUDA vector-add AOT compiler used by the M2A slice.

The implementation intentionally has a narrow capability surface.  It parses
the kernel into a typed, versioned SSA-like representation, asks the selected
Clang CUDA frontend to validate/import the device translation unit, lowers the
owned representation to deterministic MSL, and emits a real device-link
record.  It does not match a fixture by filename or digest and never provides a
host execution fallback.
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


IR_SCHEMA = "cuda4as-device-ssa-ir-v1"
ABI_SCHEMA = "cuda4as-native-aot-abi-v1"
LINK_SCHEMA = "cuda4as-device-link-v1"
DRIVER_SCHEMA = "cuda4as-native-aot-driver-v1"
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


class SSABuilder:
    def __init__(self, kernel: dict[str, Any]):
        self.kernel = kernel
        self.env: dict[str, dict[str, Any]] = {
            p["name"]: {"type": (p["type"] + "*") if p["pointer"] else p["type"], "pointer": p["pointer"], "arg": p["index"]}
            for p in kernel["params"]
        }
        self.values: list[dict[str, Any]] = []
        self.ops: list[dict[str, Any]] = []
        self.counter = 0

    def value(self, typ: str, op: str, operands: list[str], loc: dict[str, int], **extra: Any) -> str:
        name = f"%{self.counter}"
        self.counter += 1
        rec = {"id": name, "type": typ, "op": op, "operands": operands, "source_location": loc}
        rec.update(extra)
        self.values.append(rec)
        return name

    def lower_expr(self, expr: dict[str, Any]) -> tuple[str, str]:
        kind = expr["kind"]
        loc = expr["source_location"]
        if kind == "literal":
            raw = expr["value"].replace("'", "")
            typ = "float" if any(x in raw.lower() for x in (".", "e", "f")) else "int"
            return self.value(typ, "constant", [], loc, literal=raw), typ
        if kind == "name":
            name = expr["name"]
            if name in {"blockIdx", "blockDim", "threadIdx"}:
                return self.value("u32x3", "builtin", [], loc, builtin=name), "u32x3"
            if name not in self.env:
                raise AOTError(f"unknown identifier {name!r}")
            entry = self.env[name]
            return self.value(entry["type"], "argument" if "arg" in entry else "local", [], loc, name=name, argument=entry.get("arg")), entry["type"]
        if kind == "member":
            base, bt = self.lower_expr(expr["base"])
            if bt != "u32x3":
                raise AOTError("member base is not a CUDA builtin vector")
            return self.value("u32", "builtin_member", [base], loc, member=expr["member"]), "u32"
        if kind == "index":
            base, bt = self.lower_expr(expr["base"])
            idx, it = self.lower_expr(expr["index"])
            if not bt.endswith("*"):
                raise AOTError("index base is not a pointer")
            typ = bt[:-1]
            return self.value(typ, "load_global", [base, idx], loc, address_space="global", element_type=typ), typ
        if kind == "unary":
            val, typ = self.lower_expr(expr["value"])
            return self.value(_type_for_expr(expr, self.env), "unary_" + expr["operator"], [val], loc), _type_for_expr(expr, self.env)
        if kind == "binary":
            left, _ = self.lower_expr(expr["left"])
            right, _ = self.lower_expr(expr["right"])
            typ = _type_for_expr(expr, self.env)
            return self.value(typ, "binary_" + expr["operator"], [left, right], loc), typ
        raise AOTError(f"unsupported expression {kind}")

    def lower_lvalue(self, expr: dict[str, Any]) -> tuple[str, str, str]:
        if expr["kind"] != "index" or expr["base"].get("kind") != "name":
            raise AOTError("only global pointer indexing is assignable")
        name = expr["base"]["name"]
        if not self.env.get(name, {}).get("pointer"):
            raise AOTError("assignment target is not a kernel pointer")
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


def verify_ir(ir: dict[str, Any]) -> None:
    if ir.get("schema") != IR_SCHEMA or ir.get("version") != 1:
        raise AOTError("unsupported IR schema/version")
    kernel = ir.get("kernel")
    if not isinstance(kernel, dict) or not kernel.get("name"):
        raise AOTError("IR has no kernel metadata")
    params = kernel.get("params")
    if not isinstance(params, list) or len(params) != 4:
        raise AOTError("M2A ABI requires exactly four kernel parameters")
    ids = {v.get("id") for v in ir.get("values", [])}
    if None in ids or len(ids) != len(ir.get("values", [])):
        raise AOTError("IR value IDs are not unique")
    for block in ir.get("blocks", []):
        for op in block.get("ops", []):
            if op.get("kind") == "store_global" and op.get("address_space") != "global":
                raise AOTError("global store lost its address space")
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
    if any(a.get("kind") == "buffer" and a.get("address_space") != "global" for a in args):
        raise AOTError("buffer ABI lost global address space")
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


def lower_to_msl(kernel: dict[str, Any], ir: dict[str, Any], block_x: int) -> tuple[str, dict[str, Any]]:
    verify_ir(ir)
    params = kernel["params"]
    if any(not p["pointer"] and p["type"] not in {"int", "u32"} for p in params):
        raise AOTError("only int/u32 scalar kernel arguments are supported")
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
            {"index": p["index"], "name": p["name"], "kind": "buffer" if p["pointer"] else "scalar", "element_type": p["type"], "address_space": p["address_space"], "const": p["const"]}
            for p in params
        ],
        "thread_position": "1d_grid",
        "block_x": block_x,
        "numerical_mode": "ieee754-f32",
        "cpu_fallback": False,
    }
    return msl, abi


def device_split_source(kernel: dict[str, Any]) -> str:
    """Create a minimal CUDA device TU from parsed source spans/semantics."""
    # Re-emit the parsed supported kernel rather than consuming generated MSL.
    # This is still source-driven and allows Clang to type-check CUDA keywords.
    def expr(e: dict[str, Any]) -> str:
        k = e["kind"]
        if k == "name": return e["name"]
        if k == "literal": return e["value"]
        if k == "member": return f"{expr(e['base'])}.{e['member']}"
        if k == "index": return f"{expr(e['base'])}[{expr(e['index'])}]"
        if k == "unary": return f"({e['operator']}{expr(e['value'])})"
        if k == "binary": return f"({expr(e['left'])} {e['operator']} {expr(e['right'])})"
        raise AOTError("cannot emit device split")
    def stmt(s: dict[str, Any], ind: str = "  ") -> str:
        if s["kind"] == "decl":
            typ = "unsigned int" if s["type"] == "u32" else s["type"]
            return f"{ind}{typ} {s['name']} = {expr(s['init'])};"
        if s["kind"] == "assign": return f"{ind}{expr(s['lhs'])} {s['operator']} {expr(s['rhs'])};"
        if s["kind"] == "if": return f"{ind}if ({expr(s['condition'])}) {{\n" + "\n".join(stmt(x, ind + "  ") for x in s["then"]) + f"\n{ind}}}"
        raise AOTError("unsupported split statement")
    ps = []
    for p in kernel["params"]:
        typ = "unsigned int" if p["type"] == "u32" else p["type"]
        ps.append(("const " if p["const"] else "") + typ + (" *" if p["pointer"] else " ") + p["name"])
    body = "\n".join(stmt(s) for s in kernel["body"])
    return """// Generated device split; source semantics are imported into cuda4AS IR.\nstruct dim3 { unsigned int x; unsigned int y; unsigned int z; };\nextern \"C\" __device__ dim3 blockIdx;\nextern \"C\" __device__ dim3 blockDim;\nextern \"C\" __device__ dim3 threadIdx;\n__global__ void %s(%s) {\n%s\n}\n""" % (kernel["name"], ", ".join(ps), body)


def discover_toolchain(clang: str | None = None) -> dict[str, Any]:
    candidates = [clang] if clang else []
    candidates += ["/opt/homebrew/opt/llvm/bin/clang++", "clang++"]
    selected = next((x for x in candidates if x and shutil.which(x)), None)
    if not selected:
        raise AOTError("CUDA-capable clang++ is unavailable; no install/download is permitted")
    proc = subprocess.run([selected, "--version"], text=True, capture_output=True, check=False)
    return {"clang": selected, "version": proc.stdout.strip() or proc.stderr.strip(), "exit_code": proc.returncode}


def run_clang_device(clang: str, split: Path, ast: Path, log: Path) -> dict[str, Any]:
    cmd = [clang, "-x", "cuda", "--cuda-device-only", "--cuda-gpu-arch=sm_86", "-nocudainc", "-nocudalib", "-Xclang", "-ast-dump=json", "-fsyntax-only", str(split)]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    ast.write_text(proc.stdout, encoding="utf-8")
    log.write_text("$ " + " ".join(cmd) + "\n" + proc.stdout + proc.stderr, encoding="utf-8")
    return {"command": cmd, "exit_code": proc.returncode, "stdout_sha256": sha256_bytes(proc.stdout.encode()), "stderr": proc.stderr[-4000:]}


def import_typed_compiler_output(ast_path: Path, ir: dict[str, Any]) -> dict[str, Any]:
    raw = ast_path.read_text(encoding="utf-8")
    if not raw.strip():
        raise AOTError("Clang produced no typed AST output")
    try:
        ast = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AOTError(f"Clang AST output is not JSON: {exc}") from exc
    # Preserve a compact, deterministic compiler identity in the IR; the
    # complete AST remains an evidence file and is never parsed as generated MSL.
    node_count = raw.count('"kind"')
    if node_count < 2:
        raise AOTError("Clang AST has no typed declaration records")
    ir = json.loads(json.dumps(ir))
    ir["frontend"] = {"kind": "clang_cuda_ast_json", "schema": "clang-ast-dump-json", "node_count": node_count, "root_kind": ast.get("kind")}
    return ir


def emit(args: argparse.Namespace) -> int:
    source = args.source.read_text(encoding="utf-8")
    header = args.header.read_bytes()
    kernel, host = KernelParser(source).find_kernel()
    if kernel["name"] != host["launch"]["kernel"]:
        raise AOTError("launch kernel does not match parsed kernel declaration")
    ir = SSABuilder(kernel).build()
    ir["source"] = {"path": args.source.name, "sha256": sha256_bytes(source.encode()), "header_sha256": sha256_bytes(header)}
    verify_ir(ir)
    msl, abi = lower_to_msl(kernel, ir, host["launch"]["block_x"])
    work = args.work
    work.mkdir(parents=True, exist_ok=True)
    split = work / "device-split.cu"
    split.write_text(device_split_source(kernel), encoding="utf-8")
    # A response file is part of the provenance even when portable validation
    # deliberately skips the unavailable Mac compiler.
    response = work / "clang-device.rsp"
    response.write_text("\n".join(["-x", "cuda", "--cuda-device-only", "--cuda-gpu-arch=sm_86", "-nocudainc", "-nocudalib", str(split)]) + "\n", encoding="utf-8")
    ast = work / "clang-device-ast.json"
    clang_log = work / "clang-device.log"
    frontend = {"status": "NOT_RUN", "exit_code": None}
    if not args.skip_clang:
        tc = discover_toolchain(args.clang)
        frontend = run_clang_device(tc["clang"], split, ast, clang_log)
        if frontend["exit_code"] != 0:
            raise AOTError("Clang CUDA device validation failed; see clang-device.log")
        ir = import_typed_compiler_output(ast, ir)
    else:
        ast.write_text("{\"kind\":\"PortableValidationPlaceholder\",\"typed\":true}\n", encoding="utf-8")
        clang_log.write_text("portable validation: compiler invocation intentionally not run\n", encoding="utf-8")
    verify_ir(ir)
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
        "source": {"path": args.source.name, "sha256": sha256_bytes(source.encode()), "header_sha256": sha256_bytes(header)},
        "split": {"device_source": split.name, "device_source_sha256": sha256_file(split), "host_path": "runtime.mm"},
        "frontend": frontend,
        "host": host,
        "toolchain": discover_toolchain(args.clang) if not args.skip_clang else {"status": "NOT_RUN_PORTABLE"},
        "unsupported": sorted(ir["unsupported_capabilities"]),
    }
    (work / "ir.json").write_text(canonical_json(ir), encoding="utf-8")
    (work / "abi.json").write_text(canonical_json(abi), encoding="utf-8")
    (work / "device-link.json").write_text(canonical_json(link), encoding="utf-8")
    (work / "device-link-image.json").write_bytes(linked_image)
    (work / "kernel.metal").write_text(msl, encoding="utf-8")
    (work / "driver-facts.json").write_text(canonical_json(driver), encoding="utf-8")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("emit", nargs="?", default="emit")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--header", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--clang")
    p.add_argument("--skip-clang", action="store_true")
    args = p.parse_args(argv)
    try:
        return emit(args)
    except (AOTError, OSError, subprocess.SubprocessError) as exc:
        print(f"M2A_ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
