"""Semantic importer for Clang CUDA AST JSON used by Native AOT Core v1.

The importer deliberately consumes typed compiler records instead of source
tokens.  It accepts only the small AST profile required by the vector-add
slice, preserves source locations and compiler type spellings, and fails closed
when a node or CUDA builtin is not represented explicitly.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable


AST_SCHEMA = "cuda4as-clang-cuda-ast-semantic-v1"
BUILTINS = {"blockIdx", "blockDim", "threadIdx"}
CUDA_BUILTIN_ATTRS = {"CUDADeviceBuiltinAttr", "CUDADeviceAttr"}
WRAPPER_NODES = {
    "ImplicitCastExpr",
    "CStyleCastExpr",
    "CXXStaticCastExpr",
    "ParenExpr",
    "ConstantExpr",
    "ExprWithCleanups",
    "MaterializeTemporaryExpr",
}


class ClangASTImportError(ValueError):
    """A compiler-output semantic or capability error."""


def _children(node: dict[str, Any]) -> list[dict[str, Any]]:
    raw = node.get("inner", [])
    if not isinstance(raw, list) or any(not isinstance(item, dict) for item in raw):
        raise ClangASTImportError(
            f"compiler AST node {node.get('kind', '<unknown>')} has malformed inner records"
        )
    return raw


def walk(node: Any) -> Iterable[dict[str, Any]]:
    if isinstance(node, dict):
        yield node
        for value in node.values():
            if isinstance(value, (dict, list)):
                yield from walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from walk(value)


def _location(node: dict[str, Any], fallback: dict[str, int] | None = None) -> dict[str, int]:
    """Return a stable source location from a Clang AST record."""

    loc = node.get("loc")
    if isinstance(loc, dict) and isinstance(loc.get("line"), int) and isinstance(loc.get("col"), int):
        return {"line": loc["line"], "column": loc["col"]}
    span = node.get("range")
    if isinstance(span, dict):
        begin = span.get("begin")
        if isinstance(begin, dict) and isinstance(begin.get("line"), int) and isinstance(begin.get("col"), int):
            return {"line": begin["line"], "column": begin["col"]}
    # Clang can omit locations for synthetic wrappers and some compact AST
    # serializers omit repeated leaf locations. In that case inherit the
    # nearest compiler-record location while retaining the exact location when
    # the record provides one.
    if fallback is not None and fallback.get("line", 0) > 0 and fallback.get("column", 0) > 0:
        return {"line": fallback["line"], "column": fallback["column"]}
    return {"line": 0, "column": 0}


def _require_location(kind: str, location: dict[str, int]) -> dict[str, int]:
    if location.get("line", 0) <= 0 or location.get("column", 0) <= 0:
        raise ClangASTImportError(f"compiler {kind} has no source location")
    return location


def _attrs(node: dict[str, Any]) -> set[str]:
    attrs = node.get("attrs", [])
    if not isinstance(attrs, list):
        return set()
    return {item.get("kind") for item in attrs if isinstance(item, dict) and isinstance(item.get("kind"), str)}


def _qual_type(node: dict[str, Any]) -> str:
    typ = node.get("type")
    if isinstance(typ, dict) and isinstance(typ.get("qualType"), str):
        return " ".join(typ["qualType"].split())
    raise ClangASTImportError(f"compiler AST node {node.get('kind', '<unknown>')} has no typed qualType")


def _type_address_space(node: dict[str, Any], qual_type: str) -> str | None:
    typ = node.get("type")
    if isinstance(typ, dict):
        for key in ("addressSpace", "address_space"):
            value = typ.get(key)
            if isinstance(value, str) and value:
                return value
    for key in ("addressSpace", "address_space"):
        value = node.get(key)
        if isinstance(value, str) and value:
            return value
    match = re.search(r"(?:__attribute__\(\(address_space\(([^)]+)\)\)\)|__(global|constant|shared)__)", qual_type)
    if match:
        return match.group(1) or match.group(2)
    return None


def normalize_type(node: dict[str, Any], *, parameter: bool = False) -> dict[str, Any]:
    """Normalize a compiler type while retaining its complete spelling."""

    qual_type = _qual_type(node)
    compact = qual_type.replace(" &", "&").replace(" *", "*")
    pointer = "*" in compact
    const = bool(re.search(r"(?:^|\s)const(?:\s|$)", qual_type))
    address_space = _type_address_space(node, qual_type)
    if pointer:
        base = compact.split("*", 1)[0].strip()
        base = re.sub(r"\b(const|volatile|restrict)\b", "", base).strip()
        scalar = normalize_scalar(base)
        # CUDA kernel pointer parameters use the global address space unless
        # Clang records a more specific qualifier in the typed record.
        if address_space in {None, "generic", "0"} and parameter:
            address_space = "global"
        if address_space not in {"global", "constant", "shared", "local", None}:
            raise ClangASTImportError(f"unsupported pointer address space {address_space!r}")
        if parameter and address_space != "global":
            raise ClangASTImportError("M2A kernel pointer parameter is not in global address space")
        return {
            "type": scalar,
            "pointer": True,
            "const": const,
            "address_space": address_space or "global",
            "qual_type": qual_type,
        }
    scalar = normalize_scalar(compact)
    return {
        "type": scalar,
        "pointer": False,
        "const": const,
        "address_space": "constant",
        "qual_type": qual_type,
    }


def normalize_scalar(qual_type: str) -> str:
    q = " ".join(qual_type.replace("struct ", "").split())
    q = re.sub(r"\b(const|volatile|restrict)\b", "", q).strip()
    if q in {"unsigned", "unsigned int", "uint", "uint32_t"}:
        return "u32"
    if q in {"int", "signed", "signed int"}:
        return "int"
    if q in {"float", "float32_t"}:
        return "float"
    if q in {"bool", "_Bool"}:
        return "bool"
    if q in {"dim3", "const dim3"}:
        return "u32x3"
    if q == "void":
        return "void"
    raise ClangASTImportError(f"unsupported compiler type {qual_type!r}")


def _single_child(node: dict[str, Any], *, context: str) -> dict[str, Any]:
    children = _children(node)
    if len(children) != 1:
        raise ClangASTImportError(f"{context} requires one compiler child, got {len(children)}")
    return children[0]


def _referenced_name(node: dict[str, Any]) -> str:
    ref = node.get("referencedDecl")
    if isinstance(ref, dict) and isinstance(ref.get("name"), str):
        return ref["name"]
    if isinstance(node.get("name"), str):
        return node["name"]
    raise ClangASTImportError("compiler DeclRefExpr has no referenced declaration name")


class SemanticImporter:
    def __init__(self, root: dict[str, Any], function: dict[str, Any]):
        self.root = root
        self.function = function
        self.env: dict[str, dict[str, Any]] = {}
        self.builtin_decls: dict[str, dict[str, Any]] = {}
        for node in walk(root):
            if node.get("kind") == "VarDecl" and node.get("name") in BUILTINS:
                self.builtin_decls[node["name"]] = node

    def _expr_type(self, node: dict[str, Any], default: str | None = None) -> str:
        try:
            return normalize_scalar(_qual_type(node))
        except ClangASTImportError:
            if default is not None:
                return default
            raise

    def _check_builtin_decl(self, name: str, node: dict[str, Any]) -> None:
        ref = node.get("referencedDecl") if node.get("kind") == "DeclRefExpr" else node
        if not isinstance(ref, dict):
            raise ClangASTImportError(f"builtin {name} has no compiler declaration")
        declaration = self.builtin_decls.get(name, ref)
        attrs = _attrs(ref) | _attrs(declaration)
        if not (attrs & CUDA_BUILTIN_ATTRS):
            raise ClangASTImportError(f"builtin {name} lacks CUDA device_builtin compiler attribute")
        ref_type = None
        try:
            ref_type = _qual_type(ref)
        except ClangASTImportError:
            pass
        decl_type = _qual_type(declaration)
        if ref_type not in {None, "dim3", "const dim3"}:
            raise ClangASTImportError(f"builtin {name} reference is not typed as dim3")
        if decl_type not in {"dim3", "const dim3"}:
            raise ClangASTImportError(f"builtin {name} is not typed as dim3")

    def expr(self, node: dict[str, Any], inherited_location: dict[str, int] | None = None) -> dict[str, Any]:
        kind = node.get("kind")
        loc = _require_location(str(kind), _location(node, inherited_location))
        if kind in WRAPPER_NODES:
            child = self.expr(_single_child(node, context=kind), loc)
            out_type = self._expr_type(node, child["type"])
            cast_kind = node.get("castKind")
            if kind == "ParenExpr" and out_type == child["type"]:
                child.setdefault("compiler_wrappers", []).append(kind)
                return child
            if cast_kind in {None, "NoOp", "LValueToRValue"} and out_type == child["type"]:
                child.setdefault("compiler_wrappers", []).append(cast_kind or kind)
                return child
            return {
                "kind": "cast",
                "cast_kind": cast_kind or kind,
                "value": child,
                "type": out_type,
                "source_location": loc,
            }
        if kind == "DeclRefExpr":
            name = _referenced_name(node)
            if name in BUILTINS:
                self._check_builtin_decl(name, node)
                return {"kind": "name", "name": name, "builtin": name, "type": "u32x3", "source_location": loc}
            binding = self.env.get(name)
            if binding is None:
                raise ClangASTImportError(f"compiler DeclRefExpr references unknown declaration {name!r}")
            observed = self._expr_type(node, binding["type"])
            if observed != binding["type"]:
                raise ClangASTImportError(
                    f"compiler declaration type mismatch for {name!r}: {observed} != {binding['type']}"
                )
            result = {"kind": "name", "name": name, "type": binding["type"], "source_location": loc}
            if binding.get("pointer"):
                result.update({"pointer": True, "const": binding.get("const", False), "address_space": binding.get("address_space", "global")})
            if "argument" in binding:
                result["argument"] = binding["argument"]
            return result
        if kind == "MemberExpr":
            children = _children(node)
            if len(children) != 1:
                raise ClangASTImportError("compiler MemberExpr must have one base expression")
            base = self.expr(children[0], loc)
            member = node.get("name")
            if member not in {"x", "y", "z"}:
                ref = node.get("referencedMemberDecl")
                member = ref.get("name") if isinstance(ref, dict) else None
            if member not in {"x", "y", "z"} or base.get("builtin") not in BUILTINS:
                raise ClangASTImportError("compiler member access is not a CUDA x/y/z builtin")
            return {"kind": "member", "base": base, "member": member, "type": "u32", "source_location": loc}
        if kind == "ArraySubscriptExpr":
            children = _children(node)
            if len(children) != 2:
                raise ClangASTImportError("compiler ArraySubscriptExpr must have base and index")
            base, index = self.expr(children[0], loc), self.expr(children[1], loc)
            if not base.get("pointer") or base.get("address_space") != "global":
                raise ClangASTImportError("compiler array subscript base is not a global pointer")
            scalar = self._expr_type(node)
            if scalar in {"void", "u32x3"}:
                raise ClangASTImportError("compiler array subscript has unsupported element type")
            return {
                "kind": "index",
                "base": base,
                "index": index,
                "type": scalar,
                "address_space": "global",
                "source_location": loc,
            }
        if kind == "BinaryOperator":
            children = _children(node)
            if len(children) != 2:
                raise ClangASTImportError("compiler BinaryOperator must have two operands")
            opcode = node.get("opcode")
            if not isinstance(opcode, str) or opcode not in {"=", "+", "-", "*", "/", "%", "<", "<=", ">", ">=", "==", "!=", "&&", "||", "&", "|", "^", "<<", ">>"}:
                raise ClangASTImportError(f"unsupported compiler binary opcode {opcode!r}")
            left, right = self.expr(children[0], loc), self.expr(children[1], loc)
            return {
                "kind": "binary",
                "operator": opcode,
                "left": left,
                "right": right,
                "type": self._expr_type(node),
                "source_location": loc,
            }
        if kind == "UnaryOperator":
            children = _children(node)
            opcode = node.get("opcode")
            if len(children) != 1 or opcode not in {"+", "-", "!", "~"}:
                raise ClangASTImportError(f"unsupported compiler unary opcode {opcode!r}")
            return {
                "kind": "unary",
                "operator": opcode,
                "value": self.expr(children[0], loc),
                "type": self._expr_type(node),
                "source_location": loc,
            }
        if kind == "IntegerLiteral":
            value = node.get("value")
            if not isinstance(value, str | int):
                raise ClangASTImportError("compiler integer literal has no value")
            return {"kind": "literal", "value": str(value), "type": self._expr_type(node, "int"), "source_location": loc}
        if kind == "FloatingLiteral":
            value = node.get("value")
            if not isinstance(value, str | int | float):
                raise ClangASTImportError("compiler floating literal has no value")
            return {"kind": "literal", "value": str(value), "type": "float", "source_location": loc}
        if kind == "CXXBoolLiteralExpr":
            value = node.get("value")
            if not isinstance(value, bool):
                raise ClangASTImportError("compiler bool literal has no value")
            return {"kind": "literal", "value": "true" if value else "false", "type": "bool", "source_location": loc}
        raise ClangASTImportError(f"unsupported compiler expression node {kind!r}")

    def stmt(self, node: dict[str, Any], inherited_location: dict[str, int] | None = None) -> dict[str, Any]:
        kind = node.get("kind")
        loc = _require_location(str(kind), _location(node, inherited_location))
        if kind == "CompoundStmt":
            return {"kind": "compound", "statements": [self.stmt(child, loc) for child in _children(node)], "source_location": loc}
        if kind == "DeclStmt":
            children = _children(node)
            if len(children) != 1 or children[0].get("kind") != "VarDecl":
                raise ClangASTImportError("compiler DeclStmt must contain one VarDecl")
            declaration = children[0]
            name = declaration.get("name")
            if not isinstance(name, str) or name in self.env:
                raise ClangASTImportError(f"invalid or duplicate compiler local declaration {name!r}")
            typ = normalize_type(declaration)
            if typ["pointer"]:
                raise ClangASTImportError("pointer locals are outside the M2A profile")
            init_children = _children(declaration)
            if len(init_children) != 1:
                raise ClangASTImportError("M2A locals require one compiler initializer")
            init = self.expr(init_children[0], loc)
            if init["type"] != typ["type"] and not (typ["type"] == "int" and init["type"] == "u32"):
                raise ClangASTImportError(f"compiler initializer type {init['type']} does not match {typ['type']}")
            self.env[name] = {"type": typ["type"], "pointer": False, "const": typ["const"]}
            return {"kind": "decl", "name": name, "type": typ["type"], "init": init, "source_location": loc}
        if kind == "IfStmt":
            children = _children(node)
            if len(children) not in {2, 3}:
                raise ClangASTImportError("compiler IfStmt must contain condition and then body")
            if len(children) == 3:
                raise ClangASTImportError("else control flow is outside Native AOT Core v1")
            condition = self.expr(children[0], loc)
            if condition["type"] != "bool":
                raise ClangASTImportError("compiler IfStmt condition is not bool")
            then = self.stmt(children[1], loc)
            if then["kind"] != "compound":
                then = {"kind": "compound", "statements": [then], "source_location": then["source_location"]}
            return {"kind": "if", "condition": condition, "then": then["statements"], "source_location": loc}
        if kind == "BinaryOperator" and node.get("opcode") == "=":
            children = _children(node)
            if len(children) != 2:
                raise ClangASTImportError("compiler assignment must have two operands")
            lhs, rhs = self.expr(children[0], loc), self.expr(children[1], loc)
            if lhs["kind"] != "index" or lhs.get("address_space") != "global":
                raise ClangASTImportError("compiler assignment target is not a global array element")
            if rhs["type"] != lhs["type"] and not (lhs["type"] == "float" and rhs["type"] == "int"):
                raise ClangASTImportError(f"compiler store type {rhs['type']} does not match {lhs['type']}")
            return {"kind": "assign", "lhs": lhs, "operator": "=", "rhs": rhs, "source_location": loc}
        raise ClangASTImportError(f"unsupported compiler statement node {kind!r}")

    def kernel(self) -> tuple[dict[str, Any], dict[str, Any]]:
        attrs = _attrs(self.function)
        if "CUDAKernelAttr" not in attrs and not self.function.get("isCUDAKernel"):
            raise ClangASTImportError("compiler FunctionDecl is not marked CUDA kernel")
        function_type = _qual_type(self.function)
        if not function_type.startswith("void ("):
            raise ClangASTImportError("M2A CUDA kernel must return void according to Clang")
        params: list[dict[str, Any]] = []
        body: dict[str, Any] | None = None
        function_location = _require_location("FunctionDecl", _location(self.function))
        for child in _children(self.function):
            if child.get("kind") == "ParmVarDecl":
                typ = normalize_type(child, parameter=True)
                name = child.get("name")
                if not isinstance(name, str):
                    raise ClangASTImportError("compiler kernel parameter has no name")
                if typ["type"] not in {"float", "int", "u32"}:
                    raise ClangASTImportError(f"unsupported kernel parameter type {typ['qual_type']!r}")
                param = {
                    "index": len(params),
                    "name": name,
                    "type": typ["type"],
                    "pointer": typ["pointer"],
                    "const": typ["const"],
                    "address_space": typ["address_space"],
                    "qual_type": typ["qual_type"],
                    "source_location": _require_location("ParmVarDecl", _location(child)),
                }
                params.append(param)
                self.env[name] = {
                    "type": (typ["type"] + "*") if typ["pointer"] else typ["type"],
                    "pointer": typ["pointer"],
                    "const": typ["const"],
                    "address_space": typ["address_space"],
                    "argument": param["index"],
                }
            elif child.get("kind") == "CompoundStmt":
                if body is not None:
                    raise ClangASTImportError("compiler kernel has multiple bodies")
                body = child
        if body is None:
            raise ClangASTImportError("compiler kernel has no CompoundStmt body")
        if len(params) != 4:
            raise ClangASTImportError("M2A ABI requires exactly four compiler kernel parameters")
        semantic_body = self.stmt(body)
        kernel = {
            "name": self.function.get("name"),
            "return_type": "void",
            "params": params,
            "body": semantic_body["statements"],
            "source_location": function_location,
            "body_location": _require_location("CompoundStmt", _location(body)),
            "semantic_source": "clang_ast",
            "compiler_function_id": self.function.get("id"),
            "compiler_mangled_name": self.function.get("mangledName"),
            "compiler_attributes": sorted(attrs),
        }
        metadata = {
            "kind": "clang_cuda_ast_json",
            "schema": AST_SCHEMA,
            "authoritative": True,
            "function_id": self.function.get("id"),
            "function_name": self.function.get("name"),
            "mangled_name": self.function.get("mangledName"),
            "kernel_attributes": sorted(attrs),
            "node_count": sum(1 for _ in walk(self.function)),
        }
        return kernel, metadata


def import_ast(root: dict[str, Any], kernel_name: str, *, ast_sha256: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    if root.get("kind") != "TranslationUnitDecl":
        raise ClangASTImportError("compiler AST root is not TranslationUnitDecl")
    matches = [node for node in walk(root) if node.get("kind") == "FunctionDecl" and node.get("name") == kernel_name]
    if len(matches) != 1:
        raise ClangASTImportError(f"expected one compiler kernel FunctionDecl {kernel_name!r}, found {len(matches)}")
    kernel, metadata = SemanticImporter(root, matches[0]).kernel()
    if ast_sha256 is not None:
        metadata["ast_sha256"] = ast_sha256
    return kernel, metadata


def import_ast_file(path: Path, kernel_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        raw = path.read_bytes()
        root = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ClangASTImportError(f"cannot load Clang AST JSON: {exc}") from exc
    return import_ast(root, kernel_name, ast_sha256=hashlib.sha256(raw).hexdigest())
