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
CUDA_BUILTIN_ATTR = "CUDADeviceBuiltinAttr"
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


def _point(candidate: Any, source_bytes: bytes | None = None) -> dict[str, Any] | None:
    if not isinstance(candidate, dict):
        return None
    offset = candidate.get("offset")
    result: dict[str, Any] = {}
    if isinstance(offset, int) and offset >= 0:
        result["offset"] = offset
    line, column = candidate.get("line"), candidate.get("col")
    if isinstance(line, int) and isinstance(column, int):
        result.update({"line": line, "column": column, "provenance": "exact"})
        return result
    if "offset" in result and source_bytes is not None and offset <= len(source_bytes):
        prefix = source_bytes[:offset]
        result["line"] = prefix.count(b"\n") + 1
        last_newline = prefix.rfind(b"\n")
        result["column"] = offset - (last_newline + 1) + 1
        result["provenance"] = "exact_offset"
        return result
    return None


def _location(
    node: dict[str, Any],
    fallback: dict[str, Any] | None = None,
    *,
    source_bytes: bytes | None = None,
) -> dict[str, Any]:
    """Return a compiler-exact or explicitly inherited source location."""

    loc = node.get("loc")
    candidates = [loc]
    if isinstance(loc, dict):
        candidates.extend([loc.get("expansionLoc"), loc.get("spellingLoc")])
    span = node.get("range")
    if isinstance(span, dict):
        begin = span.get("begin")
        candidates.append(begin)
        if isinstance(begin, dict):
            candidates.extend([begin.get("expansionLoc"), begin.get("spellingLoc")])
    for candidate in candidates:
        point = _point(candidate, source_bytes)
        if point is not None:
            return point
    # Clang may omit locations for synthetic wrappers. Preserve the inherited
    # label so downstream evidence never presents a parent coordinate as exact.
    if fallback is not None and fallback.get("line", 0) > 0 and fallback.get("column", 0) > 0:
        inherited = dict(fallback)
        inherited["provenance"] = "inherited"
        return inherited
    return {"line": 0, "column": 0, "provenance": "missing"}


def _require_location(kind: str, location: dict[str, Any]) -> dict[str, Any]:
    if location.get("line", 0) <= 0 or location.get("column", 0) <= 0:
        raise ClangASTImportError(f"compiler {kind} has no source location")
    if location.get("provenance") not in {"exact", "exact_offset", "inherited"}:
        raise ClangASTImportError(f"compiler {kind} has unclassified source location provenance")
    return location


def _direct_attributes(node: dict[str, Any]) -> list[dict[str, Any]]:
    """Return only direct Clang *Attr children of one declaration node."""

    raw = node.get("inner", [])
    if not isinstance(raw, list) or any(not isinstance(item, dict) for item in raw):
        raise ClangASTImportError(
            f"compiler declaration {node.get('kind', '<unknown>')} has malformed inner records"
        )
    return [item for item in raw if isinstance(item.get("kind"), str) and item["kind"].endswith("Attr")]


def _attrs(node: dict[str, Any]) -> set[str]:
    """Read declaration attributes from Clang's direct inner child records."""

    return {item["kind"] for item in _direct_attributes(node)}


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
            if isinstance(value, (str, int)) and not isinstance(value, bool) and value != "":
                return str(value)
    for key in ("addressSpace", "address_space"):
        value = node.get(key)
        if isinstance(value, (str, int)) and not isinstance(value, bool) and value != "":
            return str(value)
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
        # An absent address-space field is an unknown compiler fact. Preserve
        # it rather than silently applying cuda4AS's global-pointer ABI rule.
        # Backend lowering decides separately whether that fact is usable.
        semantic_address_space = address_space if address_space is not None else "unknown"
        return {
            "type": scalar,
            "pointer": True,
            "const": const,
            "address_space": semantic_address_space,
            "qual_type": qual_type,
        }
    scalar = normalize_scalar(compact)
    return {
        "type": scalar,
        "pointer": False,
        "const": const,
        "address_space": address_space if address_space is not None else "unknown",
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
    def __init__(self, root: dict[str, Any], function: dict[str, Any], source_bytes: bytes | None = None):
        self.root = root
        self.function = function
        self.source_bytes = source_bytes
        self.env: dict[str, dict[str, Any]] = {}
        self.env_by_id: dict[str, dict[str, Any]] = {}
        self.declarations_by_id: dict[str, dict[str, Any]] = {}
        for node in walk(root):
            node_id = node.get("id")
            if isinstance(node_id, str) and node.get("kind") in {"VarDecl", "ParmVarDecl"}:
                # The first declaration record is the canonical one; later
                # referencedDecl snapshots must not shadow its attributes.
                self.declarations_by_id.setdefault(node_id, node)

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
        ref_id = ref.get("id")
        if not isinstance(ref_id, str):
            raise ClangASTImportError(f"builtin {name} has no compiler declaration identity")
        declaration = self.declarations_by_id.get(ref_id)
        if not isinstance(declaration, dict) or declaration.get("kind") != "VarDecl" or declaration.get("name") != name:
            raise ClangASTImportError(f"builtin {name} does not resolve to its compiler VarDecl")
        attrs = _attrs(declaration)
        if CUDA_BUILTIN_ATTR not in attrs or "CUDADeviceAttr" not in attrs:
            raise ClangASTImportError(f"builtin {name} lacks direct CUDA device/builtin compiler attributes")
        if "CUDAHostAttr" in attrs:
            raise ClangASTImportError(f"builtin {name} has conflicting CUDA host attribute")
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

    def expr(self, node: dict[str, Any], inherited_location: dict[str, Any] | None = None) -> dict[str, Any]:
        kind = node.get("kind")
        loc = _require_location(str(kind), _location(node, inherited_location, source_bytes=self.source_bytes))
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
            ref = node.get("referencedDecl")
            ref_id = ref.get("id") if isinstance(ref, dict) else None
            if not isinstance(ref_id, str):
                raise ClangASTImportError(f"compiler DeclRefExpr for {name!r} has no declaration identity")
            binding = self.env_by_id.get(ref_id)
            if binding is None:
                raise ClangASTImportError(f"compiler DeclRefExpr references unknown declaration {name!r} ({ref_id})")
            if binding.get("name") != name:
                raise ClangASTImportError(f"compiler declaration identity/name mismatch for {name!r}")
            observed = self._expr_type(node, binding["type"])
            if observed != binding["type"]:
                raise ClangASTImportError(
                    f"compiler declaration type mismatch for {name!r}: {observed} != {binding['type']}"
                )
            result = {"kind": "name", "name": name, "type": binding["type"], "source_location": loc}
            if binding.get("pointer"):
                result.update({"pointer": True, "const": binding.get("const", False), "address_space": binding.get("address_space", "unknown")})
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
            if not base.get("pointer"):
                raise ClangASTImportError("compiler array subscript base is not a pointer")
            address_space = base.get("address_space", "unknown")
            if not isinstance(address_space, str):
                raise ClangASTImportError("compiler array subscript base has no address-space fact")
            scalar = self._expr_type(node)
            if scalar in {"void", "u32x3"}:
                raise ClangASTImportError("compiler array subscript has unsupported element type")
            return {
                "kind": "index",
                "base": base,
                "index": index,
                "type": scalar,
                "address_space": address_space,
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

    def stmt(self, node: dict[str, Any], inherited_location: dict[str, Any] | None = None) -> dict[str, Any]:
        kind = node.get("kind")
        loc = _require_location(str(kind), _location(node, inherited_location, source_bytes=self.source_bytes))
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
            local_attrs = _attrs(declaration)
            unsupported_local_cuda = {item for item in local_attrs if item.startswith("CUDA")}
            if unsupported_local_cuda:
                raise ClangASTImportError(
                    f"compiler local declaration has unsupported CUDA attributes: {sorted(unsupported_local_cuda)}"
                )
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
            declaration_id = declaration.get("id")
            if not isinstance(declaration_id, str):
                raise ClangASTImportError(f"compiler local declaration {name!r} has no identity")
            self.env_by_id[declaration_id] = {
                "type": typ["type"],
                "pointer": False,
                "const": typ["const"],
                "name": name,
            }
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
        if "CUDAGlobalAttr" not in attrs:
            raise ClangASTImportError("compiler FunctionDecl has no direct CUDAGlobalAttr")
        if attrs & {"CUDADeviceAttr", "CUDAHostAttr"}:
            raise ClangASTImportError("compiler FunctionDecl has conflicting CUDA execution attributes")
        unexpected_cuda = {item for item in attrs if item.startswith("CUDA") and item != "CUDAGlobalAttr"}
        if unexpected_cuda:
            raise ClangASTImportError(f"compiler FunctionDecl has unsupported CUDA attributes: {sorted(unexpected_cuda)}")
        function_type = _qual_type(self.function)
        if not function_type.startswith("void ("):
            raise ClangASTImportError("M2A CUDA kernel must return void according to Clang")
        params: list[dict[str, Any]] = []
        body: dict[str, Any] | None = None
        function_location = _require_location("FunctionDecl", _location(self.function, source_bytes=self.source_bytes))
        for child in _children(self.function):
            child_kind = child.get("kind")
            if isinstance(child_kind, str) and child_kind.endswith("Attr"):
                continue
            if child_kind == "ParmVarDecl":
                parameter_attrs = _attrs(child)
                unsupported_parameter_cuda = {item for item in parameter_attrs if item.startswith("CUDA")}
                if unsupported_parameter_cuda:
                    raise ClangASTImportError(
                        f"compiler kernel parameter has unsupported CUDA attributes: {sorted(unsupported_parameter_cuda)}"
                    )
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
                    "source_location": _require_location("ParmVarDecl", _location(child, source_bytes=self.source_bytes)),
                }
                params.append(param)
                self.env[name] = {
                    "type": (typ["type"] + "*") if typ["pointer"] else typ["type"],
                    "pointer": typ["pointer"],
                    "const": typ["const"],
                    "address_space": typ["address_space"],
                    "argument": param["index"],
                    "name": name,
                }
                if not isinstance(child.get("id"), str):
                    raise ClangASTImportError(f"compiler kernel parameter {name!r} has no identity")
                self.env_by_id[child["id"]] = self.env[name]
            elif child_kind == "CompoundStmt":
                if body is not None:
                    raise ClangASTImportError("compiler kernel has multiple bodies")
                body = child
            else:
                raise ClangASTImportError(f"unsupported compiler FunctionDecl child {child_kind!r}")
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
            "body_location": _require_location("CompoundStmt", _location(body, source_bytes=self.source_bytes)),
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


def import_ast(
    root: dict[str, Any],
    kernel_name: str,
    *,
    ast_sha256: str | None = None,
    source_bytes: bytes | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if root.get("kind") != "TranslationUnitDecl":
        raise ClangASTImportError("compiler AST root is not TranslationUnitDecl")
    matches = [node for node in walk(root) if node.get("kind") == "FunctionDecl" and node.get("name") == kernel_name]
    if len(matches) != 1:
        raise ClangASTImportError(f"expected one compiler kernel FunctionDecl {kernel_name!r}, found {len(matches)}")
    kernel, metadata = SemanticImporter(root, matches[0], source_bytes=source_bytes).kernel()
    if ast_sha256 is not None:
        metadata["ast_sha256"] = ast_sha256
    return kernel, metadata


def import_ast_file(path: Path, kernel_name: str, *, source: Path | None = None, source_bytes: bytes | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        raw = path.read_bytes()
        root = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ClangASTImportError(f"cannot load Clang AST JSON: {exc}") from exc
    if source is not None and source_bytes is not None:
        raise ClangASTImportError("source and source_bytes are mutually exclusive")
    if source is not None:
        try:
            source_bytes = source.read_bytes()
        except OSError as exc:
            raise ClangASTImportError(f"cannot load Clang AST source bytes: {exc}") from exc
    return import_ast(root, kernel_name, ast_sha256=hashlib.sha256(raw).hexdigest(), source_bytes=source_bytes)
