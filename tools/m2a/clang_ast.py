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
CUDA_DEVICE_ATTR = "CUDADeviceAttr"
FACADE_BUILTIN_SCHEMA = "cuda4as-clang-cuda-facade-builtins-v1"
FACADE_MANIFEST_SCHEMA = "cuda4as-clang-cuda-facade-manifest-v1"
FACADE_VERSION = "1.0.0"
FACADE_HEADER_REL = "tools/m2a/clang_cuda_facade/cuda_runtime.h"
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


def _paths_match(recorded: str, expected: str) -> bool:
    """Compare an AST location path with an exact pinned source path."""

    recorded_norm = recorded.replace("\\", "/")
    expected_norm = expected.replace("\\", "/")
    if recorded_norm == expected_norm:
        return True
    try:
        return Path(recorded).is_absolute() and Path(expected).is_absolute() and Path(recorded).resolve() == Path(expected).resolve()
    except OSError:
        return False


def _source_for_file(
    file_name: str | None,
    source_bytes: bytes | None,
    source_bytes_by_file: list[tuple[str, bytes]] | None,
) -> bytes | None:
    if source_bytes_by_file is None:
        return source_bytes
    if not file_name:
        return None
    for expected, data in source_bytes_by_file:
        if _paths_match(file_name, expected):
            return data
    # A named location that cannot be tied to one of the unchanged inputs must
    # not accidentally use bytes from a different source file.
    return None


def _resolve_capture_path(file_name: str, capture_working_directory: Path | None) -> str:
    """Resolve an emitted relative filename only with an explicit capture cwd."""

    normalized = file_name.replace("\\", "/")
    path = Path(normalized)
    if path.is_absolute() or capture_working_directory is None:
        return normalized
    return str((capture_working_directory / path).resolve())


_ADDRESS_SPACE_RE = re.compile(
    r"__attribute__\s*\(\(\s*address_space\s*\(\s*([^)]*?)\s*\)\s*\)\)"
)
_CUDA_ADDRESS_SPACE_NAMES = {
    "__global__": "global",
    "global": "global",
    "__constant__": "constant",
    "constant": "constant",
    "__shared__": "shared",
    "shared": "shared",
    "__generic__": "generic",
    "generic": "generic",
}


def _strip_type_address_annotations(text: str) -> str:
    """Remove address-space spelling from a type before scalar normalization."""

    text = _ADDRESS_SPACE_RE.sub(" ", text)
    text = re.sub(r"__(?:global|constant|shared|generic)__", " ", text)
    return " ".join(text.split())


def _point(
    candidate: Any,
    source_bytes: bytes | None = None,
    source_bytes_by_file: list[tuple[str, bytes]] | None = None,
) -> dict[str, Any] | None:
    if not isinstance(candidate, dict):
        return None
    offset = candidate.get("offset")
    result: dict[str, Any] = {}
    file_name = candidate.get("file")
    if isinstance(file_name, str) and file_name:
        result["file"] = file_name
    included_from = candidate.get("includedFrom")
    if isinstance(included_from, dict):
        result["included_from"] = included_from
    if isinstance(offset, int) and offset >= 0:
        result["offset"] = offset
    line, column = candidate.get("line"), candidate.get("col")
    if isinstance(line, int) and isinstance(column, int):
        result.update({"line": line, "column": column, "provenance": "exact"})
        return result
    offset_source = _source_for_file(file_name if isinstance(file_name, str) else None, source_bytes, source_bytes_by_file)
    if "offset" in result and offset_source is not None and offset <= len(offset_source):
        prefix = offset_source[:offset]
        result["line"] = prefix.count(b"\n") + 1
        last_newline = prefix.rfind(b"\n")
        result["column"] = offset - (last_newline + 1) + 1
        result["provenance"] = "exact_offset"
        return result
    return None


class _LocationNormalizer:
    """Reconstruct JSONNodeDumper's stateful filename/line elision."""

    _COORDINATE_KEYS = {"file", "line", "col", "offset", "tokLen"}

    def __init__(
        self,
        *,
        source_bytes: bytes | None,
        source_bytes_by_file: list[tuple[str, bytes]] | None,
        capture_working_directory: Path | None,
    ) -> None:
        self.source_bytes = source_bytes
        self.source_bytes_by_file = source_bytes_by_file
        self.capture_working_directory = capture_working_directory
        self.last_file: str | None = None
        self.last_line: int | None = None
        self.points: dict[int, dict[str, Any]] = {}
        self.errors: dict[int, str] = {}

    def _record(self, candidate: dict[str, Any]) -> dict[str, Any]:
        raw_file = candidate.get("file")
        has_file = raw_file is not None
        if has_file and (not isinstance(raw_file, str) or not raw_file):
            raise ClangASTImportError("compiler source location has an invalid filename")
        if has_file:
            file_name = _resolve_capture_path(raw_file, self.capture_working_directory)
        else:
            file_name = self.last_file
            if file_name is not None and not Path(file_name).is_absolute() and self.capture_working_directory is None:
                raise ClangASTImportError(
                    "compiler source location recovers a relative filename without capture working directory"
                )
        if file_name is None:
            raise ClangASTImportError("compiler source location elides its first filename")

        raw_line = candidate.get("line")
        has_line = raw_line is not None
        if has_line and (not isinstance(raw_line, int) or isinstance(raw_line, bool) or raw_line <= 0):
            raise ClangASTImportError("compiler source location has an invalid line")
        line = raw_line if has_line else self.last_line
        if line is None and candidate.get("offset") is None:
            raise ClangASTImportError("compiler source location elides its first line")
        column = candidate.get("col")
        if column is not None and (not isinstance(column, int) or isinstance(column, bool) or column <= 0):
            raise ClangASTImportError("compiler source location has an invalid column")
        offset = candidate.get("offset")
        if offset is not None and (not isinstance(offset, int) or isinstance(offset, bool) or offset < 0):
            raise ClangASTImportError("compiler source location has an invalid offset")

        result: dict[str, Any] = {"file": file_name}
        if offset is not None:
            result["offset"] = offset
            source = _source_for_file(file_name, self.source_bytes, self.source_bytes_by_file)
            if source is None or offset > len(source):
                raise ClangASTImportError("compiler source location offset has no identified source bytes")
            prefix = source[:offset]
            offset_line = prefix.count(b"\n") + 1
            last_newline = prefix.rfind(b"\n")
            offset_column = offset - (last_newline + 1) + 1
            if has_line and offset_line != line:
                raise ClangASTImportError("compiler source location conflicts with pinned source bytes")
            if column is not None and offset_column != column:
                raise ClangASTImportError("compiler source location conflicts with pinned source bytes")
            line = offset_line
            column = offset_column
            result["provenance"] = "exact_offset" if has_file and has_line else "exact_offset_deelided"
        else:
            if column is None:
                raise ClangASTImportError("compiler source location has no valid column")
            result["provenance"] = "exact" if has_file and has_line else "exact_deelided"
        result.update({"line": line, "column": column})

        self.last_file = file_name
        self.last_line = line
        return result

    def _scan_location_value(self, value: Any) -> None:
        if not isinstance(value, dict):
            return
        if self._COORDINATE_KEYS.intersection(value):
            try:
                point = self._record(value)
            except ClangASTImportError as exc:
                self.errors[id(value)] = str(exc)
            else:
                if isinstance(value.get("includedFrom"), dict):
                    point["included_from"] = value["includedFrom"]
                self.points[id(value)] = point
        # spellingLoc/expansionLoc are visited in their JSON insertion order.
        for key, nested in value.items():
            if key in {"spellingLoc", "expansionLoc"}:
                self._scan_location_value(nested)

    def scan(self, value: Any) -> None:
        """Walk records in parsed JSON insertion order, matching emission order."""

        if isinstance(value, dict):
            for key, nested in value.items():
                if key in {"loc", "range"}:
                    if key == "range" and isinstance(nested, dict):
                        for range_key, point in nested.items():
                            if range_key in {"begin", "end"}:
                                self._scan_location_value(point)
                    else:
                        self._scan_location_value(nested)
                elif key in {"inner", "referencedDecl", "definitionData"}:
                    self.scan(nested)
                elif isinstance(nested, (dict, list)):
                    self.scan(nested)
        elif isinstance(value, list):
            for nested in value:
                self.scan(nested)

    def resolve(self, candidate: Any) -> dict[str, Any] | None:
        if not isinstance(candidate, dict):
            return None
        candidate_id = id(candidate)
        if candidate_id in self.errors:
            raise ClangASTImportError(self.errors[candidate_id])
        point = self.points.get(candidate_id)
        return dict(point) if point is not None else None


def _location(
    node: dict[str, Any],
    fallback: dict[str, Any] | None = None,
    *,
    source_bytes: bytes | None = None,
    source_bytes_by_file: list[tuple[str, bytes]] | None = None,
    location_normalizer: _LocationNormalizer | None = None,
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
        end = span.get("end")
        candidates.append(end)
        if isinstance(end, dict):
            candidates.extend([end.get("expansionLoc"), end.get("spellingLoc")])
    for candidate in candidates:
        point = (
            location_normalizer.resolve(candidate)
            if location_normalizer is not None
            else _point(candidate, source_bytes, source_bytes_by_file)
        )
        if point is not None:
            # Some Clang records put the file on the range point rather than
            # the shorthand loc. Preserve it for provenance checks.
            if "file" not in point:
                for sibling in candidates:
                    if isinstance(sibling, dict) and isinstance(sibling.get("file"), str):
                        point["file"] = sibling["file"]
                        break
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
    if location.get("provenance") not in {"exact", "exact_offset", "exact_deelided", "exact_offset_deelided", "inherited"}:
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
            if key in typ:
                raise ClangASTImportError(
                    f"compiler AST type contains non-emitted address-space key {key!r}"
                )
    for key in ("addressSpace", "address_space"):
        if key in node:
            raise ClangASTImportError(
                f"compiler AST node contains non-emitted address-space key {key!r}"
            )
    match = _ADDRESS_SPACE_RE.search(qual_type)
    if match:
        spelling = match.group(1).strip().lower()
        return _CUDA_ADDRESS_SPACE_NAMES.get(spelling, spelling)
    for spelling, canonical in _CUDA_ADDRESS_SPACE_NAMES.items():
        if re.search(rf"(?<!\w){re.escape(spelling)}(?!\w)", qual_type, flags=re.IGNORECASE):
            return canonical
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
        base = _strip_type_address_annotations(base)
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
    scalar = normalize_scalar(_strip_type_address_annotations(compact))
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
    raise ClangASTImportError("compiler DeclRefExpr has no referenced declaration name")


class SemanticImporter:
    def __init__(
        self,
        root: dict[str, Any],
        function: dict[str, Any],
        source_bytes: bytes | None = None,
        source_path: Path | None = None,
        facade_contract: dict[str, Any] | None = None,
        capture_working_directory: Path | None = None,
    ):
        self.root = root
        self.function = function
        self.source_bytes = source_bytes
        self.source_path = source_path
        self.facade_contract = facade_contract
        self.capture_working_directory = capture_working_directory
        self.source_bytes_by_file: list[tuple[str, bytes]] = []
        if source_path is not None and source_bytes is not None:
            self.source_bytes_by_file.append((str(source_path), source_bytes))
            source_resolved = source_path if source_path.is_absolute() or capture_working_directory is None else capture_working_directory / source_path
            self.source_bytes_by_file.append((str(source_resolved.resolve()), source_bytes))
        elif source_path is not None:
            try:
                source_data = source_path.read_bytes()
            except OSError as exc:
                raise ClangASTImportError(f"cannot load compiler source bytes: {exc}") from exc
            self.source_bytes = source_data
            self.source_bytes_by_file.append((str(source_path), source_data))
            source_resolved = source_path if source_path.is_absolute() or capture_working_directory is None else capture_working_directory / source_path
            self.source_bytes_by_file.append((str(source_resolved.resolve()), source_data))
        if isinstance(facade_contract, dict):
            header_path = facade_contract.get("header_path")
            if isinstance(header_path, str):
                try:
                    header_bytes = Path(header_path).read_bytes()
                except OSError as exc:
                    raise ClangASTImportError(f"cannot load pinned CUDA facade header: {exc}") from exc
                self.source_bytes_by_file.append((header_path, header_bytes))
                header_resolved = Path(header_path) if Path(header_path).is_absolute() or capture_working_directory is None else capture_working_directory / header_path
                self.source_bytes_by_file.append((str(header_resolved.resolve()), header_bytes))
                manifest = facade_contract.get("manifest")
                builtin_contract = facade_contract.get("builtin_contract")
                if isinstance(manifest, dict) and isinstance(manifest.get("header"), dict):
                    manifest_header = manifest["header"].get("path")
                    if isinstance(manifest_header, str) and isinstance(header_path, str):
                        self.source_bytes_by_file.append((manifest_header, header_bytes))
                        manifest_resolved = Path(manifest_header) if Path(manifest_header).is_absolute() or capture_working_directory is None else capture_working_directory / manifest_header
                        self.source_bytes_by_file.append((str(manifest_resolved.resolve()), header_bytes))
        self.location_normalizer = _LocationNormalizer(
            source_bytes=self.source_bytes,
            source_bytes_by_file=self.source_bytes_by_file,
            capture_working_directory=capture_working_directory,
        )
        self.location_normalizer.scan(root)
        self.env: dict[str, dict[str, Any]] = {}
        self.env_by_id: dict[str, dict[str, Any]] = {}
        self.declarations_by_id: dict[str, dict[str, Any]] = {}
        for node in walk(root):
            node_id = node.get("id")
            if isinstance(node_id, str) and node.get("kind") in {"VarDecl", "ParmVarDecl"}:
                # The first declaration record is the canonical one; later
                # referencedDecl snapshots must not shadow its attributes.
                self.declarations_by_id.setdefault(node_id, node)

    def _location_for(self, node: dict[str, Any], fallback: dict[str, Any] | None = None) -> dict[str, Any]:
        return _location(
            node,
            fallback,
            source_bytes=self.source_bytes,
            source_bytes_by_file=self.source_bytes_by_file,
            location_normalizer=self.location_normalizer,
        )

    def _point_for(self, candidate: Any) -> dict[str, Any] | None:
        return self.location_normalizer.resolve(candidate)

    def _builtin_location(self, node: dict[str, Any], expected_paths: tuple[str, ...]) -> dict[str, Any]:
        """Prefer a spelling/range point proven to belong to the pinned facade."""

        loc = node.get("loc")
        candidates: list[Any] = [loc]
        if isinstance(loc, dict):
            candidates.extend([loc.get("spellingLoc"), loc.get("expansionLoc")])
        span = node.get("range")
        if isinstance(span, dict):
            begin = span.get("begin")
            candidates.append(begin)
            if isinstance(begin, dict):
                candidates.extend([begin.get("spellingLoc"), begin.get("expansionLoc")])
            end = span.get("end")
            candidates.append(end)
            if isinstance(end, dict):
                candidates.extend([end.get("spellingLoc"), end.get("expansionLoc")])
        for candidate in candidates:
            point = self._point_for(candidate)
            if point is None:
                continue
            if "file" not in point:
                for sibling in candidates:
                    if isinstance(sibling, dict) and isinstance(sibling.get("file"), str):
                        point["file"] = sibling["file"]
                        break
            file_name = point.get("file")
            if isinstance(file_name, str) and any(_paths_match(file_name, expected) for expected in expected_paths):
                return point
        return self._location_for(node)

    def _facade_builtin(self, name: str) -> tuple[dict[str, Any], tuple[str, ...], bytes]:
        contract = self.facade_contract
        if not isinstance(contract, dict):
            raise ClangASTImportError(f"builtin {name} has no verified CUDA facade contract")
        builtin_contract = contract.get("builtin_contract")
        if not isinstance(builtin_contract, dict) or builtin_contract.get("schema") != FACADE_BUILTIN_SCHEMA:
            raise ClangASTImportError("compiler CUDA facade builtin contract has an unsupported schema")
        if builtin_contract.get("device_builtin_semantics") != {
            "spelling": "device_builtin",
            "status": "IGNORED_NON_AST",
            "ast_node": 0,
            "ast_json_child": False,
        }:
            raise ClangASTImportError("compiler CUDA facade contract has an invalid device_builtin serialization contract")
        declarations = builtin_contract.get("declarations")
        if not isinstance(declarations, dict) or set(declarations) != BUILTINS:
            raise ClangASTImportError("compiler CUDA facade builtin contract has the wrong declaration set")
        declaration = declarations.get(name)
        if not isinstance(declaration, dict):
            raise ClangASTImportError(f"builtin {name} is not in the verified CUDA facade contract")
        header_path = contract.get("header_path")
        manifest = contract.get("manifest")
        manifest_header_path = builtin_contract.get("header_path")
        if not isinstance(header_path, str) or not isinstance(manifest_header_path, str):
            raise ClangASTImportError("compiler CUDA facade contract has no immutable header identity")
        if (
            not isinstance(manifest, dict)
            or manifest.get("schema") != FACADE_MANIFEST_SCHEMA
            or manifest.get("version") != FACADE_VERSION
            or manifest.get("builtin_contract") != builtin_contract
        ):
            raise ClangASTImportError("compiler CUDA facade builtin contract is not bound to its manifest")
        manifest_header = manifest.get("header")
        if (
            not isinstance(manifest_header, dict)
            or manifest_header.get("path") != manifest_header_path
            or manifest_header_path != FACADE_HEADER_REL
        ):
            raise ClangASTImportError("compiler CUDA facade header path is not consistently pinned")
        manifest_path = contract.get("manifest_path")
        manifest_hash = contract.get("manifest_sha256")
        if not isinstance(manifest_path, str) or not isinstance(manifest_hash, str):
            raise ClangASTImportError("compiler CUDA facade manifest has no immutable identity")
        try:
            manifest_bytes = Path(manifest_path).read_bytes()
            disk_manifest = json.loads(manifest_bytes.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ClangASTImportError(f"cannot load pinned CUDA facade manifest: {exc}") from exc
        if hashlib.sha256(manifest_bytes).hexdigest() != manifest_hash or disk_manifest != manifest:
            raise ClangASTImportError("compiler CUDA facade manifest does not match its loaded contract")
        try:
            header_bytes = Path(header_path).read_bytes()
        except OSError as exc:
            raise ClangASTImportError(f"cannot load pinned CUDA facade header: {exc}") from exc
        expected_hash = builtin_contract.get("header_sha256")
        if not isinstance(expected_hash, str) or hashlib.sha256(header_bytes).hexdigest() != expected_hash:
            raise ClangASTImportError("compiler CUDA facade header hash does not match its builtin contract")
        if contract.get("header_sha256") != expected_hash or manifest_header.get("sha256") != expected_hash:
            raise ClangASTImportError("compiler CUDA facade header hash is not consistently pinned")
        header_lines = header_bytes.decode("utf-8").splitlines()
        line = declaration.get("line")
        column = declaration.get("column")
        anchor = declaration.get("source_anchor")
        if (
            not isinstance(line, int)
            or line <= 0
            or not isinstance(column, int)
            or column <= 0
            or declaration.get("qual_type") != "const dim3"
            or not isinstance(anchor, str)
            or line > len(header_lines)
            or header_lines[line - 1].strip() != anchor
        ):
            raise ClangASTImportError(f"compiler CUDA facade contract has invalid {name} declaration anchor")
        expected_paths = [header_path, manifest_header_path]
        # Resolve the pinned contract paths from the importer process/contract,
        # never from the capture cwd supplied by the AST. A wrong capture cwd
        # must therefore fail the facade provenance check instead of redefining
        # what the pinned header means.
        expected_paths.extend(
            str(Path(item).resolve())
            for item in (header_path, manifest_header_path)
            if not Path(item).is_absolute()
        )
        return declaration, tuple(expected_paths), header_bytes

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
        if ref.get("name") != name:
            raise ClangASTImportError(f"builtin {name} reference name does not match its compiler declaration identity")
        declaration = self.declarations_by_id.get(ref_id)
        if not isinstance(declaration, dict) or declaration.get("kind") != "VarDecl" or declaration.get("name") != name:
            raise ClangASTImportError(f"builtin {name} does not resolve to its compiler VarDecl")
        attrs = _attrs(declaration)
        if CUDA_DEVICE_ATTR not in attrs:
            raise ClangASTImportError(f"builtin {name} lacks direct CUDA device compiler attribute")
        unsupported_cuda = {item for item in attrs if item.startswith("CUDA") and item != CUDA_DEVICE_ATTR}
        if unsupported_cuda:
            raise ClangASTImportError(f"builtin {name} has unsupported CUDA compiler attributes: {sorted(unsupported_cuda)}")
        contract_declaration, expected_paths, _header_bytes = self._facade_builtin(name)
        ref_type = None
        try:
            ref_type = _qual_type(ref)
        except ClangASTImportError:
            pass
        decl_type = _qual_type(declaration)
        expected_type = contract_declaration["qual_type"]
        if ref_type not in {None, "dim3", expected_type}:
            raise ClangASTImportError(f"builtin {name} reference is not typed as dim3")
        if decl_type != expected_type:
            raise ClangASTImportError(f"builtin {name} declaration type is not the pinned facade type")
        location = _require_location("builtin VarDecl", self._builtin_location(declaration, expected_paths))
        if location.get("provenance") not in {"exact", "exact_offset", "exact_deelided", "exact_offset_deelided"}:
            raise ClangASTImportError(f"builtin {name} has no exact facade source provenance")
        file_name = location.get("file")
        if not isinstance(file_name, str) or not any(_paths_match(file_name, expected) for expected in expected_paths):
            raise ClangASTImportError(f"builtin {name} source location is outside the pinned CUDA facade")
        if location.get("line") != contract_declaration["line"] or location.get("column") != contract_declaration["column"]:
            raise ClangASTImportError(f"builtin {name} source location does not match the pinned facade declaration")

    def expr(self, node: dict[str, Any], inherited_location: dict[str, Any] | None = None) -> dict[str, Any]:
        kind = node.get("kind")
        loc = _require_location(str(kind), self._location_for(node, inherited_location))
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
        loc = _require_location(str(kind), self._location_for(node, inherited_location))
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
            if lhs["kind"] != "index" or lhs.get("address_space") not in {"unknown", "global"}:
                raise ClangASTImportError("compiler assignment target is not a supported kernel buffer element")
            if rhs["type"] != lhs["type"] and not (lhs["type"] == "float" and rhs["type"] == "int"):
                raise ClangASTImportError(f"compiler store type {rhs['type']} does not match {lhs['type']}")
            return {"kind": "assign", "lhs": lhs, "operator": "=", "rhs": rhs, "source_location": loc}
        raise ClangASTImportError(f"unsupported compiler statement node {kind!r}")

    def kernel(self) -> tuple[dict[str, Any], dict[str, Any]]:
        if "attrs" in self.function or "isCUDAKernel" in self.function:
            raise ClangASTImportError("compiler FunctionDecl uses the rejected legacy CUDA kernel schema")
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
        function_location = _require_location("FunctionDecl", self._location_for(self.function))
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
                    "source_location": _require_location("ParmVarDecl", self._location_for(child)),
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
            "body_location": _require_location("CompoundStmt", self._location_for(body)),
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
            "location_normalization": {
                "schema": "cuda4as-clang-json-location-normalization-v1",
                "serializer_state": ["LastLocFilename", "LastLocLine"],
                "capture_working_directory": str(self.capture_working_directory) if self.capture_working_directory is not None else "NOT_RECORDED",
                "elided_provenance": ["exact_deelided", "exact_offset_deelided"],
                "parent_fallback_provenance": "inherited",
            },
        }
        return kernel, metadata


def import_ast(
    root: dict[str, Any],
    kernel_name: str,
    *,
    ast_sha256: str | None = None,
    source_bytes: bytes | None = None,
    source_path: Path | None = None,
    facade_contract: dict[str, Any] | None = None,
    capture_working_directory: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if root.get("kind") != "TranslationUnitDecl":
        raise ClangASTImportError("compiler AST root is not TranslationUnitDecl")
    matches = [node for node in walk(root) if node.get("kind") == "FunctionDecl" and node.get("name") == kernel_name]
    if len(matches) != 1:
        raise ClangASTImportError(f"expected one compiler kernel FunctionDecl {kernel_name!r}, found {len(matches)}")
    kernel, metadata = SemanticImporter(
        root,
        matches[0],
        source_bytes=source_bytes,
        source_path=source_path,
        facade_contract=facade_contract,
        capture_working_directory=capture_working_directory,
    ).kernel()
    if ast_sha256 is not None:
        metadata["ast_sha256"] = ast_sha256
    return kernel, metadata


def import_ast_file(
    path: Path,
    kernel_name: str,
    *,
    source: Path | None = None,
    source_bytes: bytes | None = None,
    facade_contract: dict[str, Any] | None = None,
    capture_working_directory: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
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
    return import_ast(
        root,
        kernel_name,
        ast_sha256=hashlib.sha256(raw).hexdigest(),
        source_bytes=source_bytes,
        source_path=source,
        facade_contract=facade_contract,
        capture_working_directory=capture_working_directory,
    )
