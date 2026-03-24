"""
qa_agent/tools/spec_tools.py
─────────────────────────────
OpenAPI specification parsing and querying tools.

Decision boundary
─────────────────
These tools answer only:
  • "What does this OpenAPI spec declare?"
  • "What are the request/response contracts for this endpoint?"
  • "Where are the spec files in this repository?"

They never generate test scenarios, classify risk, or make routing decisions.
SpecParserAgent consumes the output of these tools to produce TestScenarios.

LLM calls: NONE. Pure file I/O + schema traversal.

Supported spec formats
──────────────────────
  • OpenAPI 3.0.x  (YAML or JSON)
  • Auto-discovery searches for openapi.yaml, swagger.yaml, api.yaml,
    openapi.json, swagger.json, and any file under docs/api/ or api/

Dependencies
────────────
  openapi-spec-validator>=0.7.0
  PyYAML>=6.0.0
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel


# ──────────────────────────────────────────────────────────────────────────────
# Local result models
# ──────────────────────────────────────────────────────────────────────────────

class SpecParameter(BaseModel):
    """A single parameter (path, query, header, cookie) from an operation."""
    name:     str
    location: str           # path | query | header | cookie  (OpenAPI "in")
    required: bool = False
    schema:   Optional[Dict[str, Any]] = None
    description: str = ""


class SpecResponse(BaseModel):
    """A declared response for one HTTP status code."""
    status_code: str        # "200", "404", "default"
    description: str = ""
    content_schema: Optional[Dict[str, Any]] = None   # resolved JSON Schema


class SpecOperation(BaseModel):
    """
    A single endpoint operation (method + path) from the OpenAPI spec.
    Carries everything SpecParserAgent needs to derive TestScenarios.
    """
    path:         str
    method:       str               # GET | POST | PUT | PATCH | DELETE | ...
    operation_id: Optional[str] = None
    summary:      str = ""
    description:  str = ""
    tags:         List[str] = []
    parameters:   List[SpecParameter] = []
    request_body: Optional[Dict[str, Any]] = None
    responses:    List[SpecResponse] = []
    security:     List[Dict[str, Any]] = []    # e.g. [{"bearerAuth": []}]
    deprecated:   bool = False


class ParsedSpec(BaseModel):
    """
    Complete parsed representation of an OpenAPI spec file.
    Returned by parse_openapi_spec(); consumed by SpecParserAgent.
    """
    spec_path:   str
    title:       str = ""
    version:     str = ""
    operations:  List[SpecOperation] = []
    components:  Dict[str, Any] = {}    # raw components section for $ref resolution
    raw:         Dict[str, Any] = {}    # full raw spec dict (for LLM context windows)


class SpecSection(BaseModel):
    """
    A focused slice of the spec relevant to one endpoint.
    Returned by extract_spec_for_endpoint(); passed to SelfHealingAgent
    when it needs to understand what the spec says about a failing test.
    """
    endpoint:    str
    method:      str
    operation:   Optional[SpecOperation] = None
    raw_section: str = ""               # YAML-serialised slice for LLM prompts


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

_SEARCH_NAMES = {
    "openapi.yaml", "openapi.json",
    "swagger.yaml", "swagger.json",
    "api.yaml",     "api.json",
    "api-spec.yaml","api-spec.json",
}

_SEARCH_DIRS = [".", "api", "docs", "docs/api", "spec", "openapi"]


def _load_raw_spec(spec_path: str) -> Dict[str, Any]:
    """Load and parse a YAML or JSON spec file. Raises on invalid file."""
    path = Path(spec_path)
    if not path.exists():
        raise FileNotFoundError(f"Spec file not found: {spec_path}")

    with path.open(encoding="utf-8") as fh:
        if spec_path.endswith((".yaml", ".yml")):
            return yaml.safe_load(fh)
        return json.load(fh)


def _validate_spec(raw: Dict[str, Any]) -> None:
    """
    Validate the spec against the OpenAPI 3.0 schema.
    Raises openapi_spec_validator.exceptions.OpenAPISpecValidatorError on failure.
    """
    try:
        from openapi_spec_validator import OpenAPIV30SpecValidator
        OpenAPIV30SpecValidator(raw).validate()
    except Exception:
        # Fall back to the newer API surface introduced in 0.6+
        try:
            from openapi_spec_validator import validate
            validate(raw)
        except Exception as exc:
            raise ValueError(f"OpenAPI spec validation failed: {exc}") from exc


def _resolve_schema_ref(
    ref: str, components: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Resolve a simple local $ref like '#/components/schemas/User'.
    Does NOT follow remote $refs — those require a full resolver.
    """
    if not ref.startswith("#/"):
        return None
    parts = ref.lstrip("#/").split("/")
    node: Any = components
    # parts = ["components", "schemas", "User"] — skip the leading "components"
    for part in parts[1:]:
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return None
    return node if isinstance(node, dict) else None


def _extract_content_schema(
    content: Dict[str, Any],
    components: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Pull the JSON Schema from an OpenAPI 'content' object.
    Prefers application/json; falls back to the first available media type.
    """
    media_type = content.get("application/json") or next(iter(content.values()), {})
    schema = media_type.get("schema", {})
    if "$ref" in schema:
        resolved = _resolve_schema_ref(schema["$ref"], {"components": components})
        return resolved or schema
    return schema or None


def _parse_parameters(
    params: List[Dict[str, Any]],
) -> List[SpecParameter]:
    return [
        SpecParameter(
            name=p.get("name", ""),
            location=p.get("in", "query"),
            required=p.get("required", False),
            schema=p.get("schema"),
            description=p.get("description", ""),
        )
        for p in params
        if isinstance(p, dict)
    ]


def _parse_responses(
    responses: Dict[str, Any],
    components: Dict[str, Any],
) -> List[SpecResponse]:
    result: List[SpecResponse] = []
    for status_code, resp_obj in responses.items():
        if not isinstance(resp_obj, dict):
            continue
        content = resp_obj.get("content", {})
        schema = _extract_content_schema(content, components) if content else None
        result.append(
            SpecResponse(
                status_code=str(status_code),
                description=resp_obj.get("description", ""),
                content_schema=schema,
            )
        )
    return result


def _parse_operations(raw: Dict[str, Any]) -> List[SpecOperation]:
    """Walk paths → methods and build the SpecOperation list."""
    components = raw.get("components", {})
    ops: List[SpecOperation] = []
    http_methods = {"get", "post", "put", "patch", "delete", "head", "options"}

    for path, path_item in raw.get("paths", {}).items():
        if not isinstance(path_item, dict):
            continue
        # Parameters defined at path level apply to all operations
        path_level_params = _parse_parameters(path_item.get("parameters", []))

        for method in http_methods:
            op = path_item.get(method)
            if not isinstance(op, dict):
                continue

            # Merge path-level params; operation-level ones override by name+location
            op_params = _parse_parameters(op.get("parameters", []))
            param_keys = {(p.name, p.location) for p in op_params}
            merged_params = op_params + [
                p for p in path_level_params
                if (p.name, p.location) not in param_keys
            ]

            # Request body
            request_body: Optional[Dict[str, Any]] = None
            if "requestBody" in op:
                rb = op["requestBody"]
                content = rb.get("content", {})
                request_body = {
                    "required": rb.get("required", False),
                    "schema": (
                        _extract_content_schema(content, components)
                        if content else None
                    ),
                    "description": rb.get("description", ""),
                }

            ops.append(
                SpecOperation(
                    path=path,
                    method=method.upper(),
                    operation_id=op.get("operationId"),
                    summary=op.get("summary", ""),
                    description=op.get("description", ""),
                    tags=op.get("tags", []),
                    parameters=merged_params,
                    request_body=request_body,
                    responses=_parse_responses(op.get("responses", {}), components),
                    security=op.get("security", raw.get("security", [])),
                    deprecated=op.get("deprecated", False),
                )
            )
    return ops


# ──────────────────────────────────────────────────────────────────────────────
# Public tool functions
# ──────────────────────────────────────────────────────────────────────────────

def parse_openapi_spec(spec_path: str, validate: bool = True) -> ParsedSpec:
    """
    Parse an OpenAPI 3.0 spec file (YAML or JSON) into a structured object.

    Decision boundary: loads and structures the spec as written. Does NOT
    interpret which endpoints are high-risk or generate test scenarios —
    SpecParserAgent does that after receiving this output.

    Parameters
    ----------
    spec_path : str    Absolute or relative path to the spec file.
    validate  : bool   Run openapi-spec-validator before parsing (default True).
                       Set False to tolerate minor spec violations in development.

    Returns
    -------
    ParsedSpec
        Fully structured spec ready for SpecParserAgent to consume.

    Raises
    ------
    FileNotFoundError   If the spec file does not exist.
    ValueError          If the spec fails OpenAPI 3.0 schema validation.
    """
    raw = _load_raw_spec(spec_path)

    if validate:
        _validate_spec(raw)

    info = raw.get("info", {})
    return ParsedSpec(
        spec_path=spec_path,
        title=info.get("title", ""),
        version=info.get("version", ""),
        operations=_parse_operations(raw),
        components=raw.get("components", {}),
        raw=raw,
    )


def extract_spec_for_endpoint(
    method: str,
    path: str,
    parsed_spec: ParsedSpec,
) -> SpecSection:
    """
    Return the spec section relevant to a single endpoint + method.

    Decision boundary: extracts and serialises spec data. Does NOT decide
    whether the spec is satisfied or violated — SelfHealingAgent does that
    when deciding how to patch a failing test.

    Parameters
    ----------
    method      : str         HTTP method in any case (e.g. "GET", "get").
    path        : str         OpenAPI path template (e.g. "/users/{id}").
    parsed_spec : ParsedSpec  Output of parse_openapi_spec().

    Returns
    -------
    SpecSection
        The matching operation plus a YAML-serialised raw_section string
        suitable for insertion into LLM prompts.
        If the endpoint is not found, returns an empty SpecSection so that
        callers do not need to handle None.
    """
    method_upper = method.upper()
    operation: Optional[SpecOperation] = next(
        (
            op for op in parsed_spec.operations
            if op.method == method_upper and op.path == path
        ),
        None,
    )

    raw_section: Dict[str, Any] = {}
    if operation:
        raw_section = {
            "path": operation.path,
            "method": operation.method,
            "summary": operation.summary,
            "parameters": [p.model_dump() for p in operation.parameters],
            "requestBody": operation.request_body,
            "responses": [r.model_dump() for r in operation.responses],
            "security": operation.security,
        }

    return SpecSection(
        endpoint=path,
        method=method_upper,
        operation=operation,
        raw_section=yaml.dump(raw_section, default_flow_style=False),
    )


def find_spec_files(repo_path: str = ".") -> List[str]:
    """
    Discover OpenAPI spec files inside a repository directory.

    Decision boundary: returns file paths only. Does NOT load or validate
    them — the caller (typically the parse_specs node setup) decides which
    path to parse.

    Search strategy (in order):
      1. Well-known filenames at repo root and common subdirs.
      2. Any .yaml/.json file whose first line contains "openapi:" or
         "swagger:".

    Parameters
    ----------
    repo_path : str   Root directory to search (default: current directory).

    Returns
    -------
    List[str]
        Absolute paths to candidate spec files, ordered by specificity
        (well-known names first, then heuristic matches).
    """
    root = Path(repo_path).resolve()
    found: List[str] = []
    seen: set = set()

    # Pass 1 — well-known filenames in common directories
    for dir_rel in _SEARCH_DIRS:
        dir_abs = root / dir_rel
        if not dir_abs.is_dir():
            continue
        for name in _SEARCH_NAMES:
            candidate = dir_abs / name
            if candidate.exists() and str(candidate) not in seen:
                found.append(str(candidate))
                seen.add(str(candidate))

    # Pass 2 — heuristic scan: first-line "openapi:" or "swagger:" marker
    for ext in ("*.yaml", "*.yml", "*.json"):
        for candidate in root.rglob(ext):
            if str(candidate) in seen:
                continue
            # Skip node_modules / .git / __pycache__
            if any(p in candidate.parts for p in
                   ("node_modules", ".git", "__pycache__", ".venv", "venv")):
                continue
            try:
                with candidate.open(encoding="utf-8", errors="ignore") as fh:
                    first_line = fh.readline().strip().lower()
                if first_line.startswith(("openapi:", "swagger:", '{"openapi"', '{"swagger"')):
                    found.append(str(candidate))
                    seen.add(str(candidate))
            except (OSError, UnicodeDecodeError):
                pass

    return found


def get_spec_summary(parsed_spec: ParsedSpec) -> str:
    """
    Return a compact text summary of the spec for use in LLM context windows.

    Decision boundary: summarises only. Never filters by relevance or risk
    — SpecParserAgent decides which operations matter for the current PR.

    Returns a human-readable string listing all operation IDs, methods,
    paths, and tag groups — small enough to fit in a single LLM message.
    """
    lines = [
        f"API: {parsed_spec.title} (v{parsed_spec.version})",
        f"Spec: {parsed_spec.spec_path}",
        f"Operations: {len(parsed_spec.operations)}",
        "",
    ]
    tag_groups: Dict[str, List[str]] = {}
    for op in parsed_spec.operations:
        tags = op.tags or ["(untagged)"]
        entry = (
            f"  {op.method:<7} {op.path}"
            + (f"  [{op.operation_id}]" if op.operation_id else "")
            + (" [DEPRECATED]" if op.deprecated else "")
        )
        for tag in tags:
            tag_groups.setdefault(tag, []).append(entry)

    for tag, entries in sorted(tag_groups.items()):
        lines.append(f"[{tag}]")
        lines.extend(entries)
        lines.append("")

    return "\n".join(lines)
