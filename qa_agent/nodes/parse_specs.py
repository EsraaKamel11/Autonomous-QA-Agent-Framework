"""
qa_agent/nodes/parse_specs.py
───────────────────────────────
SpecParserAgent — node 2 of 12.

Decision boundary
─────────────────
Only answers: "What behaviours does the spec mandate for the changed endpoints?"
Never reads application code, never generates executable tests.

Inputs consumed from state
──────────────────────────
  state["change_manifest"]   ChangeManifest — affected_endpoints, change_summary

State keys written
──────────────────
  test_scenarios   List[TestScenario]
  current_phase    str
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional

import yaml
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from qa_agent.observability import log_node_execution
from qa_agent.prompts.spec_parser import SPEC_PARSER_SYSTEM, SPEC_PARSER_USER
from qa_agent.state import (
    AgentError,
    QAAgentState,
    ScenarioType,
    TestScenario,
)
from qa_agent.tools.spec_tools import (
    ParsedSpec,
    SpecSection,
    extract_spec_for_endpoint,
    find_spec_files,
    get_spec_summary,
    parse_openapi_spec,
)

_VOCAREUM_API_KEY  = os.environ.get("VOCAREUM_API_KEY",  "voc-placeholder")
_VOCAREUM_BASE_URL = os.environ.get("VOCAREUM_BASE_URL", "https://openai.vocareum.com/v1")
_NODE_NAME = "parse_specs"

# ──────────────────────────────────────────────────────────────────────────────
# Intermediate models — LLM outputs raw scenarios; node stamps scenario_id
# ──────────────────────────────────────────────────────────────────────────────

class _RawScenario(BaseModel):
    """LLM-facing TestScenario without the node-generated scenario_id."""
    endpoint:          str
    method:            str
    scenario_type:     ScenarioType
    description:       str
    preconditions:     List[str]      = Field(default_factory=list)
    expected_behavior: str
    expected_status:   Optional[int]  = None
    expected_body:     Optional[dict] = None
    priority:          int            = Field(default=2, ge=1, le=3)


class _ScenarioSet(BaseModel):
    scenarios: List[_RawScenario]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_parsed_spec() -> Optional[ParsedSpec]:
    """
    Discover and parse the first OpenAPI spec file found in the working tree.
    Returns None if no spec file exists or parsing fails.
    """
    spec_files = find_spec_files(".")
    if not spec_files:
        return None
    try:
        return parse_openapi_spec(spec_files[0], validate=False)
    except Exception:
        return None


def _build_spec_sections(
    affected_endpoints: list,
    parsed_spec: Optional[ParsedSpec],
) -> str:
    """
    Build a YAML block of spec sections relevant to each affected endpoint.
    Falls back to a minimal placeholder when the spec is unavailable.
    """
    if parsed_spec is None:
        return "(No OpenAPI spec found — SpecParserAgent will generate scenarios from PR context only)"

    parts: List[str] = []
    for ep in affected_endpoints:
        section: SpecSection = extract_spec_for_endpoint(
            method=ep.method,
            path=ep.path,
            parsed_spec=parsed_spec,
        )
        header = f"# {ep.method} {ep.path}  [change_type: {ep.change_type}]"
        body   = section.raw_section or "(endpoint not found in spec)"
        parts.append(f"{header}\n{body}")

    return "\n---\n".join(parts) if parts else "(No matching spec sections found)"


def _make_scenario_id(
    method: str,
    path: str,
    scenario_type: str,
    seq: int,
) -> str:
    """
    Generate a stable, deterministic scenario_id from its defining fields.
    Format: METHOD_path_tokens_scenariotype_NNN
    """
    clean_path = (
        path.replace("/", "_")
            .replace("{", "")
            .replace("}", "")
            .strip("_")
    )
    return f"{method.upper()}_{clean_path}_{scenario_type}_{seq:03d}"


def _raw_to_scenario(raw: _RawScenario, seq: int) -> TestScenario:
    """Convert LLM _RawScenario → canonical TestScenario with generated ID."""
    return TestScenario(
        scenario_id=_make_scenario_id(
            raw.method, raw.endpoint, raw.scenario_type.value, seq
        ),
        endpoint=raw.endpoint,
        method=raw.method.upper(),
        scenario_type=raw.scenario_type,
        description=raw.description,
        preconditions=raw.preconditions,
        expected_behavior=raw.expected_behavior,
        expected_status=raw.expected_status,
        expected_body=raw.expected_body,
        priority=raw.priority,
    )


def _fallback_scenarios(
    affected_endpoints: list,
) -> List[TestScenario]:
    """
    Generate minimal placeholder scenarios when the LLM call fails.
    Ensures the pipeline can continue with at least a happy-path stub.
    """
    fallback: List[TestScenario] = []
    for i, ep in enumerate(affected_endpoints):
        fallback.append(TestScenario(
            scenario_id=_make_scenario_id(
                ep.method, ep.path, ScenarioType.HAPPY_PATH.value, i
            ),
            endpoint=ep.path,
            method=ep.method.upper(),
            scenario_type=ScenarioType.HAPPY_PATH,
            description=f"[FALLBACK] Happy path for {ep.method.upper()} {ep.path}",
            preconditions=[],
            expected_behavior="Returns 2xx with valid response body",
            expected_status=200,
            expected_body=None,
            priority=1,
        ))
    return fallback


# ──────────────────────────────────────────────────────────────────────────────
# Node function
# ──────────────────────────────────────────────────────────────────────────────

def parse_specs(state: QAAgentState) -> dict:
    """
    SpecParserAgent: read the OpenAPI spec for changed endpoints and generate
    TestScenarios (happy path + edge + error + security).

    Pipeline position: Node 2.
    Input:  state["change_manifest"]
    Output: state["test_scenarios"]

    LLM call: yes — with_structured_output(_ScenarioSet) via Vocareum.
    Tool calls: spec_tools.find_spec_files, parse_openapi_spec,
                extract_spec_for_endpoint.
    """
    start_ts = time.time()
    manifest = state["change_manifest"]
    errors   = []

    # Guard: no change manifest means analyze_pr failed; pass through safely
    if manifest is None:
        log_node_execution(
            node_name=_NODE_NAME,
            input_summary={"change_manifest": "None — skipping"},
            output_summary={"test_scenarios": 0},
            routing_hint="retrieve_history",
            pr_number=state["pr_metadata"].pr_number,
        )
        return {"test_scenarios": [], "current_phase": _NODE_NAME}

    affected = manifest.affected_endpoints

    # ── Step 1: Load spec ──────────────────────────────────────────────────
    parsed_spec   = _load_parsed_spec()
    spec_sections = _build_spec_sections(affected, parsed_spec)

    affected_ep_json = json.dumps(
        [{"method": ep.method, "path": ep.path, "change_type": ep.change_type}
         for ep in affected],
        indent=2,
    )

    user_message = SPEC_PARSER_USER.format(
        change_summary=manifest.change_summary[:3000],
        affected_endpoints=affected_ep_json,
        spec_sections=spec_sections[:8000],
    )

    # ── Step 2: LLM call ───────────────────────────────────────────────────
    error_str: str | None = None
    scenarios:  List[TestScenario]

    try:
        llm = ChatOpenAI(
            model="gpt-4o",
            api_key=_VOCAREUM_API_KEY,
            base_url=_VOCAREUM_BASE_URL,
            temperature=0,
        ).with_structured_output(_ScenarioSet)

        scenario_set: _ScenarioSet = llm.invoke([
            {"role": "system", "content": SPEC_PARSER_SYSTEM},
            {"role": "user",   "content": user_message},
        ])

        scenarios = [
            _raw_to_scenario(raw, seq)
            for seq, raw in enumerate(scenario_set.scenarios)
        ]

    except Exception as exc:
        error_str = f"{type(exc).__name__}: {exc}"
        errors.append(AgentError(
            agent="SpecParserAgent",
            phase="parse_specs",
            error=error_str,
            recoverable=True,
        ))
        scenarios = _fallback_scenarios(affected)

    # ── Step 3: Log before returning ──────────────────────────────────────
    duration_ms = (time.time() - start_ts) * 1000
    type_counts: Dict[str, int] = {}
    for s in scenarios:
        type_counts[s.scenario_type.value] = type_counts.get(s.scenario_type.value, 0) + 1

    log_node_execution(
        node_name=_NODE_NAME,
        input_summary={
            "affected_endpoints": len(affected),
            "spec_found":         parsed_spec is not None,
            "change_summary":     manifest.change_summary[:100],
        },
        output_summary={
            "total_scenarios": len(scenarios),
            "by_type":         str(type_counts),
        },
        routing_hint="retrieve_history",
        pr_number=state["pr_metadata"].pr_number,
        duration_ms=duration_ms,
        error=error_str,
    )

    result: dict = {"test_scenarios": scenarios, "current_phase": _NODE_NAME}
    if errors:
        result["errors"] = errors
    return result
