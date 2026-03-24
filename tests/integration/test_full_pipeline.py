"""
tests/integration/test_full_pipeline.py
─────────────────────────────────────────
End-to-end integration test for the QA Agent LangGraph pipeline.

Strategy
────────
- All LLM calls in node modules are patched to raise Exception immediately,
  which triggers each node's fallback logic (no actual HTTP calls made).
- execute_all_scripts is mocked to return a clean ExecutionReport (2 passed)
  so the pipeline does not actually invoke pytest as a subprocess.
- ChromaDB retrieval/storage calls are mocked to return [] / None, avoiding
  ChromaDB server connectivity requirements.
- GitHub and Jira tools default to mock mode (USE_LIVE_GITHUB=false,
  USE_LIVE_JIRA=false) via conftest.py.
- A unique thread_id is used per test so SqliteSaver state does not bleed
  between runs.

Assertions
──────────
- Pipeline completes without raising
- Final state contains: coverage_report, execution_report, commit_status
- Log files (decisions.jsonl) are created and contain valid JSON lines
- current_phase ends at "finalize"
"""

from __future__ import annotations

import contextlib
import json
import os
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from qa_agent.state import (
    ExecutionReport,
    MergeRecommendation,
    PRMetadata,
    TestResult,
    initial_state,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def mock_pr_metadata() -> PRMetadata:
    return PRMetadata(
        pr_number=999,
        repo="integration/test-repo",
        head_sha="int_head_sha_abc123",
        base_sha="int_base_sha_def456",
        author="integration-tester",
        title="integration: test full pipeline",
        description="End-to-end pipeline integration test with fully mocked LLMs.",
    )


@pytest.fixture(scope="module")
def mock_execution_report() -> ExecutionReport:
    """Clean execution report: all 2 tests passed, zero failures."""
    return ExecutionReport(
        total=2, passed=2, failed=0, errors=0, skipped=0,
        duration_seconds=0.05,
        per_test_results=[
            TestResult(
                test_id="integration::test_api_returns_200",
                scenario_id="GET_users_id_happy_path_000",
                status="passed",
                duration_ms=25.0,
            ),
            TestResult(
                test_id="integration::test_error_response_schema",
                scenario_id="GET_users_id_error_case_001",
                status="passed",
                duration_ms=20.0,
            ),
        ],
    )


def _llm_error() -> Exception:
    """Consistent exception used to trigger all node fallback paths."""
    return Exception("Mock: LLM unavailable in integration tests")


# ──────────────────────────────────────────────────────────────────────────────
# Main integration test
# ──────────────────────────────────────────────────────────────────────────────

_LLM_PATCH_TARGETS = [
    "qa_agent.nodes.analyze_pr.ChatOpenAI",
    "qa_agent.nodes.parse_specs.ChatOpenAI",
    "qa_agent.nodes.generate_tests.ChatOpenAI",
    "qa_agent.nodes.self_heal.ChatOpenAI",
    "qa_agent.nodes.evaluate_coverage.ChatOpenAI",
    "qa_agent.nodes.augment_tests.ChatOpenAI",
    "qa_agent.nodes.generate_defects.ChatOpenAI",
]


def test_full_pipeline_completes(mock_pr_metadata, mock_execution_report):
    """
    Full pipeline runs without raising and produces a final state with all
    required keys populated.
    """
    from qa_agent.graph import compiled_graph

    thread_id = f"integration-{uuid.uuid4().hex[:8]}"
    config    = {"configurable": {"thread_id": thread_id}}
    state     = initial_state(mock_pr_metadata)

    err = _llm_error()

    with contextlib.ExitStack() as stack:
        # Patch all node-level ChatOpenAI references to raise immediately
        for target in _LLM_PATCH_TARGETS:
            stack.enter_context(patch(target, side_effect=err))

        # Mock execute_all_scripts to avoid actually running pytest
        stack.enter_context(
            patch(
                "qa_agent.nodes.execute_tests.execute_all_scripts",
                return_value=mock_execution_report,
            )
        )

        # Mock ChromaDB calls to avoid server connectivity
        stack.enter_context(
            patch("qa_agent.nodes.retrieve_history.retrieve_similar_tests",
                  return_value=[])
        )
        stack.enter_context(
            patch("qa_agent.nodes.store_tests.store_test_in_memory",
                  return_value=None)
        )

        final_state = compiled_graph.invoke(state, config=config)

    # ── Core assertions ───────────────────────────────────────────────────────
    assert final_state is not None, "Pipeline must produce a final state"

    # Required state keys are present
    assert "coverage_report"   in final_state
    assert "execution_report"  in final_state
    assert "commit_status"     in final_state
    assert "defects"           in final_state
    assert "current_phase"     in final_state

    # Pipeline terminated at finalize
    assert final_state["current_phase"] == "finalize", (
        f"Expected current_phase='finalize', got {final_state['current_phase']!r}"
    )

    # commit_status is a valid value
    assert final_state["commit_status"] in ("success", "failure", "error", "pending")

    # coverage_report was produced (either from LLM or fallback)
    cov = final_state["coverage_report"]
    assert cov is not None, "coverage_report must be set by evaluate_coverage node"
    assert 0.0 <= cov.overall_score <= 10.0
    assert isinstance(cov.merge_recommendation, MergeRecommendation)

    # execution_report was produced
    rpt = final_state["execution_report"]
    assert rpt is not None
    assert rpt.total >= 0


def test_pipeline_defects_is_list(mock_pr_metadata, mock_execution_report):
    """defects in final state is always a list (never None)."""
    from qa_agent.graph import compiled_graph

    thread_id = f"integration-{uuid.uuid4().hex[:8]}"
    config    = {"configurable": {"thread_id": thread_id}}
    state     = initial_state(mock_pr_metadata)

    err = _llm_error()

    with contextlib.ExitStack() as stack:
        for target in _LLM_PATCH_TARGETS:
            stack.enter_context(patch(target, side_effect=err))
        stack.enter_context(
            patch("qa_agent.nodes.execute_tests.execute_all_scripts",
                  return_value=mock_execution_report)
        )
        stack.enter_context(
            patch("qa_agent.nodes.retrieve_history.retrieve_similar_tests",
                  return_value=[])
        )
        stack.enter_context(
            patch("qa_agent.nodes.store_tests.store_test_in_memory",
                  return_value=None)
        )

        final_state = compiled_graph.invoke(state, config=config)

    assert isinstance(final_state.get("defects"), list)


def test_pipeline_decisions_log_created(mock_pr_metadata, mock_execution_report):
    """
    decisions.jsonl is created and contains valid JSON entries after the run.
    Each entry must have 'node' and 'timestamp' keys.
    """
    from qa_agent.graph import compiled_graph

    thread_id = f"integration-{uuid.uuid4().hex[:8]}"
    config    = {"configurable": {"thread_id": thread_id}}
    state     = initial_state(mock_pr_metadata)

    err = _llm_error()

    with contextlib.ExitStack() as stack:
        for target in _LLM_PATCH_TARGETS:
            stack.enter_context(patch(target, side_effect=err))
        stack.enter_context(
            patch("qa_agent.nodes.execute_tests.execute_all_scripts",
                  return_value=mock_execution_report)
        )
        stack.enter_context(
            patch("qa_agent.nodes.retrieve_history.retrieve_similar_tests",
                  return_value=[])
        )
        stack.enter_context(
            patch("qa_agent.nodes.store_tests.store_test_in_memory",
                  return_value=None)
        )

        compiled_graph.invoke(state, config=config)

    log_dir  = os.environ.get("QA_AGENT_LOG_DIR", "./logs")
    log_file = Path(log_dir) / "decisions.jsonl"

    assert log_file.exists(), f"decisions.jsonl not found at {log_file}"

    entries = []
    with log_file.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                entry = json.loads(line)   # raises if invalid JSON
                entries.append(entry)

    assert len(entries) > 0, "decisions.jsonl must contain at least one entry"

    for entry in entries:
        assert "node"      in entry, f"Entry missing 'node': {entry}"
        assert "timestamp" in entry, f"Entry missing 'timestamp': {entry}"


def test_pipeline_no_errors_key_on_clean_run(mock_pr_metadata, mock_execution_report):
    """
    On a clean run where all nodes use fallbacks gracefully, the errors list
    (operator.add accumulator) should be a list (may be non-empty due to LLM
    fallback warnings but should not cause the pipeline to raise).
    """
    from qa_agent.graph import compiled_graph

    thread_id = f"integration-{uuid.uuid4().hex[:8]}"
    config    = {"configurable": {"thread_id": thread_id}}
    state     = initial_state(mock_pr_metadata)

    err = _llm_error()

    with contextlib.ExitStack() as stack:
        for target in _LLM_PATCH_TARGETS:
            stack.enter_context(patch(target, side_effect=err))
        stack.enter_context(
            patch("qa_agent.nodes.execute_tests.execute_all_scripts",
                  return_value=mock_execution_report)
        )
        stack.enter_context(
            patch("qa_agent.nodes.retrieve_history.retrieve_similar_tests",
                  return_value=[])
        )
        stack.enter_context(
            patch("qa_agent.nodes.store_tests.store_test_in_memory",
                  return_value=None)
        )

        # Must not raise
        final_state = compiled_graph.invoke(state, config=config)

    assert isinstance(final_state.get("errors", []), list)
