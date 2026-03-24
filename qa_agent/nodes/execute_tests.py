"""
qa_agent/nodes/execute_tests.py
──────────────────────────────────
TestExecutorAgent — node 6 of 12.

Decision boundary
─────────────────
Only answers: "Did these tests pass?"
Runs tests, classifies failures heuristically, returns raw results.
Never interprets root causes, suggests fixes, or makes routing decisions.

Inputs consumed from state
──────────────────────────
  state["generated_tests"]   List[TestScript]

State keys written
──────────────────
  test_results      List[TestResult]    — accumulated via operator.add
  execution_report  ExecutionReport     — latest cycle only (overwritten)
  current_phase     str
"""

from __future__ import annotations

import os
import time
from typing import List, Optional

from qa_agent.healing.classifier import classify_failure
from qa_agent.observability import log_node_execution
from qa_agent.state import (
    AgentError,
    ExecutionReport,
    FailureType,
    QAAgentState,
    TestResult,
    TestScript,
)
from qa_agent.tools.execution_tools import execute_all_scripts

_NODE_NAME = "execute_tests"

_BASE_URL    = os.environ.get("BASE_URL",    "http://localhost:8080")
_AUTH_TOKEN  = os.environ.get("AUTH_TOKEN",  "")
_TIMEOUT     = int(os.environ.get("TEST_TIMEOUT_SECONDS", "120"))


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _apply_classifier(results: List[TestResult]) -> List[TestResult]:
    """
    Apply the two-stage failure classifier to every failed/errored result.

    The heuristic classifier in execution_tools already stamps a preliminary
    failure_type.  Here we run the full classifier for any result where:
      - failure_type is None (heuristic couldn't decide), OR
      - confidence was low (the heuristic stamped a type but wasn't sure)

    This is a best-effort enhancement; results are never blocked by it.
    """
    enriched: List[TestResult] = []
    for result in results:
        if result.status not in ("failed", "error"):
            enriched.append(result)
            continue

        # If heuristic already classified confidently, keep it
        if result.failure_type is not None:
            enriched.append(result)
            continue

        # Run full classifier (rule-based first, LLM fallback if needed)
        try:
            classification = classify_failure(
                error_message=result.error_message or "",
                stderr=result.stderr or "",
                stdout=result.stdout or "",
            )
            enriched.append(result.model_copy(
                update={"failure_type": classification.failure_type}
            ))
        except Exception:
            enriched.append(result)   # keep as-is on classifier error

    return enriched


def _empty_report() -> ExecutionReport:
    return ExecutionReport(
        total=0, passed=0, failed=0,
        errors=0, skipped=0, duration_seconds=0.0,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Node function
# ──────────────────────────────────────────────────────────────────────────────

def execute_tests(state: QAAgentState) -> dict:
    """
    TestExecutorAgent: run all generated test scripts in isolated subprocesses
    and return structured results.

    Pipeline position: Node 6 (also re-entered after self_heal patches tests).
    Input:  state["generated_tests"]
    Output: state["test_results"] (accumulated), state["execution_report"] (latest)

    LLM calls: NONE.
    Tool calls: execution_tools.execute_all_scripts (subprocess per script).
    Failure mode: AgentError written, empty ExecutionReport returned.

    NOTE on operator.add:
      test_results uses operator.add reducer — results from all execution cycles
      accumulate for audit purposes.  routing.py and self_heal read from
      execution_report (latest cycle) rather than the full history.
    """
    start_ts  = time.time()
    scripts   = state.get("generated_tests") or []
    errors: list = []

    if not scripts:
        log_node_execution(
            node_name=_NODE_NAME,
            input_summary={"generated_tests": 0},
            output_summary={"note": "no scripts to execute"},
            routing_hint="evaluate_coverage",
            pr_number=state["pr_metadata"].pr_number,
        )
        empty = _empty_report()
        return {
            "test_results":    [],
            "execution_report": empty,
            "current_phase":   _NODE_NAME,
        }

    # ── Execute all scripts ───────────────────────────────────────────────────
    error_str: Optional[str] = None
    report: ExecutionReport

    try:
        report = execute_all_scripts(
            test_scripts=scripts,
            base_url=_BASE_URL,
            auth_token=_AUTH_TOKEN,
            timeout_seconds=_TIMEOUT,
        )
    except Exception as exc:
        error_str = f"{type(exc).__name__}: {exc}"
        errors.append(AgentError(
            agent="TestExecutorAgent",
            phase="execute_tests",
            error=error_str,
            recoverable=False,
        ))
        report = ExecutionReport(
            total=len(scripts),
            passed=0,
            failed=0,
            errors=len(scripts),
            skipped=0,
            duration_seconds=round(time.time() - start_ts, 2),
            per_test_results=[
                TestResult(
                    test_id=s.file_path,
                    scenario_id=s.scenario_id,
                    status="error",
                    duration_ms=0.0,
                    error_message=error_str,
                    failure_type=FailureType.ENVIRONMENT,
                    stdout="",
                    stderr=error_str,
                )
                for s in scripts
            ],
        )

    # ── Enhance failure classification ────────────────────────────────────────
    report = report.model_copy(
        update={"per_test_results": _apply_classifier(report.per_test_results)}
    )

    # ── Determine routing hint for log ────────────────────────────────────────
    failed_count = report.failed + report.errors
    retry        = state.get("retry_count", 0)
    max_retries  = state.get("max_retries", 3)

    if failed_count > 0 and retry < max_retries:
        routing_hint = "self_heal"
    elif failed_count > 0:
        routing_hint = "human_review"
    else:
        routing_hint = "evaluate_coverage"

    # ── Log before returning ──────────────────────────────────────────────────
    duration_ms = (time.time() - start_ts) * 1000
    failure_types = {}
    for r in report.per_test_results:
        if r.failure_type:
            k = r.failure_type.value
            failure_types[k] = failure_types.get(k, 0) + 1

    log_node_execution(
        node_name=_NODE_NAME,
        input_summary={
            "scripts_to_run": len(scripts),
            "retry_count":    retry,
        },
        output_summary={
            "total":          report.total,
            "passed":         report.passed,
            "failed":         report.failed,
            "errors":         report.errors,
            "duration_s":     report.duration_seconds,
            "failure_types":  str(failure_types),
        },
        routing_hint=routing_hint,
        pr_number=state["pr_metadata"].pr_number,
        duration_ms=duration_ms,
        error=error_str,
    )

    result: dict = {
        "test_results":     report.per_test_results,   # operator.add — appended
        "execution_report": report,                    # overwritten each cycle
        "current_phase":    _NODE_NAME,
    }
    if errors:
        result["errors"] = errors
    return result
