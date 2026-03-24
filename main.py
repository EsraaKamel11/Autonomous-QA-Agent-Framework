#!/usr/bin/env python3
"""
main.py — CLI entrypoint for the Autonomous QA Agent Framework.

Usage
─────
  # Full run against a live GitHub PR:
  python main.py --repo owner/repo --pr 123

  # With a specific OpenAPI spec file:
  python main.py --repo owner/repo --pr 123 --spec ./openapi.yaml

  # Dry-run with fully mocked data — no credentials needed:
  python main.py --mock

  # Resume a suspended pipeline after HITL review:
  python main.py --resume --thread-id <id> --action approve
  python main.py --resume --thread-id <id> --action reject
  python main.py --resume --thread-id <id> --action modify

Exit Codes
──────────
  0  APPROVE         — all tests passed, coverage sufficient
  1  BLOCK           — critical failures, merge blocked
  2  REQUEST_CHANGES — coverage gaps or minor failures
  3  ERROR           — pipeline raised an unhandled exception

Environment Variables
──────────────────────
  VOCAREUM_API_KEY    — Vocareum OpenAI proxy key  (required for LLM calls)
  VOCAREUM_BASE_URL   — Vocareum proxy base URL
  LANGCHAIN_API_KEY   — LangSmith API key (optional tracing)
  GITHUB_TOKEN        — GitHub personal access token (required for live runs)
  USE_LIVE_GITHUB     — "true" to post real PR comments (default: "false")
  USE_LIVE_JIRA       — "true" to create real Jira tickets (default: "false")
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Optional

# ── Load .env file before anything else ──────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)   # env vars already set take priority
except ImportError:
    pass  # python-dotenv is optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("qa_agent.main")

# ──────────────────────────────────────────────────────────────────────────────
# Exit codes
# ──────────────────────────────────────────────────────────────────────────────

_EXIT_APPROVE         = 0
_EXIT_BLOCK           = 1
_EXIT_REQUEST_CHANGES = 2
_EXIT_ERROR           = 3


def _verdict_to_exit(commit_status: str, coverage_verdict: Optional[str]) -> int:
    """Map pipeline outcome to a POSIX exit code."""
    if commit_status == "success" or coverage_verdict == "APPROVE":
        return _EXIT_APPROVE
    if coverage_verdict == "BLOCK":
        return _EXIT_BLOCK
    if coverage_verdict == "REQUEST_CHANGES":
        return _EXIT_REQUEST_CHANGES
    if commit_status == "failure":
        return _EXIT_BLOCK
    return _EXIT_ERROR


# ──────────────────────────────────────────────────────────────────────────────
# Mock LLM — used by --mock mode
# ──────────────────────────────────────────────────────────────────────────────

def _make_mock_llm_class() -> type:
    """
    Build and return a MockChatOpenAI class that mimics langchain_openai.ChatOpenAI.

    When with_structured_output(schema) is called, the returned chain's
    .invoke() produces a minimal valid Pydantic instance of *schema* by
    introspecting its model_fields.  Special-cases content fields to return
    syntactically valid pytest source code so execute_tests can run the files.
    """
    import enum
    import types
    import typing
    from unittest.mock import MagicMock

    from pydantic import BaseModel

    try:
        from pydantic_core import PydanticUndefined as _Undef
    except ImportError:
        _Undef = None  # type: ignore[assignment]

    def _field_is_required(finfo: Any) -> bool:
        if _Undef is not None:
            return finfo.default is _Undef and finfo.default_factory is None
        return finfo.is_required()

    def _val_for(field_name: str, ann: Any, depth: int) -> Any:
        """Return a type-appropriate default value for one Pydantic field."""
        if depth > 4 or ann is None:
            return None

        origin = getattr(ann, "__origin__", None)
        args   = getattr(ann, "__args__", None) or ()

        # ── Generic container types ───────────────────────────────────────────
        if origin is list:
            if not args:
                return []
            item_t = args[0]
            if item_t is str:
                return [f"mock_{field_name}"]
            if item_t is int:
                return [1]
            if item_t is float:
                return [5.0]
            item = _make_minimal(item_t, depth + 1)
            return [item] if item is not None else []

        if origin is typing.Union:
            # Optional[X] — prefer None (field has no default, but None is valid)
            return None

        # Python 3.10+ X | Y union syntax
        if hasattr(types, "UnionType") and isinstance(ann, types.UnionType):
            return None

        if origin is dict:
            return {}

        # ── Concrete types ────────────────────────────────────────────────────
        if not isinstance(ann, type):
            return None

        if issubclass(ann, bool):
            return False
        if issubclass(ann, int):
            return 0
        if issubclass(ann, float):
            return 5.0   # > 0 gives sensible coverage scores
        # Check Enum BEFORE str — StrEnum subclasses are both str and Enum;
        # we need the enum member (has .value), not a plain string.
        if issubclass(ann, enum.Enum):
            return list(ann)[0]   # first declared enum member
        if issubclass(ann, str):
            # Content fields need syntactically valid Python so pytest can run them
            if field_name in ("content", "patched_content"):
                return (
                    "import pytest\n\n"
                    "def test_mock_generated():\n"
                    '    pytest.skip("mock -- LLM replaced in --mock mode")\n'
                )
            _str_hints = {
                "method": "GET",
                "path": "/mock/endpoint",
                "scenario_type": "happy_path",
                "framework": "pytest",
                "scenario_id": "MOCK_mock_endpoint_happy_path_000",
            }
            return _str_hints.get(field_name, f"mock_{field_name}")
        if issubclass(ann, BaseModel):
            return _make_minimal(ann, depth + 1)

        return None

    def _make_minimal(schema: type, depth: int = 0) -> Any:
        """Create a minimal valid Pydantic BaseModel instance."""
        if depth > 4:
            return None
        if not (isinstance(schema, type) and issubclass(schema, BaseModel)):
            return None

        # Try zero-argument construction first (all fields have defaults)
        try:
            return schema()
        except Exception:
            pass

        # Build required fields only
        kwargs: dict = {}
        for fname, finfo in schema.model_fields.items():
            try:
                required = _field_is_required(finfo)
            except Exception:
                required = True
            if required:
                kwargs[fname] = _val_for(fname, finfo.annotation, depth)

        try:
            return schema(**kwargs)
        except Exception:
            return schema.model_construct(**kwargs)

    class _MockChain:
        def __init__(self, schema: type) -> None:
            self._schema = schema

        def invoke(self, messages: Any) -> Any:
            return _make_minimal(self._schema)

    class MockChatOpenAI:
        """Drop-in replacement for langchain_openai.ChatOpenAI in --mock mode."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def with_structured_output(self, schema: type) -> _MockChain:
            return _MockChain(schema)

        def invoke(self, messages: Any) -> Any:
            try:
                from langchain_core.messages import AIMessage
                return AIMessage(content="mock response")
            except ImportError:
                return MagicMock(content="mock response")

    return MockChatOpenAI


def _install_mock_mode() -> None:
    """
    Patch ChatOpenAI in every node module's namespace for --mock mode.

    Since each node does `from langchain_openai import ChatOpenAI` at module
    level, replacing the name binding in each module's __dict__ is the correct
    (and minimal) approach.  Lazy imports (like in healing/classifier.py)
    are handled by patching langchain_openai.ChatOpenAI directly.
    """
    import importlib

    mock_cls = _make_mock_llm_class()

    # Modules that import ChatOpenAI at module level
    _MODULE_TARGETS = [
        "qa_agent.nodes.analyze_pr",
        "qa_agent.nodes.parse_specs",
        "qa_agent.nodes.generate_tests",
        "qa_agent.nodes.self_heal",
        "qa_agent.nodes.evaluate_coverage",
        "qa_agent.nodes.augment_tests",
        "qa_agent.nodes.generate_defects",
    ]

    for mod_path in _MODULE_TARGETS:
        try:
            mod = importlib.import_module(mod_path)
            if hasattr(mod, "ChatOpenAI"):
                setattr(mod, "ChatOpenAI", mock_cls)
        except Exception as exc:
            logger.debug("Failed to patch %s: %s", mod_path, exc)

    # Also patch langchain_openai itself for lazy imports (classifier, etc.)
    try:
        import langchain_openai
        langchain_openai.ChatOpenAI = mock_cls  # type: ignore[attr-defined]
    except Exception as exc:
        logger.debug("Failed to patch langchain_openai.ChatOpenAI: %s", exc)

    # Mock execute_all_scripts to return a clean report — avoids writing/running
    # real pytest files during --mock runs.
    try:
        from qa_agent.state import ExecutionReport, TestResult

        mock_report = ExecutionReport(
            total=2, passed=2, failed=0, errors=0, skipped=0,
            duration_seconds=0.1,
            per_test_results=[
                TestResult(
                    test_id="mock::test_generate_api_returns_200",
                    scenario_id="MOCK_mock_endpoint_happy_path_000",
                    status="passed",
                    duration_ms=42.0,
                ),
                TestResult(
                    test_id="mock::test_error_case_handled",
                    scenario_id="MOCK_mock_endpoint_error_case_001",
                    status="passed",
                    duration_ms=38.0,
                ),
            ],
        )

        import qa_agent.nodes.execute_tests as _et
        import qa_agent.tools.execution_tools as _exec
        _exec.execute_all_scripts = lambda *a, **kw: mock_report  # type: ignore[attr-defined]
        _et.execute_all_scripts   = lambda *a, **kw: mock_report  # type: ignore[attr-defined]

    except Exception as exc:
        logger.debug("Failed to install mock execute_all_scripts: %s", exc)

    # Mock ChromaDB calls to avoid network timeouts
    try:
        import qa_agent.tools.memory_tools as _mt
        import qa_agent.nodes.retrieve_history as _rh
        import qa_agent.nodes.store_tests as _st

        _mt.retrieve_similar_tests   = lambda *a, **kw: []  # type: ignore[attr-defined]
        _rh.retrieve_similar_tests   = lambda *a, **kw: []  # type: ignore[attr-defined]
        _mt.store_test_in_memory     = lambda *a, **kw: None  # type: ignore[attr-defined]
        _st.store_test_in_memory     = lambda *a, **kw: None  # type: ignore[attr-defined]

    except Exception as exc:
        logger.debug("Failed to install mock ChromaDB calls: %s", exc)

    logger.info("Mock mode active — all LLM/ChromaDB calls replaced with local stubs")


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qa_agent",
        description="Autonomous QA Agent — runs a full LLM-powered QA pipeline on a PR.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Live run arguments
    live = parser.add_argument_group("live run")
    live.add_argument("--repo", metavar="OWNER/REPO",
                      help="GitHub repository slug, e.g. acme/api")
    live.add_argument("--pr", type=int, metavar="NUMBER",
                      help="Pull request number to evaluate")
    live.add_argument("--spec", metavar="PATH",
                      help="Path to OpenAPI spec file (auto-detected if omitted)")

    # Mock run
    parser.add_argument("--mock", action="store_true",
                        help="Run with fully mocked data — no credentials required")

    # Resume after HITL interrupt
    resume = parser.add_argument_group("resume after human review")
    resume.add_argument("--resume", action="store_true",
                        help="Resume a suspended pipeline")
    resume.add_argument("--thread-id", metavar="ID",
                        help="Thread ID printed when the pipeline was suspended")
    resume.add_argument(
        "--action",
        choices=["approve", "reject", "modify"],
        default="approve",
        help="Human decision for resume (default: approve)",
    )
    resume.add_argument("--comment", metavar="TEXT", default="",
                        help="Optional reviewer comment for resume")

    parser.add_argument("--log-level",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        default="INFO", help="Logging verbosity (default: INFO)")

    return parser


# ──────────────────────────────────────────────────────────────────────────────
# Mock PR metadata
# ──────────────────────────────────────────────────────────────────────────────

def _build_mock_pr_metadata():
    """Return a realistic PRMetadata object for --mock runs."""
    from qa_agent.state import PRMetadata
    return PRMetadata(
        pr_number=42,
        repo="acme/api",
        head_sha="abc123def456abc123def456abc123def456abc12",
        base_sha="111aaa222bbb333ccc444ddd555eee666fff777a",
        author="alice",
        title="feat: Add structured error responses to GET /users/{id}",
        description=(
            "Returns RFC 7807 Problem Details JSON instead of plain-text errors.\n"
            "Affected endpoint: GET /users/{id}\n"
            "Risk: medium — changes existing error response schema."
        ),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline runner
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(args: argparse.Namespace) -> int:
    """
    Invoke the QA Agent pipeline and return a POSIX exit code.

    Handles three modes:
    1. Live run    — --repo + --pr (real GitHub API calls)
    2. Mock run    — --mock (no external calls, all stubs)
    3. Resume run  — --resume --thread-id (continue after HITL)
    """
    # ── Configure logging level ───────────────────────────────────────────────
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # ── Enable LangSmith tracing (no-op if key absent) ───────────────────────
    from qa_agent.observability import configure_langsmith
    configure_langsmith()

    # ── Mock mode: patch LLMs BEFORE importing the graph ─────────────────────
    if args.mock:
        _install_mock_mode()

    # ── Lazy graph import (after patching so mocks are in place) ─────────────
    from qa_agent.graph import compiled_graph

    # ── Resume mode ───────────────────────────────────────────────────────────
    if args.resume:
        return _resume_pipeline(compiled_graph, args)

    # ── Build initial state ───────────────────────────────────────────────────
    pr_meta = _build_pr_metadata(args)
    from qa_agent.state import initial_state
    state = initial_state(pr_meta)

    # Register spec file in env if provided
    if args.spec:
        os.environ.setdefault("QA_AGENT_SPEC_PATH", str(Path(args.spec).resolve()))

    thread_id = str(pr_meta.pr_number)
    config    = {"configurable": {"thread_id": thread_id}}

    print(f"\n{'=' * 60}")
    print(f"  QA Agent -- PR #{pr_meta.pr_number}  [{pr_meta.repo}]")
    if args.mock:
        print("  Mode: MOCK (all LLM/API calls are local stubs)")
    print(f"  Thread ID: {thread_id}")
    print(f"{'=' * 60}\n")

    # ── Invoke pipeline ───────────────────────────────────────────────────────
    final_state = None
    try:
        final_state = compiled_graph.invoke(state, config=config)

    except KeyboardInterrupt:
        print(
            f"\n\n⚠  Pipeline interrupted.\n"
            f"   To resume: python main.py --resume --thread-id {thread_id}"
        )
        return _EXIT_ERROR

    except Exception as exc:
        logger.exception("Pipeline raised an unhandled exception: %s", exc)
        return _EXIT_ERROR

    # ── Print summary ─────────────────────────────────────────────────────────
    return _print_summary_and_exit(final_state, pr_meta.pr_number)


def _build_pr_metadata(args: argparse.Namespace):
    """Build PRMetadata from CLI args or return mock data."""
    if args.mock:
        return _build_mock_pr_metadata()

    if not args.repo or not args.pr:
        print("ERROR: --repo and --pr are required for live runs. Use --mock for testing.")
        sys.exit(_EXIT_ERROR)

    # Fetch the actual PR metadata from GitHub
    try:
        from qa_agent.tools.github_tools import get_pr_diff, get_changed_files
        from qa_agent.state import PRMetadata

        # Use env vars / mock
        head_sha = os.environ.get("GITHUB_HEAD_SHA", "HEAD")
        base_sha = os.environ.get("GITHUB_BASE_SHA", "BASE")

        return PRMetadata(
            pr_number=args.pr,
            repo=args.repo,
            head_sha=head_sha,
            base_sha=base_sha,
            author=os.environ.get("GITHUB_ACTOR", "unknown"),
            title=os.environ.get("PR_TITLE", f"PR #{args.pr}"),
            description=os.environ.get("PR_DESCRIPTION", ""),
        )
    except Exception as exc:
        logger.error("Failed to build PR metadata: %s", exc)
        sys.exit(_EXIT_ERROR)


def _resume_pipeline(graph: Any, args: argparse.Namespace) -> int:
    """Resume a suspended graph from a checkpoint."""
    if not args.thread_id:
        print("ERROR: --thread-id is required for --resume")
        return _EXIT_ERROR

    from langgraph.types import Command

    decision = {"action": args.action, "comment": args.comment}
    config   = {"configurable": {"thread_id": args.thread_id}}

    print(f"\n  Resuming thread {args.thread_id!r} with action={args.action!r}\n")

    try:
        final_state = graph.invoke(Command(resume=decision), config=config)
    except KeyboardInterrupt:
        print(f"\n⚠  Resume interrupted. Thread ID: {args.thread_id}")
        return _EXIT_ERROR
    except Exception as exc:
        logger.exception("Resume failed: %s", exc)
        return _EXIT_ERROR

    pr_num = None
    try:
        pr_num = final_state["pr_metadata"].pr_number
    except Exception:
        pass
    return _print_summary_and_exit(final_state, pr_num)


def _print_summary_and_exit(final_state: Any, pr_number: Optional[int]) -> int:
    """Print the final pipeline summary and return the appropriate exit code."""
    if final_state is None:
        print("\n❌  Pipeline produced no final state.\n")
        return _EXIT_ERROR

    try:
        coverage = final_state.get("coverage_report")
        verdict  = coverage.merge_recommendation.value if coverage else None
        score    = coverage.overall_score if coverage else None
        commit   = final_state.get("commit_status", "error")
        defects  = final_state.get("defects") or []
        errors   = final_state.get("errors") or []

        _VERDICT_ICON = {
            "APPROVE":          "[APPROVE]",
            "REQUEST_CHANGES":  "[REQUEST_CHANGES]",
            "BLOCK":            "[BLOCK]",
        }

        print(f"\n{'=' * 60}")
        print(f"  QA Agent -- Final Report")
        print(f"{'=' * 60}")
        print(f"  PR #:          {pr_number}")
        print(f"  Verdict:       {_VERDICT_ICON.get(verdict or '', verdict or 'N/A')}")
        print(f"  Coverage:      {score:.1f}/10.0" if score is not None else "  Coverage:      N/A")
        print(f"  Commit status: {commit}")
        print(f"  Defects filed: {len(defects)}")
        if errors:
            print(f"  Agent errors:  {len(errors)} (see decisions.jsonl for details)")
        print(f"{'=' * 60}\n")

        # Pipeline summary from log
        from qa_agent.observability import get_run_summary
        summary = get_run_summary(pr_number=pr_number)
        if summary["nodes_executed"] > 0:
            print(f"  Nodes executed:   {summary['nodes_executed']}")
            print(f"  Total LLM tokens: {summary['total_tokens']:,}")
            print(f"  Est. cost (USD):  ${summary['cost_usd']:.4f}")
            print(f"  Total duration:   {summary['duration_ms_sum']/1000:.1f}s")
            print()

        exit_code = _verdict_to_exit(commit, verdict)
        print(f"  Exit code: {exit_code}\n")
        return exit_code

    except Exception as exc:
        logger.exception("Failed to print summary: %s", exc)
        return _EXIT_ERROR


# ──────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser  = _build_parser()
    args    = parser.parse_args()

    # Require at least one mode flag
    if not args.mock and not args.resume and (not args.repo or not args.pr):
        parser.print_help()
        print(
            "\nQuick start:\n"
            "  python main.py --mock                          # no credentials needed\n"
            "  python main.py --repo owner/repo --pr 123     # live run\n"
        )
        sys.exit(_EXIT_ERROR)

    exit_code = run_pipeline(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
