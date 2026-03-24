"""
qa_agent/tools/execution_tools.py
───────────────────────────────────
Test execution tools: subprocess-isolated pytest runner and Playwright runner.

Decision boundary
─────────────────
These tools answer only:
  • "Did this test file pass when executed?"
  • "What was the raw output?"

They never interpret why a test failed, classify failures, or make routing
decisions. TestExecutorAgent calls these functions and hands the raw
ExecutionReport to the SelfHealingAgent classifier.

Isolation model
───────────────
Every execution runs in a fresh subprocess with a clean temporary directory.
This prevents test-to-test contamination and mirrors CI behaviour.

The pytest runner uses --json-report to produce machine-readable output
so that per-test statuses can be parsed without scraping stdout.

Cross-platform notes
────────────────────
• All temp paths use tempfile.mkdtemp() — works on Windows, Linux, macOS.
• pytest is invoked via sys.executable -m pytest to guarantee the correct
  virtualenv is used (never a system-level pytest).
• The Playwright runner uses sys.executable -m pytest --browser chromium
  via the pytest-playwright plugin (same binary, different plugin markers).

Dependencies
────────────
  pytest>=8.0.0
  pytest-json-report>=1.5.0
  pytest-timeout>=2.0.0
  playwright>=1.45.0  (+ `playwright install chromium`)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel

from qa_agent.state import (
    ExecutionReport,
    FailureType,
    TestResult,
    TestScript,
)


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _classify_failure_heuristic(
    error_message: str, stdout: str, stderr: str
) -> Optional[FailureType]:
    """
    Quick rule-based failure classification from raw output.

    This duplicates part of healing/classifier.py intentionally — the
    execution tool stamps a preliminary failure_type so that the supervisor
    can route to self_heal or human_review without waiting for the full
    classifier LLM pass.

    Returns None if the type cannot be determined from heuristics alone;
    the full classifier in healing/classifier.py will handle those cases.
    """
    combined = f"{error_message}\n{stdout}\n{stderr}".lower()

    if any(x in combined for x in ["401", "403", "unauthorized", "forbidden"]):
        return FailureType.AUTH
    if any(x in combined for x in
           ["connectionerror", "connection refused", "503", "502", "name or service not known",
            "timeout", "read timed out", "timed out waiting"]):
        return FailureType.ENVIRONMENT
    if any(x in combined for x in
           ["element not found", "locator(", "timeout waiting for selector",
            "waiting for", "playwright"]):
        return FailureType.SELECTOR
    if "jsonschema" in combined or (
        "schema" in combined and "assert" in combined
    ):
        return FailureType.SCHEMA
    if any(x in combined for x in ["assertionerror", "assert ", "expected:"]):
        return FailureType.ASSERTION

    return None


def _parse_json_report(
    report_path: str,
    stdout:      str,
    stderr:      str,
    start_ts:    float,
) -> ExecutionReport:
    """
    Parse a pytest-json-report JSON file into an ExecutionReport.

    Falls back to a minimal error report if the file is missing or malformed
    (e.g. pytest itself crashed before writing the report).
    """
    if not Path(report_path).exists():
        return ExecutionReport(
            total=0,
            passed=0,
            failed=0,
            errors=1,
            skipped=0,
            duration_seconds=round(time.time() - start_ts, 2),
            per_test_results=[
                TestResult(
                    test_id="pytest_collection_error",
                    status="error",
                    duration_ms=0.0,
                    error_message="pytest did not produce a JSON report — "
                                  "check stderr for collection errors.",
                    failure_type=FailureType.ENVIRONMENT,
                    stdout=stdout[:4000],
                    stderr=stderr[:4000],
                )
            ],
        )

    with open(report_path, encoding="utf-8") as fh:
        report = json.load(fh)

    summary   = report.get("summary", {})
    tests_raw = report.get("tests",   [])

    per_test: List[TestResult] = []
    for t in tests_raw:
        outcome     = t.get("outcome", "error")    # passed|failed|error|skipped
        call_info   = t.get("call",    {}) or {}
        longrepr    = call_info.get("longrepr", "") or ""
        duration_ms = round((t.get("duration", 0.0)) * 1000, 2)
        test_stdout = (call_info.get("stdout") or "")
        test_stderr = (call_info.get("stderr") or "")

        failure_type: Optional[FailureType] = None
        if outcome in ("failed", "error"):
            failure_type = _classify_failure_heuristic(
                longrepr, test_stdout, test_stderr
            )

        per_test.append(
            TestResult(
                test_id       = t.get("nodeid", "unknown"),
                scenario_id   = None,   # back-filled by execute_tests node
                status        = outcome,
                duration_ms   = duration_ms,
                error_message = longrepr[:2000] if longrepr else None,
                failure_type  = failure_type,
                stdout        = test_stdout[:2000],
                stderr        = test_stderr[:2000],
            )
        )

    return ExecutionReport(
        total            = summary.get("total",   len(per_test)),
        passed           = summary.get("passed",  0),
        failed           = summary.get("failed",  0),
        errors           = summary.get("errors",  0),
        skipped          = summary.get("skipped", 0),
        duration_seconds = round(time.time() - start_ts, 2),
        per_test_results = per_test,
    )


def _write_test_file(content: str, work_dir: str, filename: str) -> str:
    """Write test content to a file inside work_dir. Returns absolute path."""
    path = Path(work_dir) / filename
    path.write_text(content, encoding="utf-8")
    return str(path)


def _build_env(base_url: str, auth_token: str) -> dict:
    """
    Construct the subprocess environment for test execution.

    Inherits the current process env so that installed packages are visible,
    but overrides BASE_URL and AUTH_TOKEN so tests pick them up via os.environ.
    PYTHONDONTWRITEBYTECODE prevents .pyc pollution in temp dirs.
    """
    env = os.environ.copy()
    env["BASE_URL"]                = base_url
    env["AUTH_TOKEN"]              = auth_token
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # Prevent pytest from writing .cache into the temp dir
    env["PYTEST_NO_HEADER"]        = "1"
    return env


# ──────────────────────────────────────────────────────────────────────────────
# Public tool functions
# ──────────────────────────────────────────────────────────────────────────────

def execute_pytest_suite(
    test_script:     TestScript,
    base_url:        str,
    auth_token:      str = "",
    timeout_seconds: int = 120,
    extra_args:      Optional[List[str]] = None,
) -> ExecutionReport:
    """
    Write the test script to a temp dir, execute it with pytest, and return
    a structured ExecutionReport.

    Decision boundary: executes and collects raw results. Does NOT interpret
    why any test failed or decide how to fix it — SelfHealingAgent does that.

    Isolation: each call creates and destroys its own temp directory, so
    parallel calls do not share state.

    Parameters
    ----------
    test_script     : TestScript   The script to execute (framework must be "pytest").
    base_url        : str          URL of the system under test, injected as BASE_URL.
    auth_token      : str          Auth bearer token injected as AUTH_TOKEN.
    timeout_seconds : int          Wall-clock timeout for the subprocess. Default 120s.
    extra_args      : list[str]    Additional pytest flags, e.g. ["-k", "test_create"].

    Returns
    -------
    ExecutionReport
        Structured pass/fail/error counts and per-test details.
    """
    work_dir   = tempfile.mkdtemp(prefix="qa_pytest_")
    run_id     = uuid.uuid4().hex[:8]
    test_file  = _write_test_file(
        test_script.content, work_dir, f"test_{run_id}.py"
    )
    report_file = str(Path(work_dir) / f"report_{run_id}.json")
    start_ts    = time.time()

    cmd = [
        sys.executable, "-m", "pytest",
        test_file,
        "--json-report",
        f"--json-report-file={report_file}",
        "--tb=short",
        "-v",
        f"--timeout={timeout_seconds}",
        "-p", "no:cacheprovider",
    ]
    if extra_args:
        cmd.extend(extra_args)

    env = _build_env(base_url, auth_token)

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout_seconds + 15,   # outer timeout > pytest --timeout
        )
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
    except subprocess.TimeoutExpired as exc:
        stdout = (exc.stdout or b"").decode("utf-8", errors="replace")
        stderr = f"[execution_tools] Subprocess timed out after {timeout_seconds + 15}s\n"
        stderr += (exc.stderr or b"").decode("utf-8", errors="replace")
    except Exception as exc:
        stdout = ""
        stderr = f"[execution_tools] Failed to launch subprocess: {exc}"
    finally:
        # Always attempt to parse whatever report exists before cleanup
        report = _parse_json_report(report_file, stdout, stderr, start_ts)
        shutil.rmtree(work_dir, ignore_errors=True)

    return report


def execute_playwright_suite(
    test_script:     TestScript,
    base_url:        str,
    auth_token:      str = "",
    timeout_seconds: int = 180,
    browser:         str = "chromium",
    headless:        bool = True,
) -> ExecutionReport:
    """
    Execute a Playwright (pytest-playwright) test script in a subprocess.

    Playwright tests are written as standard pytest files using the
    `playwright` fixture from pytest-playwright. This runner invokes pytest
    with the --browser flag rather than a separate Playwright CLI.

    Decision boundary: executes and collects raw results. Does NOT classify
    SELECTOR failures or decide how to repair broken locators — that is
    SelfHealingAgent's job.

    Parameters
    ----------
    test_script     : TestScript   Script with framework == "playwright".
    base_url        : str          Injected as BASE_URL env var.
    auth_token      : str          Injected as AUTH_TOKEN env var.
    timeout_seconds : int          Wall-clock subprocess timeout. Default 180s.
    browser         : str          Playwright browser: chromium | firefox | webkit.
    headless        : bool         Run browser headless. Default True (CI-safe).

    Returns
    -------
    ExecutionReport
        Same schema as execute_pytest_suite — one per-test entry per spec.
    """
    work_dir    = tempfile.mkdtemp(prefix="qa_playwright_")
    run_id      = uuid.uuid4().hex[:8]
    test_file   = _write_test_file(
        test_script.content, work_dir, f"test_pw_{run_id}.py"
    )
    report_file = str(Path(work_dir) / f"report_pw_{run_id}.json")
    start_ts    = time.time()

    env = _build_env(base_url, auth_token)
    if headless:
        env["PWHEADLESS"] = "1"

    cmd = [
        sys.executable, "-m", "pytest",
        test_file,
        f"--browser={browser}",
        "--json-report",
        f"--json-report-file={report_file}",
        "--tb=short",
        "-v",
        f"--timeout={timeout_seconds}",
        "-p", "no:cacheprovider",
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout_seconds + 20,
        )
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
    except subprocess.TimeoutExpired as exc:
        stdout = (exc.stdout or b"").decode("utf-8", errors="replace")
        stderr = (
            f"[execution_tools] Playwright subprocess timed out "
            f"after {timeout_seconds + 20}s\n"
        )
        stderr += (exc.stderr or b"").decode("utf-8", errors="replace")
    except Exception as exc:
        stdout = ""
        stderr = f"[execution_tools] Failed to launch Playwright subprocess: {exc}"
    finally:
        report = _parse_json_report(report_file, stdout, stderr, start_ts)
        shutil.rmtree(work_dir, ignore_errors=True)

    return report


def dispatch_execution(
    test_script:     TestScript,
    base_url:        str,
    auth_token:      str = "",
    timeout_seconds: int = 120,
) -> ExecutionReport:
    """
    Dispatch a test script to the correct runner based on its framework field.

    Decision boundary: routes to execute_pytest_suite or
    execute_playwright_suite. Never interprets results.

    Parameters
    ----------
    test_script : TestScript   Script with framework == "pytest" or "playwright".
    base_url    : str          Target service URL.
    auth_token  : str          Bearer token for auth.
    timeout_seconds : int      Subprocess wall-clock timeout.

    Returns
    -------
    ExecutionReport

    Raises
    ------
    ValueError   If test_script.framework is not "pytest" or "playwright".
    """
    framework = test_script.framework.lower()

    if framework == "pytest":
        return execute_pytest_suite(
            test_script=test_script,
            base_url=base_url,
            auth_token=auth_token,
            timeout_seconds=timeout_seconds,
        )
    elif framework == "playwright":
        return execute_playwright_suite(
            test_script=test_script,
            base_url=base_url,
            auth_token=auth_token,
            timeout_seconds=timeout_seconds,
        )
    else:
        raise ValueError(
            f"Unknown test framework '{framework}'. "
            f"Expected 'pytest' or 'playwright'."
        )


def execute_all_scripts(
    test_scripts:    List[TestScript],
    base_url:        str,
    auth_token:      str = "",
    timeout_seconds: int = 120,
) -> ExecutionReport:
    """
    Execute every script in sequence and merge results into one ExecutionReport.

    Sequential (not parallel) to avoid port/resource conflicts between test
    processes on the same machine. The graph's parallelism is at the node
    level (Phase 2), not within the execution phase.

    Decision boundary: aggregates ExecutionReports from dispatch_execution.
    Never decides what to do with failures — TestExecutorAgent and routing
    functions do that.

    Parameters
    ----------
    test_scripts    : List[TestScript]  All scripts to execute.
    base_url        : str               Target service URL.
    auth_token      : str               Bearer token.
    timeout_seconds : int               Per-script timeout.

    Returns
    -------
    ExecutionReport
        Merged report with combined totals and all per-test results.
    """
    if not test_scripts:
        return ExecutionReport(
            total=0, passed=0, failed=0,
            errors=0, skipped=0, duration_seconds=0.0,
        )

    overall_start = time.time()
    merged_results: List[TestResult] = []
    total = passed = failed = errors = skipped = 0

    for script in test_scripts:
        try:
            report = dispatch_execution(
                test_script=script,
                base_url=base_url,
                auth_token=auth_token,
                timeout_seconds=timeout_seconds,
            )
        except Exception as exc:
            # Script-level dispatch failure — record as a single error result
            report = ExecutionReport(
                total=1, passed=0, failed=0, errors=1, skipped=0,
                duration_seconds=0.0,
                per_test_results=[
                    TestResult(
                        test_id=script.file_path,
                        scenario_id=script.scenario_id,
                        status="error",
                        duration_ms=0.0,
                        error_message=str(exc),
                        failure_type=FailureType.ENVIRONMENT,
                        stdout="",
                        stderr=str(exc),
                    )
                ],
            )

        # Back-fill scenario_id on per-test results from this script
        for result in report.per_test_results:
            if not result.scenario_id:
                result = result.model_copy(
                    update={"scenario_id": script.scenario_id}
                )
            merged_results.append(result)

        total   += report.total
        passed  += report.passed
        failed  += report.failed
        errors  += report.errors
        skipped += report.skipped

    return ExecutionReport(
        total            = total,
        passed           = passed,
        failed           = failed,
        errors           = errors,
        skipped          = skipped,
        duration_seconds = round(time.time() - overall_start, 2),
        per_test_results = merged_results,
    )
