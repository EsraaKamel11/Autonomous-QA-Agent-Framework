"""
tests/conftest.py
──────────────────
Pytest configuration executed before any test module is collected.

Setting env vars at module level (not inside fixtures) ensures they are
visible before qa_agent.graph is imported — qa_agent/graph.py reads
QA_AGENT_CHECKPOINT_DIR at import time to build the compiled_graph.
"""
import os
import tempfile

# ── Redirect all file I/O to temp directories for test isolation ─────────────
_tmpdir = tempfile.mkdtemp(prefix="qa_agent_tests_")

os.environ.setdefault("QA_AGENT_CHECKPOINT_DIR", os.path.join(_tmpdir, "checkpoints"))
os.environ.setdefault("QA_AGENT_LOG_DIR",        os.path.join(_tmpdir, "logs"))

# ── Disable live external services ───────────────────────────────────────────
os.environ.setdefault("USE_LIVE_GITHUB", "false")
os.environ.setdefault("USE_LIVE_JIRA",   "false")

# ── Use a non-real API key so LLM calls fail fast (not timeout) ──────────────
os.environ.setdefault("VOCAREUM_API_KEY",  "voc-test-placeholder")
os.environ.setdefault("VOCAREUM_BASE_URL", "http://127.0.0.1:1/nonexistent")
