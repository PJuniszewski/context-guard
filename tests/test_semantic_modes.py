"""
Tests for semantic mode detection and forensic heuristic.

These tests verify:
1. TrimMode enum and marker detection
2. Forensic question heuristic patterns
3. Fail-safe blocking behavior
4. Explicit marker unlocking

Run: pytest tests/test_semantic_modes.py -v
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

# Add scripts to path
scripts_dir = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))


# =============================================================================
# UNIT TESTS: Mode Detection
# =============================================================================

class TestModeDetection:
    """Tests for detect_semantic_mode() function."""

    def test_default_is_analysis(self):
        """Default mode should be ANALYSIS."""
        from trimmer_hook import detect_semantic_mode, TrimMode

        assert detect_semantic_mode("Some prompt without marker") == TrimMode.ANALYSIS
        assert detect_semantic_mode("What categories exist?") == TrimMode.ANALYSIS

    def test_explicit_analysis_marker(self):
        """Explicit #trimmer:mode=analysis marker."""
        from trimmer_hook import detect_semantic_mode, TrimMode

        prompt = "#trimmer:mode=analysis\nWhat are the categories?"
        assert detect_semantic_mode(prompt) == TrimMode.ANALYSIS

    def test_summary_marker(self):
        """#trimmer:mode=summary marker."""
        from trimmer_hook import detect_semantic_mode, TrimMode

        prompt = "#trimmer:mode=summary\nDescribe the data structure"
        assert detect_semantic_mode(prompt) == TrimMode.SUMMARY

    def test_forensics_marker(self):
        """#trimmer:mode=forensics marker."""
        from trimmer_hook import detect_semantic_mode, TrimMode

        prompt = "#trimmer:mode=forensics\nWhy did request id=abc123 fail?"
        assert detect_semantic_mode(prompt) == TrimMode.FORENSICS


# =============================================================================
# UNIT TESTS: Forensic Question Detection
# =============================================================================

class TestForensicDetection:
    """Tests for detect_forensic_question() heuristic."""

    def test_request_id_pattern(self):
        """Detect 'request id=X' pattern."""
        from trimmer_hook import detect_forensic_question

        is_forensic, match = detect_forensic_question("Why did request id=abc123 fail?")
        assert is_forensic
        assert "request id=abc123" in match.lower()

    def test_user_id_pattern(self):
        """Detect 'user id=X' pattern."""
        from trimmer_hook import detect_forensic_question

        is_forensic, match = detect_forensic_question("Show details for user id: xyz789")
        assert is_forensic
        assert "user id" in match.lower()

    def test_why_did_fail_pattern(self):
        """Detect 'why did X fail' pattern."""
        from trimmer_hook import detect_forensic_question

        is_forensic, match = detect_forensic_question("Why did the payment fail?")
        assert is_forensic
        assert "why did" in match.lower()

    def test_what_happened_to_pattern(self):
        """Detect 'what happened to X' pattern."""
        from trimmer_hook import detect_forensic_question

        is_forensic, match = detect_forensic_question("What happened to order-12345?")
        assert is_forensic
        assert "what happened to" in match.lower()

    def test_no_forensic_for_global_questions(self):
        """Global questions should NOT trigger forensic detection."""
        from trimmer_hook import detect_forensic_question

        global_questions = [
            "What categories exist in this data?",
            "How many records are there?",
            "What is the price range?",
            "Show me the data structure",
            "List all unique values",
        ]

        for question in global_questions:
            is_forensic, _ = detect_forensic_question(question)
            assert not is_forensic, f"Should not be forensic: {question}"

    def test_id_with_long_value(self):
        """Detect ID patterns with 6+ character values."""
        from trimmer_hook import detect_forensic_question

        is_forensic, match = detect_forensic_question("Find id=abcdef123")
        assert is_forensic

        # Short IDs should not match this pattern
        is_forensic_short, _ = detect_forensic_question("Find id=abc")
        # Note: may still match other patterns like "find X with id"


# =============================================================================
# INTEGRATION TESTS: Hook Behavior
# =============================================================================

def run_hook_with_prompt(prompt: str, data: list | dict = None) -> dict:
    """
    Run the hook with a prompt and optional data.

    Returns dict with:
    - exit_code: int
    - stdout: str
    - stderr: str
    - blocked: bool
    """
    if data is not None:
        full_prompt = f"{prompt}\n\n{json.dumps(data)}"
    else:
        full_prompt = prompt

    hook_input = {"prompt": full_prompt}

    result = subprocess.run(
        [sys.executable, str(scripts_dir / "trimmer_hook.py")],
        input=json.dumps(hook_input),
        capture_output=True,
        text=True,
        env={
            **dict(__import__("os").environ),
            "TOKEN_GUARD_PROMPT_LIMIT": "500",
            "TOKEN_GUARD_MIN_CHARS_BEFORE_COUNT": "100",
        },
    )

    return {
        "exit_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "blocked": result.returncode == 2,
    }


class TestHookBehavior:
    """Integration tests for hook blocking behavior."""

    @pytest.fixture
    def large_data(self):
        """Create large dataset that exceeds token limit."""
        return [{"id": i, "name": f"Item {i}", "value": i * 100} for i in range(200)]

    def test_forensics_mode_blocks(self, large_data):
        """Forensics mode should BLOCK, not sample."""
        prompt = "#trimmer:mode=forensics\nWhy did request id=abc123 fail?"
        result = run_hook_with_prompt(prompt, large_data)

        assert result["blocked"], "Forensics mode should block"
        assert "FORENSICS MODE" in result["stderr"]
        assert "sampling not allowed" in result["stderr"].lower()

    def test_forensic_question_without_mode_blocks(self, large_data):
        """Forensic question without explicit mode should BLOCK (fail-safe)."""
        prompt = "Why did request id=abc123 fail?"
        result = run_hook_with_prompt(prompt, large_data)

        assert result["blocked"], "Forensic question should block by default"
        assert "FORENSIC QUESTION DETECTED" in result["stderr"]
        assert "Sampling may hide" in result["stderr"]

    def test_forensic_question_with_explicit_analysis_allows(self, large_data):
        """Forensic question WITH explicit #trimmer:mode=analysis should allow sampling."""
        prompt = "#trimmer:mode=analysis\nWhy did request id=abc123 fail?"
        result = run_hook_with_prompt(prompt, large_data)

        # Should NOT block - explicit mode unlocks
        assert not result["blocked"], "Explicit analysis mode should allow"

    def test_forensic_question_with_force_allows(self, large_data):
        """Forensic question WITH #trimmer:force should allow."""
        prompt = "#trimmer:force\nWhy did request id=abc123 fail?"
        result = run_hook_with_prompt(prompt, large_data)

        # Should NOT block - force bypasses
        assert not result["blocked"], "Force marker should bypass"

    def test_global_question_samples_normally(self, large_data):
        """Global questions should sample normally (no blocking)."""
        prompt = "What categories exist in this data?"
        result = run_hook_with_prompt(prompt, large_data)

        # Should NOT block - global question, sampling OK
        assert not result["blocked"], "Global question should allow sampling"

    def test_summary_mode_allows_aggressive_trimming(self, large_data):
        """Summary mode should allow trimming."""
        prompt = "#trimmer:mode=summary\nDescribe the structure"
        result = run_hook_with_prompt(prompt, large_data)

        assert not result["blocked"], "Summary mode should allow trimming"


# =============================================================================
# UNIT TESTS: Behavior Matrix
# =============================================================================

class TestBehaviorMatrix:
    """Test the complete behavior matrix from the plan."""

    @pytest.fixture
    def large_data(self):
        return [{"id": i} for i in range(200)]

    def test_matrix_global_default_samples(self, large_data):
        """Global question + default mode = SAMPLE"""
        prompt = "What categories exist?"
        result = run_hook_with_prompt(prompt, large_data)
        assert not result["blocked"]

    def test_matrix_forensic_default_blocks(self, large_data):
        """Forensic question + default mode = BLOCK"""
        prompt = "Why did request id=req123 fail?"
        result = run_hook_with_prompt(prompt, large_data)
        assert result["blocked"]
        assert "FORENSIC QUESTION DETECTED" in result["stderr"]

    def test_matrix_forensic_forensics_blocks(self, large_data):
        """Forensic question + forensics mode = BLOCK (too large)"""
        prompt = "#trimmer:mode=forensics\nWhy did request id=req123 fail?"
        result = run_hook_with_prompt(prompt, large_data)
        assert result["blocked"]
        assert "FORENSICS MODE" in result["stderr"]

    def test_matrix_forensic_analysis_samples(self, large_data):
        """Forensic question + explicit analysis mode = SAMPLE"""
        prompt = "#trimmer:mode=analysis\nWhy did request id=req123 fail?"
        result = run_hook_with_prompt(prompt, large_data)
        assert not result["blocked"]


# =============================================================================
# RUN STANDALONE
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
