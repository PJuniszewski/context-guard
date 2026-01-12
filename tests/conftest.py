"""
Pytest configuration and fixtures for LLM context tests.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

# Add scripts to path for imports
scripts_dir = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))

from llm_client import TestLLMClient, MockLLMClient


# =============================================================================
# Pytest Configuration
# =============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "llm: marks tests that require real LLM API calls (deselect with '-m \"not llm\"')"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


def pytest_collection_modifyitems(config, items):
    """Auto-skip LLM tests if no API key is available."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        skip_llm = pytest.mark.skip(reason="ANTHROPIC_API_KEY not set")
        for item in items:
            if "llm" in item.keywords:
                item.add_marker(skip_llm)


# =============================================================================
# LLM Client Fixtures
# =============================================================================

@pytest.fixture
def llm_client() -> TestLLMClient:
    """Real LLM client for integration tests."""
    return TestLLMClient()


@pytest.fixture
def mock_llm_client() -> MockLLMClient:
    """Mock LLM client for unit tests."""
    return MockLLMClient()


@pytest.fixture
def llm_or_mock() -> TestLLMClient | MockLLMClient:
    """Returns real client if API key available, otherwise mock."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return TestLLMClient()
    return MockLLMClient()


# =============================================================================
# Data Fixtures
# =============================================================================

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def iris_data() -> list[dict[str, Any]]:
    """Load Iris dataset fixture."""
    with open(FIXTURES_DIR / "iris.json") as f:
        return json.load(f)


@pytest.fixture
def products_data() -> list[dict[str, Any]]:
    """Load products dataset fixture."""
    with open(FIXTURES_DIR / "products.json") as f:
        return json.load(f)


@pytest.fixture
def api_logs_data() -> list[dict[str, Any]]:
    """Load API logs dataset fixture."""
    with open(FIXTURES_DIR / "api_logs.json") as f:
        return json.load(f)


# =============================================================================
# Trimming Fixtures
# =============================================================================

@pytest.fixture
def token_budget_small():
    """Small token budget for aggressive trimming."""
    from trim_lib import TokenBudget
    return TokenBudget(total_budget=500, min_records=2)


@pytest.fixture
def token_budget_medium():
    """Medium token budget."""
    from trim_lib import TokenBudget
    return TokenBudget(total_budget=1500, min_records=5)


@pytest.fixture
def token_budget_large():
    """Large token budget for minimal trimming."""
    from trim_lib import TokenBudget
    return TokenBudget(total_budget=3500, min_records=10)


# =============================================================================
# Helper Fixtures
# =============================================================================

@pytest.fixture
def trim_and_report():
    """Factory fixture for trimming data and getting report."""
    from trim_lib import intelligent_trim, TokenBudget, SamplingStrategy

    def _trim(
        data: list | dict,
        budget: int = 500,
        strategy: SamplingStrategy = SamplingStrategy.FIRST_LAST_EVEN,
    ):
        token_budget = TokenBudget(total_budget=budget)
        return intelligent_trim(
            data=data,
            token_budget=token_budget,
            sampling_strategy=strategy,
        )

    return _trim


@pytest.fixture
def compare_answers():
    """Factory fixture for comparing LLM answers."""

    def _compare(
        full_answer: str,
        trimmed_answer: str,
        expected_keywords: list[str],
    ) -> dict[str, Any]:
        """
        Compare answers and return analysis.

        Returns dict with:
        - full_has_keywords: bool
        - trimmed_has_keywords: bool
        - keywords_preserved: list of preserved keywords
        - keywords_lost: list of lost keywords
        - context_score: float 0.0-1.0
        """
        full_lower = full_answer.lower()
        trimmed_lower = trimmed_answer.lower()

        full_found = [kw for kw in expected_keywords if kw.lower() in full_lower]
        trimmed_found = [kw for kw in expected_keywords if kw.lower() in trimmed_lower]

        preserved = set(trimmed_found) & set(full_found)
        lost = set(full_found) - set(trimmed_found)

        score = len(preserved) / len(expected_keywords) if expected_keywords else 1.0

        return {
            "full_has_keywords": len(full_found) == len(expected_keywords),
            "trimmed_has_keywords": len(trimmed_found) == len(expected_keywords),
            "keywords_preserved": list(preserved),
            "keywords_lost": list(lost),
            "context_score": score,
        }

    return _compare
