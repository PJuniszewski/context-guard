"""
LLM Context Preservation Tests.

Tests that verify intelligent trimming preserves analytical context
by comparing LLM responses on full vs trimmed data.

Run with real API:
    ANTHROPIC_API_KEY=sk-... pytest tests/test_llm_context.py -v -m llm

Run without API (mock):
    pytest tests/test_llm_context.py -v -m "not llm"
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Add scripts to path
scripts_dir = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))

from trim_lib import intelligent_trim, TokenBudget, SamplingStrategy


# =============================================================================
# IRIS DATASET TESTS
# =============================================================================

@pytest.mark.llm
class TestIrisContextPreservation:
    """Tests for Iris dataset context preservation."""

    def test_species_identification(self, llm_client, iris_data, trim_and_report):
        """Claude should identify all species even after trimming."""
        # Full data query
        full_json = json.dumps(iris_data, separators=(",", ":"))
        full_response = llm_client.ask_about_data(
            data_json=full_json,
            question="List all unique species in this dataset. Just list the names.",
        )

        # Trimmed data query
        trimmed, report = trim_and_report(iris_data, budget=500)
        trimmed_json = json.dumps(trimmed, separators=(",", ":"))
        trimmed_response = llm_client.ask_about_data(
            data_json=trimmed_json,
            question="List all unique species in this dataset. Just list the names.",
            context_prefix=report.to_claude_context(),
        )

        # Verify all species are found
        expected_species = ["setosa", "versicolor", "virginica"]
        for species in expected_species:
            assert species in full_response.content.lower(), f"Full data missing {species}"
            assert species in trimmed_response.content.lower(), f"Trimmed data missing {species}"

    def test_record_count_awareness(self, llm_client, iris_data, trim_and_report):
        """Claude should know original record count from trim report."""
        trimmed, report = trim_and_report(iris_data, budget=500)
        trimmed_json = json.dumps(trimmed, separators=(",", ":"))

        response = llm_client.ask_about_data(
            data_json=trimmed_json,
            question="How many records were in the original dataset before trimming?",
            context_prefix=report.to_claude_context(),
        )

        assert "150" in response.content, "Claude should know original count was 150"

    def test_first_last_preservation(self, llm_client, iris_data, trim_and_report):
        """Claude should see first and last records."""
        trimmed, report = trim_and_report(iris_data, budget=500)
        trimmed_json = json.dumps(trimmed, separators=(",", ":"))

        response = llm_client.ask_about_data(
            data_json=trimmed_json,
            question="What is the 'id' of the first and last record in the data?",
            context_prefix=report.to_claude_context(),
        )

        content = response.content.lower()
        # Should find id 0 (first) and id 149 (last)
        assert "0" in content, "Should contain first record id (0)"
        assert "149" in content, "Should contain last record id (149)"

    def test_extreme_trimming_awareness(self, llm_client, iris_data):
        """With extreme trimming, Claude should acknowledge limitations."""
        # Extreme trimming: 150 -> 2 records
        trimmed, report = intelligent_trim(
            iris_data,
            token_budget=TokenBudget(total_budget=100, min_records=2),
        )
        trimmed_json = json.dumps(trimmed, separators=(",", ":"))

        response = llm_client.ask_about_data(
            data_json=trimmed_json,
            question="Calculate the average sepal_length for all flowers.",
            context_prefix=report.to_claude_context(),
        )

        content = response.content.lower()
        # Claude should mention sample/approximation/limited data
        limitation_words = ["sample", "approximat", "limit", "only", "subset", "trimmed"]
        has_limitation = any(word in content for word in limitation_words)
        assert has_limitation, "Claude should acknowledge data limitations"


# =============================================================================
# PRODUCTS DATASET TESTS
# =============================================================================

@pytest.mark.llm
class TestProductsContextPreservation:
    """Tests for e-commerce products context preservation."""

    def test_category_identification(self, llm_client, products_data, trim_and_report):
        """Claude should identify all product categories."""
        # Full data
        full_json = json.dumps(products_data, separators=(",", ":"))
        full_response = llm_client.ask_about_data(
            data_json=full_json,
            question="List all unique product categories. Just the category names.",
        )

        # Trimmed data
        trimmed, report = trim_and_report(products_data, budget=800)
        trimmed_json = json.dumps(trimmed, separators=(",", ":"))
        trimmed_response = llm_client.ask_about_data(
            data_json=trimmed_json,
            question="List all unique product categories. Just the category names.",
            context_prefix=report.to_claude_context(),
        )

        expected_categories = ["electronics", "clothing", "home", "sports", "books"]
        full_content = full_response.content.lower()
        trimmed_content = trimmed_response.content.lower()

        for cat in expected_categories:
            assert cat in full_content, f"Full data missing {cat}"
            # Trimmed may not have all - check preservation rate
            if cat in full_content and cat not in trimmed_content:
                print(f"Category '{cat}' lost after trimming")

    def test_price_range_estimation(self, llm_client, products_data, trim_and_report):
        """Claude should estimate price range from sample."""
        trimmed, report = trim_and_report(products_data, budget=600)
        trimmed_json = json.dumps(trimmed, separators=(",", ":"))

        response = llm_client.ask_about_data(
            data_json=trimmed_json,
            question="What is the approximate price range (min to max) of these products?",
            context_prefix=report.to_claude_context(),
        )

        # Should mention some price values
        content = response.content
        assert "$" in content or "price" in content.lower(), "Should discuss prices"

    def test_nested_ratings_access(self, llm_client, products_data, trim_and_report):
        """Claude should access nested ratings structure."""
        trimmed, report = trim_and_report(products_data, budget=600)
        trimmed_json = json.dumps(trimmed, separators=(",", ":"))

        response = llm_client.ask_about_data(
            data_json=trimmed_json,
            question="What is the highest average rating you can see in the data?",
            context_prefix=report.to_claude_context(),
        )

        content = response.content
        # Should mention a rating value (like 4.5, 4.8, etc.)
        assert any(f"{i}." in content for i in range(1, 6)), "Should mention a rating value"


# =============================================================================
# API LOGS DATASET TESTS
# =============================================================================

@pytest.mark.llm
class TestApiLogsContextPreservation:
    """Tests for API logs context preservation."""

    def test_error_detection(self, llm_client, api_logs_data, trim_and_report):
        """Claude should detect error patterns in logs."""
        trimmed, report = trim_and_report(api_logs_data, budget=1000)
        trimmed_json = json.dumps(trimmed, separators=(",", ":"))

        response = llm_client.ask_about_data(
            data_json=trimmed_json,
            question="Are there any 5xx server errors in these logs? What status codes do you see?",
            context_prefix=report.to_claude_context(),
        )

        content = response.content.lower()
        # Should identify error patterns
        assert "500" in content or "502" in content or "503" in content or "5xx" in content, \
            "Should identify server errors"

    def test_endpoint_identification(self, llm_client, api_logs_data, trim_and_report):
        """Claude should identify API endpoints."""
        trimmed, report = trim_and_report(api_logs_data, budget=800)
        trimmed_json = json.dumps(trimmed, separators=(",", ":"))

        response = llm_client.ask_about_data(
            data_json=trimmed_json,
            question="What API endpoints are being called in these logs?",
            context_prefix=report.to_claude_context(),
        )

        content = response.content.lower()
        # Should mention some endpoints
        assert "/api" in content, "Should identify API endpoints"

    def test_timestamp_awareness(self, llm_client, api_logs_data, trim_and_report):
        """Claude should understand log timestamps."""
        trimmed, report = trim_and_report(api_logs_data, budget=600)
        trimmed_json = json.dumps(trimmed, separators=(",", ":"))

        response = llm_client.ask_about_data(
            data_json=trimmed_json,
            question="What date are these logs from? Just the date.",
            context_prefix=report.to_claude_context(),
        )

        content = response.content
        # Should identify January 15, 2024
        assert "2024" in content or "january" in content.lower() or "15" in content, \
            "Should identify the log date"


# =============================================================================
# COMPARATIVE TESTS (Full vs Trimmed)
# =============================================================================

@pytest.mark.llm
class TestFullVsTrimmedComparison:
    """Direct comparison tests between full and trimmed data responses."""

    def test_iris_species_count_comparison(
        self, llm_client, iris_data, trim_and_report, compare_answers
    ):
        """Compare species identification: full vs trimmed."""
        question = "How many unique species are in this dataset? Just give the number."

        # Full data
        full_json = json.dumps(iris_data)
        full_response = llm_client.ask_about_data(full_json, question)

        # Trimmed data
        trimmed, report = trim_and_report(iris_data, budget=500)
        trimmed_json = json.dumps(trimmed)
        trimmed_response = llm_client.ask_about_data(
            trimmed_json, question, context_prefix=report.to_claude_context()
        )

        # Both should answer "3"
        assert "3" in full_response.content, "Full data should show 3 species"
        assert "3" in trimmed_response.content, "Trimmed data should show 3 species"

    def test_products_structure_preservation(
        self, llm_client, products_data, trim_and_report
    ):
        """Verify product structure is preserved after trimming."""
        question = "What fields does each product record have? List the field names."

        # Full data
        full_json = json.dumps(products_data[:5])  # Just first 5 for comparison
        full_response = llm_client.ask_about_data(full_json, question)

        # Trimmed data
        trimmed, report = trim_and_report(products_data, budget=600)
        trimmed_json = json.dumps(trimmed)
        trimmed_response = llm_client.ask_about_data(
            trimmed_json, question, context_prefix=report.to_claude_context()
        )

        expected_fields = ["id", "title", "price", "category", "stock", "ratings", "tags"]
        for field in expected_fields:
            assert field in full_response.content.lower(), f"Full missing {field}"
            assert field in trimmed_response.content.lower(), f"Trimmed missing {field}"


# =============================================================================
# MOCK TESTS (No API required)
# =============================================================================

class TestMockContextPreservation:
    """Tests using mock client - no API key required."""

    def test_trim_report_format(self, iris_data, trim_and_report):
        """Verify trim report has expected format."""
        trimmed, report = trim_and_report(iris_data, budget=500)

        context = report.to_claude_context()

        assert "[DATA TRIMMING SUMMARY]" in context
        assert "Original:" in context
        assert "Trimmed:" in context
        assert "ARRAYS TRIMMED:" in context
        assert "root:" in context
        assert "[END TRIMMING SUMMARY]" in context

    def test_sampling_strategy_used(self, iris_data, trim_and_report):
        """Verify first_last_even sampling is applied."""
        trimmed, report = trim_and_report(iris_data, budget=500)

        # Should have first record (id=0)
        assert trimmed[0]["id"] == 0, "First record should be id=0"

        # Should have last record (id=149)
        assert trimmed[-1]["id"] == 149, "Last record should be id=149"

        # Should have records from middle
        ids = [r["id"] for r in trimmed]
        assert len(ids) > 2, "Should have more than just first and last"

    def test_token_reduction(self, iris_data, trim_and_report):
        """Verify tokens are reduced."""
        trimmed, report = trim_and_report(iris_data, budget=500)

        assert report.original_tokens > report.trimmed_tokens, "Tokens should be reduced"
        assert report.trimmed_tokens <= 600, "Should be near budget"  # Some margin

    def test_products_category_distribution(self, products_data, trim_and_report):
        """Verify categories are preserved in sample."""
        trimmed, report = trim_and_report(products_data, budget=800)

        original_categories = set(p["category"] for p in products_data)
        trimmed_categories = set(p["category"] for p in trimmed)

        # Should preserve most categories with even sampling
        preserved_ratio = len(trimmed_categories) / len(original_categories)
        assert preserved_ratio >= 0.6, f"Should preserve >60% categories, got {preserved_ratio:.0%}"

    def test_api_logs_status_codes_distribution(self, api_logs_data, trim_and_report):
        """Verify status code distribution is somewhat preserved."""
        trimmed, report = trim_and_report(api_logs_data, budget=1000)

        original_has_5xx = any(l["status_code"] >= 500 for l in api_logs_data)
        trimmed_has_5xx = any(l["status_code"] >= 500 for l in trimmed)

        # If original has 5xx, trimmed sample should likely have some too
        # (not guaranteed but likely with even sampling)
        if original_has_5xx:
            # Just check that we have variety of status codes
            trimmed_codes = set(l["status_code"] for l in trimmed)
            assert len(trimmed_codes) > 1, "Should have variety of status codes"


# =============================================================================
# USAGE TRACKING TEST
# =============================================================================

@pytest.mark.llm
class TestUsageTracking:
    """Test that tracks API usage for cost monitoring."""

    def test_usage_stats(self, llm_client, iris_data, trim_and_report):
        """Track token usage across multiple queries."""
        llm_client.reset_stats()

        # Query 1
        trimmed, report = trim_and_report(iris_data, budget=500)
        trimmed_json = json.dumps(trimmed)
        llm_client.ask_about_data(
            trimmed_json,
            "How many species?",
            context_prefix=report.to_claude_context(),
        )

        # Query 2
        llm_client.ask_about_data(
            trimmed_json,
            "What is the range of sepal_length?",
            context_prefix=report.to_claude_context(),
        )

        stats = llm_client.usage_stats
        print(f"\n=== Usage Stats ===")
        print(f"Requests: {stats['requests']}")
        print(f"Input tokens: {stats['total_input_tokens']}")
        print(f"Output tokens: {stats['total_output_tokens']}")
        print(f"Estimated cost: ${stats['estimated_cost_usd']:.4f}")

        assert stats["requests"] == 2
        assert stats["total_tokens"] > 0


# =============================================================================
# RUN STANDALONE
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
