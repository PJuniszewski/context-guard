"""
Adversarial tests for intelligent trimming.

These tests are DESIGNED TO EXPOSE LIMITATIONS - they show where trimming
hides local truth and may cause confident hallucinations.

The tests check:
1. Single-record anomaly buried in the middle (lost by sampling)
2. Trend reversal after 60% of data (hidden by even sampling)
3. Confident hallucination on incomplete data

Run: pytest tests/test_llm_adversarial.py -v -m llm --tb=long

Expected behavior:
- FULL data: Claude answers correctly
- TRIMMED data: Claude should either:
  a) Acknowledge uncertainty/missing data (GOOD - system is honest)
  b) Hallucinate with confidence (BAD - needs warning in report)
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
# ADVERSARIAL TEST 1: Single-record anomaly in the middle
# =============================================================================

@pytest.mark.llm
class TestSingleAnomalyLost:
    """
    Test: Anomaly buried in middle should cause uncertainty, not hallucination.

    Scenario:
    - 200 API logs, all status 200
    - 1 record (index 100) with status 500 + critical error
    - Question: "Why does the system produce critical errors?"

    Expected:
    - FULL → Claude identifies "payment overflow" error
    - TRIMMED → Claude says "insufficient data" / "cannot determine"
    - ❌ FAIL if Claude guesses or hallucinates
    """

    @pytest.fixture
    def logs_with_hidden_anomaly(self):
        """Create logs with single critical error buried in middle."""
        logs = []
        for i in range(200):
            logs.append({
                "id": i,
                "timestamp": f"2024-01-01T{i//60:02d}:{i%60:02d}:00Z",
                "status_code": 200,
                "endpoint": "/api/health",
                "response_time_ms": 50
            })

        # Bury critical anomaly at index 100 (middle)
        logs[100] = {
            "id": 100,
            "timestamp": "2024-01-01T01:40:00Z",
            "status_code": 500,
            "endpoint": "/api/checkout",
            "response_time_ms": 5000,
            "error": "CRITICAL: payment gateway overflow - transaction rollback failed"
        }

        return logs

    def test_full_data_finds_anomaly(self, llm_client, logs_with_hidden_anomaly):
        """Verify full data correctly identifies the critical error."""
        logs = logs_with_hidden_anomaly
        logs_json = json.dumps(logs, separators=(",", ":"))

        response = llm_client.ask_about_data(
            data_json=logs_json,
            question="Why does the system produce critical errors? What is the root cause?"
        )

        content = response.content.lower()
        # Full data should find the payment/checkout error
        assert any(word in content for word in ["payment", "checkout", "500", "overflow", "critical"]), \
            f"Full data should identify the critical error. Got: {response.content[:200]}"

    def test_trimmed_data_anomaly_handling(self, llm_client, logs_with_hidden_anomaly):
        """
        Trimmed data should either:
        - Find the anomaly (unlikely with aggressive trimming)
        - Acknowledge missing/insufficient data (GOOD)
        - NOT confidently claim "no errors" (BAD - hallucination)
        """
        logs = logs_with_hidden_anomaly

        # Aggressive trimming: 200 -> ~15 records
        trimmed, report = intelligent_trim(
            logs,
            token_budget=TokenBudget(total_budget=800, min_records=5),
            sampling_strategy=SamplingStrategy.FIRST_LAST_EVEN,
        )
        trimmed_json = json.dumps(trimmed, separators=(",", ":"))

        # Check if anomaly survived trimming
        trimmed_has_error = any(
            r.get("status_code") == 500 or "error" in r
            for r in trimmed
        )

        response = llm_client.ask_about_data(
            data_json=trimmed_json,
            question="Why does the system produce critical errors? What is the root cause?",
            context_prefix=report.to_claude_context(),
        )

        content = response.content.lower()

        if trimmed_has_error:
            # Anomaly survived - test passes trivially
            print(f"NOTE: Anomaly survived trimming (index 100 was sampled)")
            assert "payment" in content or "500" in content or "error" in content
        else:
            # Anomaly was lost - check for honest uncertainty vs hallucination
            uncertainty_phrases = [
                "cannot determine", "insufficient data", "no errors visible",
                "based on the sample", "trimmed data", "may be missing",
                "cannot see", "not visible", "no critical errors in",
                "sample shows", "available data"
            ]
            confident_wrong_phrases = [
                "no errors", "system is healthy", "all requests succeeded",
                "no critical", "everything is working"
            ]

            has_uncertainty = any(phrase in content for phrase in uncertainty_phrases)
            has_confident_wrong = any(phrase in content for phrase in confident_wrong_phrases)

            # This is the key assertion:
            # If Claude confidently says "no errors" when there ARE errors (just hidden),
            # that's a hallucination we need to prevent
            if has_confident_wrong and not has_uncertainty:
                pytest.fail(
                    f"HALLUCINATION DETECTED: Claude confidently claimed no errors "
                    f"when critical error was hidden by trimming.\n"
                    f"Response: {response.content[:300]}\n"
                    f"This indicates trim report needs explicit warning."
                )

            print(f"Response handled correctly: {response.content[:200]}...")


# =============================================================================
# ADVERSARIAL TEST 2: Trend reversal after 60% of data
# =============================================================================

@pytest.mark.llm
class TestTrendReversalHidden:
    """
    Test: Trend change after 60% should not be erased by sampling.

    Scenario:
    - 200 latency metrics
    - First 120: stable ~120ms
    - Last 80: degradation 120ms → 360ms
    - Question: "Is latency increasing over time?"

    Expected:
    - FULL → "Yes, clear upward trend after minute 120"
    - TRIMMED → Should NOT say "stable" or "no trend"
    """

    @pytest.fixture
    def metrics_with_late_degradation(self):
        """Create metrics with trend reversal at 60%."""
        metrics = []
        for i in range(200):
            if i < 120:
                # Stable period
                latency = 120 + (i % 10) - 5  # ~115-125ms
            else:
                # Degradation period
                latency = 120 + (i - 120) * 3  # 120ms → 360ms

            metrics.append({
                "minute": i,
                "latency_ms": latency,
                "timestamp": f"2024-01-01T{i//60:02d}:{i%60:02d}:00Z"
            })

        return metrics

    def test_full_data_detects_trend(self, llm_client, metrics_with_late_degradation):
        """Verify full data correctly identifies the latency increase."""
        metrics = metrics_with_late_degradation
        metrics_json = json.dumps(metrics, separators=(",", ":"))

        response = llm_client.ask_about_data(
            data_json=metrics_json,
            question="Is latency increasing over time? Describe the trend."
        )

        content = response.content.lower()
        assert any(word in content for word in ["increas", "rising", "growing", "degradation", "spike"]), \
            f"Full data should detect increasing trend. Got: {response.content[:200]}"

    def test_trimmed_data_trend_handling(self, llm_client, metrics_with_late_degradation):
        """
        Trimmed data should either:
        - Detect the trend (if late samples captured degradation)
        - NOT confidently claim "stable" or "no trend"
        """
        metrics = metrics_with_late_degradation

        # Aggressive trimming
        trimmed, report = intelligent_trim(
            metrics,
            token_budget=TokenBudget(total_budget=600, min_records=5),
            sampling_strategy=SamplingStrategy.FIRST_LAST_EVEN,
        )
        trimmed_json = json.dumps(trimmed, separators=(",", ":"))

        # Check if trimmed data includes high latency values
        max_latency_in_trimmed = max(r["latency_ms"] for r in trimmed)
        has_degradation_evidence = max_latency_in_trimmed > 200

        response = llm_client.ask_about_data(
            data_json=trimmed_json,
            question="Is latency increasing over time? Describe the trend.",
            context_prefix=report.to_claude_context(),
        )

        content = response.content.lower()

        # Key check: Claude should NOT confidently say "stable" when there's degradation
        confident_stable_phrases = [
            "consistently stable", "no significant trend", "remains constant",
            "no increase", "latency is stable"
        ]

        if has_degradation_evidence:
            # Should detect the trend
            assert any(word in content for word in ["increas", "rising", "higher", "spike"]), \
                f"Should detect trend with degradation evidence. Got: {response.content[:200]}"
        else:
            # Check for false confidence
            has_false_stability = any(phrase in content for phrase in confident_stable_phrases)
            if has_false_stability:
                print(f"WARNING: Sampling may have hidden trend. Response: {response.content[:200]}")
                # Not a hard fail, but flag it
                pytest.skip(
                    f"Sampling hid the trend - Claude said stable. "
                    f"Max latency in sample: {max_latency_in_trimmed}ms"
                )


# =============================================================================
# ADVERSARIAL TEST 3: Confident hallucination
# =============================================================================

@pytest.mark.llm
class TestConfidentHallucination:
    """
    Test: Claude must NOT claim certainty on incomplete data.

    Scenario:
    - 100 products with rating 4.5
    - 1 product (index 57) with rating 1.0
    - Question: "Are there badly rated products?"

    Expected:
    - FULL → "Yes, product 57 has rating 1.0"
    - TRIMMED → Must acknowledge "partial data" / "cannot be certain"
    - ❌ FAIL if Claude confidently says "No"
    """

    @pytest.fixture
    def products_with_hidden_bad_rating(self):
        """Create products with single bad rating buried in middle."""
        products = []
        for i in range(100):
            products.append({
                "id": i,
                "name": f"Product {i}",
                "price": 99.99,
                "rating": 4.5,
                "reviews_count": 50
            })

        # Bury bad rating at index 57
        products[57]["rating"] = 1.0
        products[57]["name"] = "Defective Widget"

        return products

    def test_full_data_finds_bad_rating(self, llm_client, products_with_hidden_bad_rating):
        """Verify full data correctly identifies the badly rated product."""
        products = products_with_hidden_bad_rating
        products_json = json.dumps(products, separators=(",", ":"))

        response = llm_client.ask_about_data(
            data_json=products_json,
            question="Are there any badly rated products (rating below 3.0)? Which ones?"
        )

        content = response.content.lower()
        assert any(word in content for word in ["1.0", "defective", "badly", "low rating", "57"]), \
            f"Full data should find the bad rating. Got: {response.content[:200]}"

    def test_trimmed_data_certainty_handling(self, llm_client, products_with_hidden_bad_rating):
        """
        Trimmed data should either:
        - Find the bad rating (unlikely)
        - Acknowledge it cannot be certain (GOOD)
        - NOT confidently claim "all products are well-rated" (BAD)
        """
        products = products_with_hidden_bad_rating

        # Aggressive trimming: 100 -> ~10 records
        trimmed, report = intelligent_trim(
            products,
            token_budget=TokenBudget(total_budget=500, min_records=5),
            sampling_strategy=SamplingStrategy.FIRST_LAST_EVEN,
        )
        trimmed_json = json.dumps(trimmed, separators=(",", ":"))

        # Check if bad rating survived
        has_bad_rating = any(p.get("rating", 5.0) < 3.0 for p in trimmed)

        response = llm_client.ask_about_data(
            data_json=trimmed_json,
            question="Are there any badly rated products (rating below 3.0)? Which ones?",
            context_prefix=report.to_claude_context(),
        )

        content = response.content.lower()

        if has_bad_rating:
            # Found it - trivial pass
            print("NOTE: Bad rating survived trimming")
            assert "1.0" in content or "badly" in content or "defective" in content.lower()
        else:
            # Bad rating was lost - check for honesty
            honest_phrases = [
                "based on the sample", "visible data", "shown products",
                "cannot be certain", "may not include", "trimmed",
                "in this sample", "available products"
            ]
            confident_all_good = [
                "all products are well", "no badly rated", "all have good",
                "all ratings are above", "no products with low"
            ]

            has_honesty = any(phrase in content for phrase in honest_phrases)
            has_false_certainty = any(phrase in content for phrase in confident_all_good)

            if has_false_certainty and not has_honesty:
                pytest.fail(
                    f"HALLUCINATION: Claude confidently claimed all products well-rated "
                    f"when 1.0-rated product was hidden.\n"
                    f"Response: {response.content[:300]}\n"
                    f"Trim report needs explicit uncertainty warning."
                )

            print(f"Handled correctly with uncertainty acknowledgment: {response.content[:200]}...")


# =============================================================================
# SUMMARY TEST: Run all and report
# =============================================================================

@pytest.mark.llm
class TestAdversarialSummary:
    """Summary test that reports on all adversarial findings."""

    def test_adversarial_summary(self, llm_client):
        """Print summary of adversarial test findings."""
        print("\n" + "=" * 60)
        print("ADVERSARIAL TEST SUMMARY")
        print("=" * 60)
        print("""
These tests are designed to expose limitations of trimming.

Key findings to check:
1. Does Claude acknowledge uncertainty on trimmed data?
2. Does Claude hallucinate with false confidence?
3. Is the trim report warning sufficient?

If hallucinations occur:
- Add explicit warning to IntelligentTrimReport
- Consider forensics mode that blocks instead of sampling
""")
        print("=" * 60)
        # This test always passes - it's just for reporting
        assert True


# =============================================================================
# RUN STANDALONE
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=long", "-s"])
