"""
LLM Client for testing context preservation.

Wrapper around Anthropic API for running test queries.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import anthropic


@dataclass
class LLMResponse:
    """Response from LLM with metadata."""
    content: str
    input_tokens: int
    output_tokens: int
    model: str

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class TestLLMClient:
    """Client for testing LLM interactions with trimmed data."""

    DEFAULT_MODEL = "claude-sonnet-4-20250514"

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the test client.

        Args:
            model: Model to use (defaults to Claude Sonnet)
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
        """
        self.model = model or self.DEFAULT_MODEL
        self.client = anthropic.Anthropic(api_key=api_key)
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._request_count = 0

    def ask(
        self,
        user_prompt: str,
        system_prompt: str = "You are a data analyst. Answer questions about the provided data concisely.",
        max_tokens: int = 500,
    ) -> LLMResponse:
        """
        Ask a question and get a response.

        Args:
            user_prompt: The user's question with data context
            system_prompt: System instructions for the model
            max_tokens: Maximum tokens in response

        Returns:
            LLMResponse with content and token usage
        """
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        # Track usage
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        self._total_input_tokens += input_tokens
        self._total_output_tokens += output_tokens
        self._request_count += 1

        return LLMResponse(
            content=response.content[0].text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self.model,
        )

    def ask_about_data(
        self,
        data_json: str,
        question: str,
        context_prefix: str = "",
    ) -> LLMResponse:
        """
        Ask a question about JSON data.

        Args:
            data_json: JSON string of the data
            question: Question to ask about the data
            context_prefix: Optional prefix (e.g., trimming report)

        Returns:
            LLMResponse with answer
        """
        parts = []
        if context_prefix:
            parts.append(context_prefix)
        parts.append(f"DATA:\n```json\n{data_json}\n```")
        parts.append(f"\nQUESTION: {question}")

        user_prompt = "\n\n".join(parts)
        return self.ask(user_prompt)

    @property
    def usage_stats(self) -> dict:
        """Get cumulative usage statistics."""
        return {
            "requests": self._request_count,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "total_tokens": self._total_input_tokens + self._total_output_tokens,
            "estimated_cost_usd": self._estimate_cost(),
        }

    def _estimate_cost(self) -> float:
        """Estimate cost based on Claude Sonnet pricing."""
        # Sonnet pricing (as of 2024): $3/M input, $15/M output
        input_cost = (self._total_input_tokens / 1_000_000) * 3.0
        output_cost = (self._total_output_tokens / 1_000_000) * 15.0
        return round(input_cost + output_cost, 4)

    def reset_stats(self) -> None:
        """Reset usage statistics."""
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._request_count = 0


class MockLLMClient:
    """Mock client for testing without API calls."""

    def __init__(self, responses: Optional[dict[str, str]] = None):
        """
        Initialize mock client.

        Args:
            responses: Dict mapping question keywords to responses
        """
        self.responses = responses or {}
        self._default_responses = {
            "gatunki": "W danych występują: setosa, versicolor, virginica",
            "species": "The species are: setosa, versicolor, virginica",
            "ile rekordów": "Oryginalnie było 150 rekordów",
            "how many records": "There were originally 150 records",
            "kategorie": "Kategorie: electronics, clothing, home",
            "categories": "Categories: electronics, clothing, home",
            "błędów": "Znaleziono 15 błędów 5xx",
            "errors": "Found 15 5xx errors",
        }
        self._request_count = 0

    def ask(self, user_prompt: str, **kwargs) -> LLMResponse:
        """Return mock response based on keywords."""
        self._request_count += 1

        prompt_lower = user_prompt.lower()

        # Check custom responses first
        for keyword, response in self.responses.items():
            if keyword.lower() in prompt_lower:
                return LLMResponse(
                    content=response,
                    input_tokens=100,
                    output_tokens=50,
                    model="mock",
                )

        # Check default responses
        for keyword, response in self._default_responses.items():
            if keyword in prompt_lower:
                return LLMResponse(
                    content=response,
                    input_tokens=100,
                    output_tokens=50,
                    model="mock",
                )

        return LLMResponse(
            content="Mock response: Unable to determine specific answer.",
            input_tokens=100,
            output_tokens=50,
            model="mock",
        )

    def ask_about_data(
        self,
        data_json: str,
        question: str,
        context_prefix: str = "",
    ) -> LLMResponse:
        """Mock version of ask_about_data."""
        full_prompt = f"{context_prefix}\n{data_json}\n{question}"
        return self.ask(full_prompt)

    @property
    def usage_stats(self) -> dict:
        return {
            "requests": self._request_count,
            "total_input_tokens": self._request_count * 100,
            "total_output_tokens": self._request_count * 50,
            "total_tokens": self._request_count * 150,
            "estimated_cost_usd": 0.0,
        }

    def reset_stats(self) -> None:
        self._request_count = 0
