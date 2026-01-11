#!/usr/bin/env python3
"""Trimming library for JSON payloads."""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Any, Optional

# Default configuration
DEFAULT_MAX_ITEMS = 5
DEFAULT_MAX_STRLEN = 300
DEFAULT_LIST_KEYS = "products,users,items,results,rows,records,data,entries"
DEFAULT_PRODUCT_WHITELIST = (
    "id,title,name,price,brand,category,rating,stock,availabilityStatus,"
    "description,sku,thumbnail"
)
DEFAULT_USER_WHITELIST = (
    "id,firstName,lastName,age,gender,email,phone,username,address,company,"
    "name,role"
)


@dataclass
class TrimPolicy:
    """Configuration for trimming behavior."""

    max_items: int = DEFAULT_MAX_ITEMS
    max_strlen: int = DEFAULT_MAX_STRLEN
    list_keys: set[str] = field(default_factory=set)
    product_whitelist: set[str] = field(default_factory=set)
    user_whitelist: set[str] = field(default_factory=set)
    truncate_strings: bool = True

    @classmethod
    def from_env(cls, env: Optional[dict[str, str]] = None) -> "TrimPolicy":
        """Create policy from environment variables."""
        import os

        env = env or os.environ

        def parse_set(key: str, default: str) -> set[str]:
            val = env.get(key, default)
            return {s.strip() for s in val.split(",") if s.strip()}

        return cls(
            max_items=int(env.get("TOKEN_GUARD_TRIM_MAX_ITEMS", DEFAULT_MAX_ITEMS)),
            max_strlen=int(env.get("TOKEN_GUARD_TRIM_MAX_STRLEN", DEFAULT_MAX_STRLEN)),
            list_keys=parse_set("TOKEN_GUARD_TRIM_LIST_KEYS", DEFAULT_LIST_KEYS),
            product_whitelist=parse_set(
                "TOKEN_GUARD_TRIM_PRODUCT_WHITELIST", DEFAULT_PRODUCT_WHITELIST
            ),
            user_whitelist=parse_set(
                "TOKEN_GUARD_TRIM_USER_WHITELIST", DEFAULT_USER_WHITELIST
            ),
            truncate_strings=env.get("TOKEN_GUARD_TRIM_STRINGS", "1") == "1",
        )


@dataclass
class TrimResult:
    """Result of trimming operation."""

    trimmed: Any
    notes: list[str]
    counters: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "notes": self.notes,
            "counters": self.counters,
        }


def trim_payload(obj: Any, policy: Optional[TrimPolicy] = None) -> TrimResult:
    """
    Trim a JSON payload according to policy.

    Args:
        obj: The object to trim (dict, list, or primitive).
        policy: Trimming policy. Uses defaults if not provided.

    Returns:
        TrimResult with trimmed object, notes, and counters.
    """
    if policy is None:
        policy = TrimPolicy.from_env()

    notes: list[str] = []
    counters: dict[str, int] = {
        "lists_limited": 0,
        "fields_dropped": 0,
        "strings_truncated": 0,
        "items_removed": 0,
    }

    def truncate_string(s: str, max_len: int) -> str:
        if len(s) <= max_len:
            return s
        counters["strings_truncated"] += 1
        return s[:max_len] + "..."

    def whitelist_dict(d: dict, whitelist: set[str], key_name: str) -> dict:
        if not whitelist:
            return d
        original_keys = set(d.keys())
        filtered = {k: v for k, v in d.items() if k in whitelist}
        dropped = original_keys - set(filtered.keys())
        if dropped:
            counters["fields_dropped"] += len(dropped)
        return filtered

    def process_value(val: Any, parent_key: str = "") -> Any:
        if isinstance(val, dict):
            return process_dict(val, parent_key)
        elif isinstance(val, list):
            return process_list(val, parent_key)
        elif isinstance(val, str) and policy.truncate_strings:
            return truncate_string(val, policy.max_strlen)
        return val

    def process_list(lst: list, key: str) -> list:
        original_len = len(lst)

        # Check if this is a list we should limit
        should_limit = key.lower() in policy.list_keys

        if should_limit and original_len > policy.max_items:
            counters["lists_limited"] += 1
            counters["items_removed"] += original_len - policy.max_items
            notes.append(f"{key}: {original_len} -> {policy.max_items}")
            lst = lst[: policy.max_items]

        # Determine whitelist based on key
        whitelist: set[str] = set()
        if key.lower() in ("products", "product"):
            whitelist = policy.product_whitelist
        elif key.lower() in ("users", "user"):
            whitelist = policy.user_whitelist

        result = []
        for item in lst:
            if isinstance(item, dict) and whitelist:
                item = whitelist_dict(item, whitelist, key)
                if whitelist == policy.product_whitelist and not any(
                    n.startswith(f"{key} fields") for n in notes
                ):
                    notes.append(f"{key} fields whitelisted")
                elif whitelist == policy.user_whitelist and not any(
                    n.startswith(f"{key} fields") for n in notes
                ):
                    notes.append(f"{key} fields whitelisted")
            result.append(process_value(item, key))

        return result

    def process_dict(d: dict, parent_key: str = "") -> dict:
        result = {}
        for k, v in d.items():
            result[k] = process_value(v, k)
        return result

    # Deep copy to avoid modifying original
    trimmed = copy.deepcopy(obj)

    # Process the object
    if isinstance(trimmed, dict):
        trimmed = process_dict(trimmed)
    elif isinstance(trimmed, list):
        trimmed = process_list(trimmed, "root")
    elif isinstance(trimmed, str) and policy.truncate_strings:
        trimmed = truncate_string(trimmed, policy.max_strlen)

    # Add metadata
    if isinstance(trimmed, dict):
        trimmed["__trimmed"] = True
        trimmed["__trim_policy"] = {
            "max_items": policy.max_items,
            "max_strlen": policy.max_strlen if policy.truncate_strings else None,
        }

    return TrimResult(trimmed=trimmed, notes=notes, counters=counters)


def extract_json_from_prompt(prompt: str) -> tuple[Optional[Any], str]:
    """
    Extract JSON payload from a prompt.

    Tries these methods in order:
    1. Wrapper markers: ### PAYLOAD START ... ### PAYLOAD END
    2. Fenced JSON: ```json ... ```
    3. Entire prompt is valid JSON

    Args:
        prompt: The prompt text to extract from.

    Returns:
        Tuple of (extracted_object, method_name).
        Returns (None, "none") if extraction fails.
    """
    import json

    # Method 1: Wrapper markers
    wrapper_pattern = r"###\s*PAYLOAD\s*START\s*\n(.*?)\n###\s*PAYLOAD\s*END"
    match = re.search(wrapper_pattern, prompt, re.DOTALL | re.IGNORECASE)
    if match:
        try:
            return json.loads(match.group(1).strip()), "wrapper"
        except json.JSONDecodeError:
            pass

    # Method 2: Fenced JSON block
    fence_pattern = r"```json\s*\n(.*?)\n```"
    match = re.search(fence_pattern, prompt, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip()), "fenced"
        except json.JSONDecodeError:
            pass

    # Method 3: Entire prompt is JSON
    stripped = prompt.strip()
    if stripped.startswith(("{", "[")):
        try:
            return json.loads(stripped), "raw"
        except json.JSONDecodeError:
            pass

    return None, "none"


if __name__ == "__main__":
    # Simple CLI for testing
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(description="Trim JSON payload")
    parser.add_argument("--file", help="JSON file to trim (or use stdin)")
    parser.add_argument("--max-items", type=int, default=DEFAULT_MAX_ITEMS)
    parser.add_argument("--max-strlen", type=int, default=DEFAULT_MAX_STRLEN)
    parser.add_argument("--minify", action="store_true", help="Output minified JSON")
    args = parser.parse_args()

    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = json.load(sys.stdin)

    policy = TrimPolicy(
        max_items=args.max_items,
        max_strlen=args.max_strlen,
    )
    policy.list_keys = {s.strip() for s in DEFAULT_LIST_KEYS.split(",")}
    policy.product_whitelist = {s.strip() for s in DEFAULT_PRODUCT_WHITELIST.split(",")}
    policy.user_whitelist = {s.strip() for s in DEFAULT_USER_WHITELIST.split(",")}

    result = trim_payload(data, policy)

    indent = None if args.minify else 2
    print(json.dumps(result.trimmed, indent=indent, ensure_ascii=False))

    print(f"\nNotes: {result.notes}", file=sys.stderr)
    print(f"Counters: {result.counters}", file=sys.stderr)
