#!/usr/bin/env python3
"""Tests for JSON payload trimming."""
from __future__ import annotations

import sys
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from trim_lib import TrimPolicy, trim_payload


def test_list_limiting():
    """Test that lists are limited to max_items."""
    data = {
        "products": [
            {"id": i, "name": f"Product {i}"}
            for i in range(10)
        ]
    }

    policy = TrimPolicy(
        max_items=3,
        list_keys={"products"},
        product_whitelist=set(),
        user_whitelist=set(),
        truncate_strings=False,
    )

    result = trim_payload(data, policy)

    assert len(result.trimmed["products"]) == 3
    assert result.counters["lists_limited"] == 1
    assert result.counters["items_removed"] == 7
    assert any("products: 10 -> 3" in note for note in result.notes)


def test_list_limiting_multiple_keys():
    """Test that multiple list keys are limited."""
    data = {
        "products": [{"id": i} for i in range(8)],
        "users": [{"id": i} for i in range(6)],
        "items": [{"id": i} for i in range(4)],  # Under limit
    }

    policy = TrimPolicy(
        max_items=5,
        list_keys={"products", "users", "items"},
        product_whitelist=set(),
        user_whitelist=set(),
        truncate_strings=False,
    )

    result = trim_payload(data, policy)

    assert len(result.trimmed["products"]) == 5
    assert len(result.trimmed["users"]) == 5
    assert len(result.trimmed["items"]) == 4  # Not limited
    assert result.counters["lists_limited"] == 2


def test_product_whitelist():
    """Test that product fields are whitelisted."""
    data = {
        "products": [
            {
                "id": 1,
                "title": "Widget",
                "price": 99.99,
                "description": "A long description that should be removed",
                "internal_sku": "SKU123",
                "warehouse_location": "A1-B2",
            }
        ]
    }

    policy = TrimPolicy(
        max_items=10,
        list_keys={"products"},
        product_whitelist={"id", "title", "price"},
        user_whitelist=set(),
        truncate_strings=False,
    )

    result = trim_payload(data, policy)
    product = result.trimmed["products"][0]

    assert "id" in product
    assert "title" in product
    assert "price" in product
    assert "description" not in product
    assert "internal_sku" not in product
    assert "warehouse_location" not in product
    assert result.counters["fields_dropped"] == 3


def test_user_whitelist():
    """Test that user fields are whitelisted."""
    data = {
        "users": [
            {
                "id": 1,
                "firstName": "John",
                "lastName": "Doe",
                "email": "john@example.com",
                "ssn": "123-45-6789",  # Sensitive - should be removed
                "creditCard": "4111-1111-1111-1111",  # Sensitive
            }
        ]
    }

    policy = TrimPolicy(
        max_items=10,
        list_keys={"users"},
        product_whitelist=set(),
        user_whitelist={"id", "firstName", "lastName", "email"},
        truncate_strings=False,
    )

    result = trim_payload(data, policy)
    user = result.trimmed["users"][0]

    assert "id" in user
    assert "firstName" in user
    assert "email" in user
    assert "ssn" not in user
    assert "creditCard" not in user


def test_string_truncation():
    """Test that long strings are truncated."""
    data = {
        "description": "A" * 500,
        "short": "OK",
    }

    policy = TrimPolicy(
        max_items=10,
        max_strlen=100,
        list_keys=set(),
        product_whitelist=set(),
        user_whitelist=set(),
        truncate_strings=True,
    )

    result = trim_payload(data, policy)

    assert len(result.trimmed["description"]) == 103  # 100 + "..."
    assert result.trimmed["description"].endswith("...")
    assert result.trimmed["short"] == "OK"
    assert result.counters["strings_truncated"] == 1


def test_string_truncation_disabled():
    """Test that string truncation can be disabled."""
    data = {"long_text": "B" * 500}

    policy = TrimPolicy(
        max_items=10,
        max_strlen=100,
        list_keys=set(),
        product_whitelist=set(),
        user_whitelist=set(),
        truncate_strings=False,
    )

    result = trim_payload(data, policy)

    assert len(result.trimmed["long_text"]) == 500
    assert result.counters["strings_truncated"] == 0


def test_metadata_fields_exist():
    """Test that trimmed objects have metadata."""
    data = {"test": "value"}

    policy = TrimPolicy(
        max_items=5,
        max_strlen=100,
        list_keys=set(),
        product_whitelist=set(),
        user_whitelist=set(),
        truncate_strings=True,
    )

    result = trim_payload(data, policy)

    assert result.trimmed["__trimmed"] is True
    assert "__trim_policy" in result.trimmed
    assert result.trimmed["__trim_policy"]["max_items"] == 5
    assert result.trimmed["__trim_policy"]["max_strlen"] == 100


def test_nested_structure():
    """Test trimming of nested structures."""
    data = {
        "response": {
            "data": {
                "products": [
                    {"id": i, "name": f"P{i}", "extra": "remove"}
                    for i in range(10)
                ]
            }
        }
    }

    policy = TrimPolicy(
        max_items=3,
        list_keys={"products"},
        product_whitelist={"id", "name"},
        user_whitelist=set(),
        truncate_strings=False,
    )

    result = trim_payload(data, policy)
    products = result.trimmed["response"]["data"]["products"]

    assert len(products) == 3
    assert "extra" not in products[0]


def test_combined_trimming():
    """Test combination of list limiting, whitelisting, and truncation."""
    data = {
        "products": [
            {
                "id": i,
                "title": f"Product {i}",
                "description": "X" * 500,
                "internal_code": "ABC123",
            }
            for i in range(20)
        ]
    }

    policy = TrimPolicy(
        max_items=5,
        max_strlen=50,
        list_keys={"products"},
        product_whitelist={"id", "title", "description"},
        user_whitelist=set(),
        truncate_strings=True,
    )

    result = trim_payload(data, policy)

    assert len(result.trimmed["products"]) == 5
    assert result.counters["lists_limited"] == 1
    assert result.counters["items_removed"] == 15
    assert result.counters["fields_dropped"] == 5  # internal_code removed from 5 items
    assert result.counters["strings_truncated"] > 0

    # Check first product
    p = result.trimmed["products"][0]
    assert "id" in p
    assert "title" in p
    assert "description" in p
    assert "internal_code" not in p
    assert len(p["description"]) == 53  # 50 + "..."


def test_empty_input():
    """Test handling of empty inputs."""
    policy = TrimPolicy()

    result_dict = trim_payload({}, policy)
    assert result_dict.trimmed["__trimmed"] is True

    result_list = trim_payload([], policy)
    assert result_list.trimmed == []


def test_preserves_non_targeted_keys():
    """Test that keys not in list_keys are preserved without limiting."""
    data = {
        "products": [{"id": i} for i in range(10)],  # Should be limited
        "metadata": [{"key": i} for i in range(10)],  # Should NOT be limited
    }

    policy = TrimPolicy(
        max_items=3,
        list_keys={"products"},  # Only products
        product_whitelist=set(),
        user_whitelist=set(),
        truncate_strings=False,
    )

    result = trim_payload(data, policy)

    assert len(result.trimmed["products"]) == 3
    assert len(result.trimmed["metadata"]) == 10  # Preserved


if __name__ == "__main__":
    import traceback

    tests = [
        test_list_limiting,
        test_list_limiting_multiple_keys,
        test_product_whitelist,
        test_user_whitelist,
        test_string_truncation,
        test_string_truncation_disabled,
        test_metadata_fields_exist,
        test_nested_structure,
        test_combined_trimming,
        test_empty_input,
        test_preserves_non_targeted_keys,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            print(f"PASS: {test.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {test.__name__}")
            traceback.print_exc()
            failed += 1
        except Exception as e:
            print(f"ERROR: {test.__name__}: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
