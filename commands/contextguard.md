---
description: Context Guard utilities - preview trimming, diff analysis, search safety check
arguments:
  - name: subcommand
    description: "Action: preview, diff, or search"
    required: true
  - name: file
    description: Path to JSON file to analyze
    required: true
  - name: --budget
    description: Token budget (default 1000)
    required: false
  - name: --query
    description: Query for search mode
    required: false
---

# Context Guard Command

You are executing the `/contextguard` command for epistemic safety analysis.

## Arguments
- Subcommand: $ARGUMENTS.subcommand
- File: $ARGUMENTS.file
- Budget: $ARGUMENTS.--budget (default: 1000)
- Query: $ARGUMENTS.--query (for search)

## Instructions

1. Read the JSON file at the provided path
2. Use `trim_lib.py` from `${CLAUDE_PLUGIN_ROOT}/scripts/` to analyze the data
3. Based on the subcommand, provide the appropriate analysis:

### preview

Show what trimming would do:
- Original token count
- Trimmed token count
- Reduction percentage
- Which arrays were trimmed
- Sample of trimmed data (first 3 records)

### diff

Compare full vs sampled:
- Side-by-side token counts
- List of trimmed arrays with record counts
- Sample indices that would be kept
- Warning about what information might be lost:
  - Single-record anomalies in middle positions
  - Trend reversals after first 60% of data
  - Rare outliers outside sample

### search

Check if searching for specific records is safe:
- If data fits within budget → OK, all records preserved
- If data exceeds budget → BLOCK with options:
  1. Increase budget to fit all data
  2. Pre-filter data externally (jq, grep)
  3. Accept sampling risk with `#trimmer:mode=analysis`

## Key Functions from trim_lib

```python
from trim_lib import (
    intelligent_trim,
    TokenBudget,
    SamplingStrategy,
    estimate_tokens_heuristic,
)

# Estimate tokens
tokens = estimate_tokens_heuristic(json.dumps(data))

# Trim data
trimmed, report = intelligent_trim(
    data,
    token_budget=TokenBudget(total_budget=budget, min_records=5),
    sampling_strategy=SamplingStrategy.FIRST_LAST_EVEN,
)

# Get report
print(report.to_claude_context())
print(f"Original: {report.original_tokens}")
print(f"Trimmed: {report.trimmed_tokens}")
```

## Output Format

Use clear section headers. Show exact numbers and percentages.
For search mode, be explicit about whether the query can proceed safely.
