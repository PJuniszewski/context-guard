# context-guard

> **Note:** This plugin has been merged into the [juni](https://github.com/PJuniszewski/cook) plugin suite.
> For the latest version, install `juni-skills:juni` which includes `/juni:guard`.

---

**Prevents LLMs from reasoning with unjustified certainty when input data is incomplete.**

Detects when trimming would hide relevant data and blocks silent sampling for forensic queries.

## Installation

### Recommended: Install as part of Juni Suite

```bash
# Add the marketplace
claude /plugin
# Select "Add Marketplace" → enter: PJuniszewski/juni-skills-marketplace

# Install juni (includes guard)
claude /plugin install juni-skills:juni
claude /plugin enable juni

# Use as /juni:guard
/juni:guard data.json
```

### Standalone Installation (Legacy)

```bash
# Via marketplace
claude /plugin install juni-skills:context-guard
claude /plugin enable context-guard

# Via skills.sh
npx skills add PJuniszewski/context-guard
```

---

## Features

- **Lossless reduction** - Minify, columnar transform, remove nulls
- **Token counting** - API or heuristic fallback
- **Decision engine** - ALLOW / SAMPLE / BLOCK
- **Intelligent trimming** - First + last + evenly-spaced sampling
- **Forensic detection** - Warns when specific record queries detected

## Quick Start

```bash
# Analyze a file
/juni:guard my_data.json

# Analyze inline JSON data
/juni:guard '[{"id": 1}, {"id": 2}]'

# Force through despite warnings
/juni:guard my_data.json --force

# Check if forensic query is safe
/juni:guard logs.json --mode forensics

# Use larger token budget
/juni:guard data.json --budget-tokens 5000

# Get output as JSON for programmatic use
/juni:guard data.json --json
```

## Semantic Modes

| Mode | Sampling | Use Case |
|------|----------|----------|
| `analysis` | Allowed | "What categories exist?", "Price range?" |
| `summary` | Aggressive | "Describe the data structure" |
| `forensics` | **BLOCKED** | "Why did request id=X fail?" |

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `TOKEN_GUARD_MIN_CHARS` | `6000` | Below = always allow |
| `TOKEN_GUARD_WARN_CHARS` | `15000` | Above = warn |
| `TOKEN_GUARD_HARD_LIMIT_CHARS` | `100000` | Above = hard block |
| `TOKEN_GUARD_MODEL` | `claude-sonnet-4-20250514` | Model for token counting |
| `TOKEN_GUARD_PROMPT_LIMIT` | `3500` | Default token budget |
| `ANTHROPIC_API_KEY` | — | Required for token counting |

## Output Format

```
============================================================
CONTEXT GUARD ANALYSIS
============================================================

Decision: [OK] ALLOW | [~] SAMPLE | [X] BLOCK
Mode: analysis | summary | forensics

TOKEN ANALYSIS:
  Original:     5,234 tokens
  After reduce: 4,891 tokens (-343)
  After trim:   3,421 tokens
  Budget:       3,500 tokens

LOSSLESS REDUCTIONS:
  - Removed 47 null/empty fields
  - Columnar: 2 arrays transformed (18% reduction)

SAMPLING APPLIED:
  - products: 500 -> 25 (first + last + every ~20th)

============================================================
READY TO SEND PROMPT
============================================================
```

## License

MIT
