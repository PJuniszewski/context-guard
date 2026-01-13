# context-guard

**Context-Guard prevents LLMs from reasoning with unjustified certainty when input data is incomplete.**

LLM Epistemic Safety Layer - context integrity enforcement with sampling-aware guardrails. Automatically detects when trimming would compromise analytical accuracy and blocks silent sampling.

## Features

### Layer 1: Automatic Hook Protection
- **Token counting** via Anthropic API before prompts are processed
- **Automatic blocking** when prompts exceed configurable limits
- **Intelligent JSON trimming** with first-last-even sampling strategy
- **Semantic mode detection** (analysis/summary/forensics)
- **Forensic question heuristic** - blocks sampling when specific record lookups are detected

### Layer 2: Power User Command
- `/contextguard preview` - Preview what trimming would do
- `/contextguard diff` - Compare full vs sampled data
- `/contextguard search` - Verify if forensic analysis is safe

## Installation

### Via Juni-Tools Marketplace

```bash
# Add the marketplace (run /plugin, select "Add Marketplace", enter: PJuniszewski/juni-tools-marketplace)

# Install and enable
claude /plugin install juni-tools:context-guard
claude /plugin enable context-guard
```

### Set environment variables

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export TOKEN_GUARD_MODEL="claude-sonnet-4-20250514"
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `TOKEN_GUARD_MODEL` | `claude-sonnet-4-20250514` | Model for token counting |
| `TOKEN_GUARD_PROMPT_LIMIT` | `3500` | Block prompts above this token count |
| `TOKEN_GUARD_MIN_CHARS_BEFORE_COUNT` | `6000` | Skip API call for small prompts |
| `TOKEN_GUARD_MODE` | `block` | Mode: `block`, `warn`, or `off` |

## Semantic Modes

Context Guard supports three semantic modes that control trimming behavior:

| Mode | Sampling | Use Case |
|------|----------|----------|
| `analysis` | Allowed | Default. "What categories exist?", "Price range?" |
| `summary` | Aggressive | "Describe the data structure" |
| `forensics` | **BLOCKED** | "Why did request id=X fail?" |

### Mode Markers

Add markers to your prompt to explicitly set the mode:

```
#trimmer:mode=analysis    # Allow sampling (default)
#trimmer:mode=summary     # Allow aggressive trimming
#trimmer:mode=forensics   # Block if data exceeds limit
```

### Forensic Question Detection

Context Guard automatically detects forensic questions and blocks sampling **only when data loss is possible**:

| Forensic Pattern | Payload Size | Result |
|------------------|--------------|--------|
| Detected | Small (≤ limit) | **ALLOW** - no data loss risk |
| Detected | Large (> limit) | **BLOCK** - sampling would hide answer |
| Not detected | Any size | Normal processing |

```
# These patterns trigger forensic detection:
"Why did request id=abc123 fail?"
"What happened to user id: xyz789?"
"Show details for transaction TX-12345"
```

To override blocking: add `#trimmer:mode=analysis` or `#trimmer:force`

## Usage

### Automatic Hook (Layer 1)

The hook intercepts prompts automatically:

```
Prompt: 5000 tokens (limit: 3500). No JSON detected.

Options:
  - Wrap JSON with: ### PAYLOAD START / ### PAYLOAD END
  - Force: add #trimmer:force to your prompt
  - Or reduce payload manually
```

### With JSON Payload

Wrap your JSON with markers for automatic trimming:

```
### PAYLOAD START
{"products": [...], "users": [...]}
### PAYLOAD END

Analyze this data
```

Trimmed files are saved to `.claude/trimmer/`:
- `trimmed-TIMESTAMP.json` - Pretty printed
- `trimmed-TIMESTAMP.min.json` - Minified for pasting

### Power User Command (Layer 2)

```
/contextguard preview data.json --budget 1000
/contextguard diff data.json --budget 500
/contextguard search data.json --query "Why did request id=xyz fail?"
```

### Escape Hatches

- `#trimmer:off` - Disable for this prompt
- `#trimmer:force` - Bypass all blocking
- `#trimmer:mode=analysis` - Explicitly allow sampling for forensic-looking questions

## How It Works

### Layer 1: Hook

1. Hook intercepts `UserPromptSubmit` event
2. If prompt > `MIN_CHARS`, count tokens via Anthropic API
3. Detect semantic mode from markers or infer from question
4. **Forensic + size check** (semantics first, then size):
   - If forensic pattern detected AND tokens > `LIMIT` → BLOCK
   - If tokens ≤ `LIMIT` → ALLOW (even if forensic - no data loss risk)
5. If tokens > `LIMIT` and not forensic:
   - Extract JSON from prompt (if present)
   - Apply first-last-even sampling strategy
   - Include trim report in additionalContext
6. Otherwise, allow prompt

### Layer 2: Command

Provides direct access to trimming utilities for power users who need to understand what's happening before asking questions.

## Epistemic Safety

Context Guard prevents confident hallucinations by:

1. **Explicit warnings** in trim reports about sampling limitations
2. **Forensic mode** that blocks instead of sampling
3. **Heuristic detection** of forensic questions
4. **First-last-even sampling** that preserves data distribution

### What Might Be Hidden

When data is sampled:
- Single-record anomalies in middle positions
- Trend reversals after first 60% of data
- Rare outliers that fall outside sample

For forensic queries requiring ALL records, use forensics mode.

## Tests

```bash
# Unit tests
pytest tests/test_semantic_modes.py -v

# LLM context tests (requires ANTHROPIC_API_KEY)
pytest tests/test_llm_context.py -v -m llm

# Adversarial tests
pytest tests/test_llm_adversarial.py -v -m llm
```

## License

MIT
