#!/usr/bin/env python3
"""
Trimmer Hook for Claude Code.

Catches oversized prompts and produces trimmed payloads to keep context healthy.
Reads hook input from stdin, outputs JSON decision to stdout.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Import sibling modules
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from token_count import count_tokens_safe
from trim_lib import TrimPolicy, TrimResult, extract_json_from_prompt, trim_payload

# Configuration defaults
DEFAULT_MODEL = "claude-sonnet-4-20250514"
DEFAULT_PROMPT_LIMIT = 3500
DEFAULT_WARN_LIMIT = 2500
DEFAULT_MIN_CHARS = 6000
DEFAULT_MODE = "block"  # block = stop and show reason (auto mode may not be supported)

# Escape hatch markers
MARKER_OFF = "#trimmer:off"
MARKER_FORCE = "#trimmer:force"


def debug_log(msg: str) -> None:
    """Write debug message to file (disabled in production)."""
    pass  # Uncomment below for debugging
    # try:
    #     with open("/tmp/trimmer_debug.log", "a") as f:
    #         f.write(f"{datetime.now().isoformat()} - {msg}\n")
    # except:
    #     pass


def get_config() -> dict[str, Any]:
    """Load configuration from environment variables."""
    return {
        "model": os.environ.get("TOKEN_GUARD_MODEL", DEFAULT_MODEL),
        "prompt_limit": int(os.environ.get("TOKEN_GUARD_PROMPT_LIMIT", DEFAULT_PROMPT_LIMIT)),
        "warn_limit": int(os.environ.get("TOKEN_GUARD_WARN_LIMIT", DEFAULT_WARN_LIMIT)),
        "min_chars": int(os.environ.get("TOKEN_GUARD_MIN_CHARS_BEFORE_COUNT", DEFAULT_MIN_CHARS)),
        "mode": os.environ.get("TOKEN_GUARD_MODE", DEFAULT_MODE).lower(),
    }


def out(obj: dict, exit_code: int = 0) -> None:
    """Write JSON response to stdout and exit."""
    sys.stdout.write(json.dumps(obj, separators=(",", ":")))
    sys.stdout.flush()
    sys.exit(exit_code)


def allow() -> None:
    """Return allow decision."""
    debug_log("Outputting allow decision")
    out({
        "hookSpecificOutput": {"hookEventName": "UserPromptSubmit"},
        "suppressOutput": True,
    })


def block(reason: str) -> None:
    """Block the prompt with exit code 2."""
    debug_log(f"Blocking: {reason[:100]}")
    # For UserPromptSubmit, exit code 2 blocks the prompt
    # stderr is shown to user as the block reason
    print(reason, file=sys.stderr)
    sys.exit(2)


def warn(reason: str) -> None:
    """Return allow decision with suppressOutput (warn mode)."""
    # In warn mode, we allow but print warning to stderr
    print(f"[trimmer] WARNING: {reason}", file=sys.stderr)
    out({
        "hookSpecificOutput": {"hookEventName": "UserPromptSubmit"},
        "suppressOutput": True,
    })


def write_trimmed_file(
    trimmed_obj: Any,
    original_tokens: int,
    limit: int,
    method: str,
    trim_result: TrimResult,
) -> str:
    """
    Write trimmed payload to file.

    Returns the file path.
    """
    # Find project root (walk up to find .claude or .git)
    cwd = Path.cwd()
    project_root = cwd
    for parent in [cwd] + list(cwd.parents):
        if (parent / ".claude").exists() or (parent / ".git").exists():
            project_root = parent
            break

    # Create output directory
    output_dir = project_root / ".claude" / "trimmer"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_name = f"trimmed-{timestamp}"

    # Write pretty JSON
    pretty_path = output_dir / f"{base_name}.json"
    with open(pretty_path, "w", encoding="utf-8") as f:
        json.dump(trimmed_obj, f, indent=2, ensure_ascii=False)

    # Write minified JSON
    min_path = output_dir / f"{base_name}.min.json"
    with open(min_path, "w", encoding="utf-8") as f:
        json.dump(trimmed_obj, f, separators=(",", ":"), ensure_ascii=False)

    # Write metadata
    meta_path = output_dir / f"{base_name}.meta.json"
    meta = {
        "timestamp": datetime.now().isoformat(),
        "original_tokens": original_tokens,
        "limit": limit,
        "extraction_method": method,
        "trim_stats": trim_result.to_dict(),
        "files": {
            "pretty": str(pretty_path),
            "minified": str(min_path),
        },
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return str(pretty_path)


def format_block_reason(
    tokens: int,
    limit: int,
    method: str,
    file_path: str,
    trim_result: TrimResult,
) -> str:
    """Format block reason with options for user."""
    # Get relative path for cleaner display
    try:
        rel_path = Path(file_path).relative_to(Path.cwd())
    except ValueError:
        rel_path = Path(file_path)

    min_path = str(rel_path).replace(".json", ".min.json")
    notes_str = "; ".join(trim_result.notes) if trim_result.notes else "trimmed"

    return (
        f"Prompt: {tokens} tokens (limit: {limit})\n"
        f"Trimmed: {notes_str}\n"
        f"File: {min_path}\n\n"
        "Paste content from .min.json or add #trimmer:force to bypass"
    )


def format_extraction_failed_reason(tokens: int, limit: int) -> str:
    """Format reason when JSON extraction fails."""
    return (
        f"Prompt: {tokens} tokens (limit: {limit}). No JSON detected.\n\n"
        "Options:\n"
        "  - Wrap JSON with: ### PAYLOAD START / ### PAYLOAD END\n"
        "  - Force: add #trimmer:force to your prompt\n"
        "  - Or reduce payload manually"
    )


def run_hook(hook_input: dict[str, Any]) -> None:
    """
    Main hook logic. Calls allow/block/warn which exit the process.

    Args:
        hook_input: Parsed JSON from stdin with "prompt" field.
    """
    debug_log("run_hook started")
    config = get_config()
    debug_log(f"config: min_chars={config['min_chars']}, limit={config['prompt_limit']}")

    # Mode check
    if config["mode"] == "off":
        debug_log("Mode is off, allowing")
        allow()

    # Extract prompt
    prompt = hook_input.get("prompt", "")
    if not prompt:
        debug_log("No prompt, allowing")
        allow()

    # Escape hatch: #trimmer:off
    if MARKER_OFF in prompt:
        debug_log("Marker off found, allowing")
        allow()

    # Escape hatch: #trimmer:force
    force_mode = MARKER_FORCE in prompt

    # Early exit for small prompts
    if len(prompt) < config["min_chars"] and not force_mode:
        debug_log(f"Small prompt ({len(prompt)} < {config['min_chars']}), allowing")
        allow()

    # Count tokens
    tokens = count_tokens_safe(prompt, config["model"])
    if tokens < 0:
        # Token counting failed - fail open
        print("[trimmer] Token count failed, allowing prompt", file=sys.stderr)
        allow()

    # Check force bypass
    if force_mode:
        if tokens > config["prompt_limit"]:
            print(
                f"[trimmer] Force bypass: {tokens} tokens (limit: {config['prompt_limit']})",
                file=sys.stderr,
            )
        allow()

    # Determine threshold and mode
    mode = config["mode"]
    threshold = config["prompt_limit"]

    if mode == "warn":
        threshold = config["warn_limit"]

    # Check if under threshold
    if tokens <= threshold:
        allow()

    # Over threshold - attempt to extract and trim JSON
    extracted, method = extract_json_from_prompt(prompt)

    if extracted is None:
        # Extraction failed - block or warn
        reason = format_extraction_failed_reason(tokens, threshold)
        if mode == "warn":
            warn(reason)
        block(reason)

    # Trim the payload
    policy = TrimPolicy.from_env()
    trim_result = trim_payload(extracted, policy)

    # Write trimmed files for reference
    file_path = write_trimmed_file(
        trim_result.trimmed,
        tokens,
        threshold,
        method,
        trim_result,
    )

    # Get relative path for display
    try:
        rel_path = Path(file_path).relative_to(Path.cwd())
    except ValueError:
        rel_path = Path(file_path)

    notes_str = "; ".join(trim_result.notes) if trim_result.notes else "trimmed"

    # Handle based on mode
    if mode == "warn":
        # Warn mode: allow but show warning
        print(f"\n{'─' * 50}", file=sys.stderr)
        print(f"  TRIMMER WARNING: {tokens} tokens (limit {threshold})", file=sys.stderr)
        print(f"  {notes_str}", file=sys.stderr)
        print(f"  Trimmed file: {rel_path}", file=sys.stderr)
        print(f"{'─' * 50}\n", file=sys.stderr)
        allow()

    # Block mode: stop and show reason
    reason = format_block_reason(tokens, threshold, method, file_path, trim_result)
    block(reason)


def main() -> None:
    """Entry point for hook."""
    debug_log("Hook started")

    # Support --trim-file mode for direct trimming
    if len(sys.argv) > 1:
        if sys.argv[1] == "--trim-file" and len(sys.argv) > 2:
            file_path = sys.argv[2]
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                policy = TrimPolicy.from_env()
                result = trim_payload(data, policy)
                print(json.dumps(result.trimmed, indent=2, ensure_ascii=False))
                print(f"\nNotes: {result.notes}", file=sys.stderr)
                print(f"Counters: {result.counters}", file=sys.stderr)
            except Exception as e:
                print(f"Error: {e}", file=sys.stderr)
                sys.exit(1)
            return
        elif sys.argv[1] == "--help":
            print("Usage: trimmer_hook.py [--trim-file <path>]")
            print("  Without args: reads hook input from stdin")
            print("  --trim-file: trim a JSON file directly")
            return

    # Normal hook mode: read from stdin
    try:
        debug_log("Reading stdin")
        raw = sys.stdin.read()
        debug_log(f"Got input: {len(raw)} bytes")
        if not raw.strip():
            debug_log("Empty input, allowing")
            allow()  # Exits process

        hook_input = json.loads(raw)
        debug_log(f"Parsed JSON, prompt length: {len(hook_input.get('prompt', ''))}")

        # Run hook (calls allow/block/warn which exit)
        run_hook(hook_input)
    except Exception as e:
        debug_log(f"Exception: {e}")
        print(f"[trimmer] Error: {e}", file=sys.stderr)
        allow()  # Fail open, exits process


if __name__ == "__main__":
    main()
