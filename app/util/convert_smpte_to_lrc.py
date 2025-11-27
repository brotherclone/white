#!/usr/bin/env python3
"""Convert SMPTE-like timestamps to LRC-style timestamps.

Examples:
  Input:  01:01:06:11.49
  Output: [01:06.011]

Rules applied:
- Timestamps are matched by the pattern: (\d{1,2}:){2,3}\d{1,3}(?:[.,]\d+)?
- For a matched timestamp we take the last three colon-separated fields as MM:SS:FRAMES
  (this gracefully handles an optional leading hour field).
- The final field (frames, may contain a decimal) is converted by truncating to an integer
  (no multiplication or scaling). The integer is then zero-padded to 3 digits and used as
  the millisecond portion of the LRC timestamp.
- The result is formatted as [MM:SS.mmm]

The module exposes:
- convert_timestamp(s: str) -> str
- convert_text(text: str) -> str

CLI:
  python -m app.util.convert_smpte_to_lrc [--inplace] [files...]
  If no files are provided the script reads stdin and writes to stdout.

Use --inplace to overwrite files with the converted content.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Pattern matches e.g. 01:01:06:11.49 or 1:2:3:4 or 00:12:34:56
_TIMESTAMP_RE = re.compile(r"\b(?:\d{1,2}:){2,3}\d{1,3}(?:[.,]\d+)?\b")


def _normalize_int_part(s: str) -> int:
    """Return the truncated integer part of a numeric string (handles comma or dot decimal sep)."""
    s = s.replace(",", ".")
    try:
        # float -> int truncates toward zero
        return int(float(s))
    except Exception:
        # fallback: keep digits only
        digits = re.sub(r"[^0-9]", "", s)
        return int(digits) if digits else 0


def convert_timestamp(ts: str) -> str:
    """Convert a single SMPTE-like timestamp string to LRC-style [MM:SS.mmm].

    Example:
      convert_timestamp('01:01:06:11.49') -> '[01:06.011]'
    """
    parts = ts.split(":")
    if len(parts) < 3:
        # unexpected format: return original
        return ts

    # Use the last three components as MM, SS, FRAMES
    mm = parts[-3]
    ss = parts[-2]
    frames = parts[-1]

    # frames may contain a decimal part; truncate (no scaling)
    ms_int = _normalize_int_part(frames)
    # ensure mm and ss are two-digit, ms is three-digit zero-padded
    try:
        mm_i = int(mm)
        ss_i = int(ss)
    except Exception:
        # if mm/ss aren't numeric, fall back to original
        return ts

    return f"[{mm_i:02d}:{ss_i:02d}.{ms_int:03d}]"


def _replacer(match: re.Match) -> str:
    raw = match.group(0)
    try:
        return convert_timestamp(raw)
    except Exception:
        return raw


def convert_text(text: str) -> str:
    """Replace all SMPTE-like timestamps in text with LRC-style timestamps."""
    return _TIMESTAMP_RE.sub(_replacer, text)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert SMPTE-like timestamps to LRC [MM:SS.mmm]"
    )
    p.add_argument("files", nargs="*", help="Files to convert (reads stdin if omitted)")
    p.add_argument(
        "-i", "--inplace", action="store_true", help="Overwrite files in place"
    )
    return p.parse_args(argv)


def _process_file(p: Path, inplace: bool) -> int:
    text = p.read_text(encoding="utf8")
    out = convert_text(text)
    if inplace:
        p.write_text(out, encoding="utf8")
    else:
        sys.stdout.write(out)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.files:
        # read stdin and write stdout
        data = sys.stdin.read()
        sys.stdout.write(convert_text(data))
        return 0

    exit_code = 0
    for f in args.files:
        p = Path(f)
        if not p.exists():
            print(f"File not found: {f}", file=sys.stderr)
            exit_code = 2
            continue
        _process_file(p, inplace=args.inplace)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
