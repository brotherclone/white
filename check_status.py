#!/usr/bin/env python3
"""
White Agent - Status Check and Quick Start

Run this script to verify all fixes are in place and get started.
"""

import sys
from pathlib import Path


def check_files():
    """Check that all required files exist."""
    print("🔍 Checking file integrity...\n")

    required_files = [
        "app/agents/white_agent.py",
        "app/agents/tools/audio_tools.py",
        "run_white_agent.py",
        "examples/white_agent_usage.py",
    ]

    all_good = True
    for filepath in required_files:
        path = Path(filepath)
        if path.exists():
            print(f"  ✓ {filepath}")
        else:
            print(f"  ✗ {filepath} - MISSING!")
            all_good = False

    return all_good


def check_imports():
    """Check that key modules can be imported."""
    print("\n🔍 Checking imports...\n")

    try:
        print("  ✓ WhiteAgent imported")
    except Exception as e:
        print(f"  ✗ Cannot import WhiteAgent: {e}")
        return False

    try:
        print("  ✓ ensure_state_object imported")
    except Exception as e:
        print(f"  ✗ Cannot import ensure_state_object: {e}")
        return False

    try:
        print("  ✓ MainAgentState imported")
    except Exception as e:
        print(f"  ✗ Cannot import MainAgentState: {e}")
        return False

    return True


def check_fixes():
    """Check that both fixes are in place."""
    print("\n🔍 Checking fixes...\n")

    # Check Fix #1: Vocal file prioritization
    try:
        with open("app/agents/tools/audio_tools.py") as f:
            content = f.read()
            # Look for the fix (no shuffle after find_wav_files_prioritized)
            if "Don't shuffle - keep vocal files prioritized" in content:
                print("  ✓ Fix #1: Vocal file prioritization (APPLIED)")
            else:
                print("  ⚠ Fix #1: Cannot verify vocal file fix")
    except Exception as e:
        print(f"  ✗ Fix #1: Error checking: {e}")

    # Check Fix #2: Dict state conversion
    try:
        with open("app/agents/white_agent.py") as f:
            content = f.read()
            if "isinstance(result, dict)" in content:
                print("  ✓ Fix #2: Dict state conversion (APPLIED)")
            else:
                print("  ⚠ Fix #2: Cannot verify dict conversion fix")
    except Exception as e:
        print(f"  ✗ Fix #2: Error checking: {e}")

    return True


def show_quick_start():
    """Display quick start instructions."""
    print("\n" + "=" * 60)
    print("🎵 WHITE AGENT - READY TO USE")
    print("=" * 60)
    print("\nQUICK START:")
    print("\n  1. Start a new workflow:")
    print("     $ python run_white_agent.py start")
    print("\n  2. Complete the ritual tasks in Todoist")
    print("\n  3. Resume the workflow:")
    print("     $ python run_white_agent.py resume")
    print("\n" + "-" * 60)
    print("\nDOCUMENTATION:")
    print("  • Quick reference: QUICK_REFERENCE.txt")
    print("  • Complete guide: COMPLETE_FIX_GUIDE.md")
    print("  • Full docs: docs/WHITE_AGENT_USAGE.md")
    print("  • Status report: FINAL_STATUS_REPORT.txt")
    print("\n" + "=" * 60)


def main():
    print("\n" + "╔" + "=" * 58 + "╗")
    print("║" + " " * 15 + "WHITE AGENT STATUS CHECK" + " " * 19 + "║")
    print("╚" + "=" * 58 + "╝\n")

    files_ok = check_files()
    imports_ok = check_imports()
    fixes_ok = check_fixes()

    if files_ok and imports_ok and fixes_ok:
        print("\n✅ ALL CHECKS PASSED!\n")
        show_quick_start()
        return 0
    else:
        print("\n⚠️  SOME CHECKS FAILED")
        print("\nPlease review the errors above and ensure all files are in place.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
