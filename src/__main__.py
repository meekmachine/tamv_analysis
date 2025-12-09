#!/usr/bin/env python3
"""
TAMV Analysis Package - Command Line Entry Point

Usage:
    python -m src extract --corpus brown       # Extract TAMV features to CSV
    python -m src cga --dataset wiki           # Analyze CGA speaker-level TAMV
"""

import sys


def main():
    if len(sys.argv) < 2:
        print("TAMV Analysis Package")
        print()
        print("Usage:")
        print("  python -m src extract [args]      # Extract TAMV features to CSV")
        print("  python -m src cga [args]          # Analyze CGA conversation-level TAMV")
        print("  python -m src utterance [args]    # Analyze TAMV-toxicity at utterance level")
        print()
        print("For help with each command, use: python -m src <command> --help")
        sys.exit(0)

    command = sys.argv[1]
    sys.argv = sys.argv[1:]  # Remove 'src' from argv

    if command == 'extract':
        from .extract_tamv import main as run_extract
        run_extract()
    elif command == 'cga':
        from .analyze_cga import main as run_cga
        run_cga()
    elif command == 'utterance':
        from .utterance_analysis import main as run_utterance
        run_utterance()
    else:
        print(f"Unknown command: {command}")
        print("Available commands: extract, cga, utterance")
        sys.exit(1)


if __name__ == '__main__':
    main()
