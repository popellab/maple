#!/usr/bin/env python3
"""
Simple CLI pretty printer for CSV files with long text fields.
Formats CSV data in a readable table with proper text wrapping.
"""

import argparse
import csv
import textwrap
from pathlib import Path


def wrap_text(text, width=50):
    """Wrap text to specified width."""
    if not text:
        return ""
    return "\n".join(textwrap.wrap(str(text), width=width))


def print_csv_pretty(csv_path, width=50):
    """Print CSV file in a pretty formatted table."""
    csv_path = Path(csv_path)

    if not csv_path.exists():
        print(f"Error: File {csv_path} does not exist")
        return

    print(f"\n📄 {csv_path.name}")
    print("=" * 80)

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for i, row in enumerate(reader, 1):
            print(f"\n[{i}] {row.get('test_statistic_id', 'Unknown ID')}")
            print("-" * 60)

            for field, value in row.items():
                if field == 'test_statistic_id':
                    continue

                field_name = field.replace('_', ' ').title()
                wrapped_value = wrap_text(value, width)

                print(f"{field_name}:")
                if wrapped_value:
                    for line in wrapped_value.split('\n'):
                        print(f"  {line}")
                else:
                    print("  (empty)")
                print()

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Pretty print CSV files with long text fields",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pretty_print_csv.py file.csv
  python pretty_print_csv.py file.csv --width 60
        """
    )

    parser.add_argument(
        'csv_file',
        help='Path to the CSV file to pretty print'
    )

    parser.add_argument(
        '--width', '-w',
        type=int,
        default=70,
        help='Text wrap width (default: 70)'
    )

    args = parser.parse_args()

    print_csv_pretty(args.csv_file, args.width)


if __name__ == "__main__":
    main()