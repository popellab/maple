#!/usr/bin/env python3
"""
CLI wrapper for batch monitoring.

Entry point: qsp-batch-monitor
"""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Monitor OpenAI batch job progress",
        epilog="""
Examples:
    qsp-batch-monitor batch_abc123
    qsp-batch-monitor batch_abc123 --timeout 7200
        """
    )

    parser.add_argument(
        "batch_id",
        help="OpenAI batch ID to monitor"
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Timeout in seconds (default: 3600)"
    )

    args = parser.parse_args()

    # Import and run the batch monitor
    from qsp_llm_workflows.run.batch_monitor import main as monitor_main

    # Set up sys.argv for the monitor script
    sys.argv = [
        "batch_monitor.py",
        args.batch_id
    ]

    if args.timeout:
        sys.argv.extend(["--timeout", str(args.timeout)])

    try:
        monitor_main()
    except KeyboardInterrupt:
        print("\nMonitoring interrupted by user", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
