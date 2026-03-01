#!/usr/bin/env python3
"""
CLI wrapper for validation workflow.

Entry point: qsp-validate
"""

# Import and run the validation module directly
from maple.validate.run_all_validations import main


if __name__ == "__main__":
    main()
