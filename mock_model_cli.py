#!/usr/bin/env python3
"""Mock model CLI for testing the binary retention benchmark.

Simulates a model that remembers all binary numbers it receives
and responds correctly to recall prompts.

Usage:
  set MODEL_CMD="python mock_model_cli.py"
  python binary_retention_benchmark.py
"""

import json
import re
import sys
from pathlib import Path

DATA_FILE = Path(__file__).with_name("binary_numbers.json")

# Load the dataset so we know the correct answers
if DATA_FILE.is_file():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        DATASET = {entry["id"]: entry["binary"] for entry in json.load(f)}
else:
    DATASET = {}

# Read the prompt from stdin
prompt = sys.stdin.read().strip()

# Check if this is a recall prompt
match = re.search(r"give me the binary number (\d+)", prompt, re.IGNORECASE)
if match:
    idx = int(match.group(1))
    if idx in DATASET:
        print(DATASET[idx])
    else:
        print(f"I don't have a number at index {idx}.")
else:
    # For regular prompts (binary numbers or distractions), just acknowledge
    print("OK")
