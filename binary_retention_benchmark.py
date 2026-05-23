#!/usr/bin/env python3
"""Retention Benchmark for LLMs — Binary Sequence.

Tests a model's ability to retain 1,000 × 28-bit binary numbers
presented one per prompt, with distraction prompts interleaved,
then asks it to recall a specific number by index.

Supports:
  - OpenAI-compatible API (via API_BASE, API_KEY, MODEL_NAME env vars)
  - CLI wrapper (via MODEL_CMD env var, reads prompt from stdin)

Usage:
  # With OpenAI-compatible API:
  set API_BASE=https://api.openai.com/v1
  set API_KEY=sk-...
  set MODEL_NAME=gpt-4o-mini
  python binary_retention_benchmark.py

  # With OpenRouter:
  set API_BASE=https://openrouter.ai/api/v1
  set API_KEY=sk-or-...
  set MODEL_NAME=openai/gpt-oss-120b:free

  # With Ollama (CLI mode):
  set MODEL_CMD=ollama run llama2

  # Options:
  set NUM_PROMPTS=50           # how many binary numbers to send (default: 50)
  set NUM_RECALLS=10           # how many numbers to recall (default: 10)
  set DISTRACTION_EVERY=15     # distraction frequency (default: 15)
  set REQUEST_DELAY=1.0        # seconds between requests (default: 1.0)
"""

import json
import os
import random
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

DATA_FILE = Path(__file__).with_name("binary_numbers.json")

DISTRACTION_PROMPTS = [
    "What is the capital of Japan?",
    "Tell me a joke.",
    "What color is the sky?",
    "How many days are in a week?",
    "What is 2 + 2?",
    "Name a fruit that is yellow.",
    "What year did the moon landing happen?",
    "What is the boiling point of water in Celsius?",
    "Who wrote Romeo and Juliet?",
    "What continent is Egypt in?",
    "How many letters are in the word 'alphabet'?",
    "What animal says 'moo'?",
    "What is the largest planet in our solar system?",
    "What gas do plants absorb?",
    "How many sides does a triangle have?",
    "What is the opposite of hot?",
    "What instrument has keys, pedals, and strings?",
    "What language is spoken in Brazil?",
    "What shape is a stop sign?",
    "What is the chemical symbol for gold?",
]


def generate_binary_numbers(count: int = 1000, bits: int = 28) -> list:
    """Generate unique 28-bit binary strings."""
    seen = set()
    numbers = []
    while len(numbers) < count:
        n = random.getrandbits(bits)
        b = format(n, f"0{bits}b")
        if b not in seen:
            seen.add(b)
            numbers.append({"id": len(numbers), "binary": b})
    return numbers


def load_or_generate_data() -> list:
    """Load binary numbers from file, or generate and save them."""
    if DATA_FILE.is_file():
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    print(f"Generating {DATA_FILE} with 1000 × 28-bit binary numbers...",
          file=sys.stderr)
    data = generate_binary_numbers()
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return data


def query_model_api(messages: list) -> str:
    """Query an OpenAI-compatible API with a message history."""
    api_base = os.environ.get("API_BASE", "").rstrip("/")
    api_key = os.environ.get("API_KEY", "")
    model = os.environ.get("MODEL_NAME", "")
    max_retries = int(os.environ.get("MAX_RETRIES", "5"))

    if not all([api_base, api_key, model]):
        print("Error: API_BASE, API_KEY, and MODEL_NAME must be set.",
              file=sys.stderr)
        sys.exit(1)

    url = f"{api_base}/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 4096,
    }).encode("utf-8")

    for attempt in range(max_retries):
        req = urllib.request.Request(url, data=payload, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Bearer {api_key}")
        req.add_header("User-Agent", "ai-model-tester/1.0")

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw_body = resp.read().decode("utf-8")
                try:
                    body = json.loads(raw_body)
                    if "choices" not in body or len(body["choices"]) == 0:
                        print(f"API response has no choices: {raw_body[:200]}", file=sys.stderr)
                        return ""
                    if "message" not in body["choices"][0]:
                        print(f"API response has no message: {raw_body[:200]}", file=sys.stderr)
                        return ""
                    message = body["choices"][0]["message"]
                    content = message.get("content") or message.get("reasoning")
                    if content is None:
                        print(f"API response has null content: {raw_body[:200]}", file=sys.stderr)
                        return ""
                    return content.strip()
                except json.JSONDecodeError:
                    print(f"Invalid JSON response: {raw_body[:200]}", file=sys.stderr)
                    return ""
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            if e.code == 429:
                wait = 2 ** attempt * 5  # 5s, 10s, 20s, 40s, 80s
                print(f"  Rate limited (429). Retrying in {wait}s... (attempt {attempt + 1}/{max_retries})", file=sys.stderr)
                time.sleep(wait)
                continue
            print(f"API error {e.code}: {error_body}", file=sys.stderr)
            return ""
        except Exception as e:
            print(f"Request failed ({type(e).__name__}): {e}", file=sys.stderr)
            import traceback; traceback.print_exc(file=sys.stderr)
            return ""

    print("  Max retries exceeded for 429 errors", file=sys.stderr)
    return ""


def query_model_cli(messages: list) -> str:
    """Query a model via CLI wrapper (MODEL_CMD)."""
    import subprocess

    cmd = os.environ.get("MODEL_CMD", "")
    if not cmd:
        print("Error: MODEL_CMD must be set.", file=sys.stderr)
        sys.exit(1)

    # For CLI, concatenate all user messages into a single prompt
    prompt = "\n".join(m["content"] for m in messages if m["role"] == "user")
    try:
        result = subprocess.run(
            cmd, shell=True, input=prompt.encode("utf-8"),
            capture_output=True, timeout=120
        )
        return result.stdout.decode("utf-8").strip()
    except Exception as e:
        print(f"CLI error: {e}", file=sys.stderr)
        return ""


def query_model(messages: list) -> str:
    """Route to API or CLI based on environment."""
    if os.environ.get("API_BASE"):
        return query_model_api(messages)
    return query_model_cli(messages)


def select_recall_indices(total: int, num_recalls: int) -> list:
    """Select evenly spaced indices across the sequence."""
    return [round(i * (total - 1) / (num_recalls - 1)) for i in range(num_recalls)]


def run_benchmark():
    """Run the retention benchmark."""
    data = load_or_generate_data()
    num_prompts = int(os.environ.get("NUM_PROMPTS", "50"))
    num_recalls = int(os.environ.get("NUM_RECALLS", "10"))
    distraction_every = int(os.environ.get("DISTRACTION_EVERY", "15"))
    request_delay = float(os.environ.get("REQUEST_DELAY", "1.0"))

    # Limit data to NUM_PROMPTS
    data = data[:num_prompts]

    # Auto-adjust recalls if more than prompts
    if num_recalls > len(data):
        num_recalls = len(data)

    # Select recall indices spread across the sequence
    recall_indices = select_recall_indices(len(data), num_recalls)

    print(f"Sending {len(data)} binary numbers to model...", file=sys.stderr)
    print(f"Distraction every {distraction_every} prompts", file=sys.stderr)
    print(f"Will recall {num_recalls} numbers at indices: {recall_indices}",
          file=sys.stderr)

    # Feed all binary numbers, building conversation history
    conversation_history = []
    for i, entry in enumerate(data):
        conversation_history.append({"role": "user", "content": entry["binary"]})
        response = query_model(conversation_history)
        if response:
            conversation_history.append({"role": "assistant", "content": response})
        else:
            # Remove orphaned user message if no response was received
            conversation_history.pop()
        time.sleep(request_delay)

        # Interleave distraction prompts
        if (i + 1) % distraction_every == 0 and i < len(data) - 1:
            distraction = random.choice(DISTRACTION_PROMPTS)
            conversation_history.append({"role": "user", "content": distraction})
            response = query_model(conversation_history)
            if response:
                conversation_history.append({"role": "assistant", "content": response})
            else:
                conversation_history.pop()
            time.sleep(request_delay)

        progress_interval = max(1, len(data) // 10)
        if (i + 1) % progress_interval == 0:
            print(f"  [{i+1}/{len(data)}] numbers sent", file=sys.stderr)

    # Recall phase
    print(f"\nRecalling {num_recalls} numbers...", file=sys.stderr)
    results = []
    correct = 0

    for idx in recall_indices:
        expected = data[idx]["binary"]
        recall_prompt = f"What was the exact binary number I gave you at position {idx + 1}? Reply with ONLY the binary digits, no explanation."
        recall_messages = conversation_history + [{"role": "user", "content": recall_prompt}]
        response = query_model(recall_messages)
        time.sleep(request_delay)

        if not response:
            print(f"  [FAIL] #{idx}: No response from model", file=sys.stderr)
            continue

        model_answer = response.strip().replace(" ", "").replace("\n", "")
        is_correct = model_answer == expected
        correct += int(is_correct)
        results.append({
            "index": idx,
            "expected": expected,
            "model_response": response.strip(),
            "normalized": model_answer,
            "correct": is_correct,
        })
        status = "OK" if is_correct else "FAIL"
        print(f"  [{status}] #{idx}: expected={expected}",
              file=sys.stderr)

    score = (correct / num_recalls) * 100

    output = {
        "total_numbers": len(data),
        "num_recalls": num_recalls,
        "correct": correct,
        "score": score,
        "distraction_every": distraction_every,
        "details": results,
    }

    print(json.dumps(output, indent=2))
    print(f"\nScore: {correct}/{num_recalls} = {score:.1f}%", file=sys.stderr)


if __name__ == "__main__":
    run_benchmark()
