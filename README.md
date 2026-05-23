# Binary Retention Benchmark

Benchmark for LLM conversation memory. Feeds 28-bit binary numbers to a model one at a time, interleaves distraction prompts, then asks it to recall specific numbers by position index.

## How It Works

1. **Feeding** — Binary numbers are sent as individual messages. The full conversation history accumulates and is sent with every request.
2. **Distractions** — Random general-knowledge questions are interleaved to test retention under noise.
3. **Recall** — The model is asked to recall a specific number by position. It must retrieve it from the conversation history.

No cheat sheets. No summaries. The model must genuinely remember from context.

## Usage

### OpenAI-compatible API
```bash
set API_BASE=https://api.cerebras.ai/v1
set API_KEY=your_api_key
set MODEL_NAME=zai-glm-4.7
python binary_retention_benchmark.py
```

### Environment Variables
| Variable | Default | Description |
|---|---|---|
| `API_BASE` | — | API base URL |
| `API_KEY` | — | API key |
| `MODEL_NAME` | — | Model identifier |
| `NUM_PROMPTS` | `50` | How many binary numbers to send |
| `NUM_RECALLS` | `10` | How many numbers to recall |
| `DISTRACTION_EVERY` | `15` | Distraction frequency |
| `REQUEST_DELAY` | `1.0` | Seconds between requests |
| `MAX_RETRIES` | `5` | Max retries on 429 rate limits |
| `MODEL_CMD` | — | CLI command (reads prompt from stdin) |

### Standard Test Config
```
NUM_PROMPTS=50
NUM_RECALLS=10
DISTRACTION_EVERY=15
REQUEST_DELAY=1.0
```

## Results (15 prompts, 5 recalls)

All models tested via Cerebras API.

| Model | Score | Details |
|---|---|---|
| **zai-glm-4.7** | **100%** (5/5) | Perfect recall across all positions |
| gpt-oss-120b | 20% (1/5) | Only first number; verbose reasoning loops consumed all tokens on other positions |
| qwen-3-235b | 20% (1/5) | Only last number; no retention of earlier positions |

### Key Observations

- **zai-glm-4.7** is a reasoning model and significantly outperformed non-reasoning models. Chain-of-thought reasoning appears to aid memory retention.
- **gpt-oss-120b** burned all tokens on confused reasoning loops ("let me count... wait, which message?") instead of recalling numbers.
- **qwen-3-235b** defaulted to repeating the most recently seen number, showing zero retention of earlier positions.

## License

MIT
