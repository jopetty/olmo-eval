# Verifying SciCode against Artificial Analysis

## Goal

Reproduce Artificial Analysis's reported **41% sub_step_accuracy** on SciCode for
`google/gemma-4-31b-it`. Our initial run with default olmo-eval settings yielded
**12.5%**. This doc captures the methodology gaps and the parameter set needed
to close them.

## Reference

- AA methodology: https://artificialanalysis.ai/methodology/intelligence-benchmarking
- AA SciCode leaderboard: https://artificialanalysis.ai/evaluations/scicode
- SciCode benchmark: https://scicode-bench.github.io/

## Architecture

SciCode is implemented as an **ExternalEval** under
`src/olmo_eval/evals/external/benchmarks/scicode/`. Each problem runs as a
sequential per-sub-step cascade against the main vLLM provider (no auxiliary
provider, no second GPU pool). Verification runs in a per-problem Python
sandbox with `numpy`, `scipy`, `sympy`, and `h5py` installed and the reference
`test_data.h5` volume-mounted from weka.

## Task

- **Benchmark**: SciCode (test split, 65 problems, 288 sub-steps)
- **Prompt variant**: `with_background=true` — includes per-sub-step domain
  background in the prompt. This inflates input tokens to match AA.
- **Scorer**: `sub_step_accuracy` (fraction of sub-steps whose generated code
  passes the hidden unit tests in a sandbox).

## Parameters to match AA

| Parameter | Value | Where set |
|---|---|---|
| `with_background` | `true` | `-A with_background=true` |
| `max_tokens` | 16384 | `-A max_tokens=16384` |
| Temperature | 0.6 | `-A temperature=0.6` |
| Reasoning | on | `-K chat_template_kwargs={"enable_thinking":true}` |
| Sandbox `command_timeout` | 600s | `SciCodeArgs.command_timeout` default |

`enable_thinking` is the canonical kwarg for Gemma 4's chat template (verified
against `google/gemma-4-31b-it/chat_template.jinja` on HuggingFace).

## Reference token usage

AA's published totals for `gemma-4-31b-it` on SciCode (~1.8M tokens per run):

| Bucket | AA |
|---|---|
| Input tokens | 760,000 |
| Output tokens | 150,000 |
| Reasoning tokens | 880,000 |
| **Total** | **~1.79M** |

## Launch command

```
uv run olmo-eval beaker launch \
  -E scicode \
  -I finbarrt/olmo-eval-cu1281-trc290-amd64-sandbox-vllm \
  -K tensor_parallel_size=2 \
  -K 'chat_template_kwargs={"enable_thinking":true}' \
  -A with_background=true \
  -A max_tokens=16384 \
  -A temperature=0.6 \
  -m google/gemma-4-31b-it \
  -p urgent -w ai2/open-instruct-dev \
  -c h100 -B ai2/oe-adapt -G 2 \
  --no-follow -y
```

The `-vllm` suffix in the image name tells the launcher to use the pre-baked
`/opt/vllm-venv` instead of building one at runtime (~8–10 min savings per
launch).

## Verification

1. After the run, read `main/metrics/vllm_server_*.jsonl` and confirm
   `total_prompt_tokens ≈ 760k` and `total_completion_tokens ≈ 1M`
   (answers + reasoning).
2. `sub_step_accuracy` should land near **0.41**.
