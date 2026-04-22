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

## Task

- **Benchmark**: SciCode (test split, 65 problems, 288 sub-steps)
- **Task variant**: `scicode_with_background` — includes per-sub-step domain
  background in the prompt. This is what inflates input tokens to match AA.
- **Scorer**: `sub_step_accuracy` (fraction of sub-steps whose generated code
  passes the hidden unit tests in a Podman sandbox).

## Parameters to match AA

| Parameter | Ours (default) | AA | Where set |
|---|---|---|---|
| Task variant | `scicode` | with background | `-t scicode_with_background` |
| `max_tokens` | 4096 | 16384 | `scicode.py:430` |
| Temperature | 0.0 | 0.6 (reasoning) | `scicode.py:430` |
| Reasoning | off | on | `chat_template_kwargs={"enable_thinking": true}` |
| Scorer `timeout` | 300s | — | `scicode.py:329` (bumped to 600s) |
| Sandbox `command_timeout` | 300s | — | `presets.py:218` (bumped to 600s) |

`enable_thinking` is the canonical kwarg for Gemma 4's chat template (verified
against `google/gemma-4-31b-it/chat_template.jinja` on HuggingFace). It is
plumbed end-to-end:

- Per-request via `extra_body["chat_template_kwargs"]` in
  `src/olmo_eval/inference/providers/vllm_server.py:509-510`.
- At provider init in `src/olmo_eval/inference/providers/vllm_server.py:206`.

## Reference token usage

AA's published totals for `gemma-4-31b-it` on SciCode (~1.8M tokens per run):

| Bucket | AA | Ours (pre-fix) | Factor |
|---|---|---|---|
| Input tokens | 760,000 | 102,801 | 7.4× fewer (no per-step background) |
| Output tokens | 150,000 | 186,357 | slightly more |
| Reasoning tokens | 880,000 | 0 | reasoning mode off |
| **Total** | **~1.79M** | **~289k** | — |

The 7.4× input-token gap comes entirely from the plain `scicode` variant
omitting the per-sub-step background paragraph. The 880k reasoning-token gap
comes from not passing `enable_thinking=true`. Fixing both should bring our
totals into AA's ballpark.

Timeouts also mattered: 3 of 288 sub-steps hit the 300s sandbox limit and were
scored as failures. Bumping both scorer `timeout` and `command_timeout` to 600s
removes this source of spurious wrong-answers.

## Launch command

```
uv run olmo-eval beaker launch \
  -H scicode \
  -I finbarrt/olmo-eval-cu1281-trc290-amd64-sandbox-vllm \
  -o provider.kwargs.tensor_parallel_size=2 \
  -o 'provider.kwargs.chat_template_kwargs={"enable_thinking":true}' \
  -t scicode_with_background \
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
