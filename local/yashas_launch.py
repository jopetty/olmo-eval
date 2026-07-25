import argparse
import functools
import json
import os
import subprocess
import time

GROUP = "yashasbls-hybrid-small-evals-v4"
CLUSTER = "ai2/jupiter"
PRIORITY = "urgent"
NUM_GPUS = 1
WORKSPACE = "ai2/linear-rnns"
BUDGET = "ai2/oe-other"

CKPT_BASE = "/weka/oe-training-default/ai2-llm/checkpoints/yashasbls"

# The olmo_hybrid_small architecture is served via the transformers/vLLM plugins
# (stock transformers/vLLM, no fork). The eval job pip-installs them from this
# repo's GitHub remote at the branch/commit this launcher runs from, so they must
# be committed and pushed.
PLUGINS_REPO = "git+https://github.com/YashasSamaga/hybrid-small-suite.git"
# Ref used for the plugin git installs when --use-latest is passed (or when the
# launcher runs outside a checkout of this repo and can't resolve a git ref).
DEFAULT_PLUGINS_REF = "main"


@functools.lru_cache(maxsize=1)
def _current_git_ref() -> str:
    """Git ref (branch or commit) the eval job installs the plugins from.

    Resolves to the branch this launcher runs from so eval jobs use the plugins
    at the same ref; falls back to the commit SHA on a detached HEAD. The ref
    must be pushed to the remote, since the eval job pip-installs the plugins
    from GitHub.
    """
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    branch = subprocess.run(
        ["git", "-C", repo_dir, "rev-parse", "--abbrev-ref", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if branch != "HEAD":
        return branch
    return subprocess.run(
        ["git", "-C", repo_dir, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _plugin_dependencies_override(plugins_ref: str | None = None) -> str:
    """``provider.dependencies`` override installing both ladder plugins.

    Installs the vLLM and transformers plugins (git subdirectory installs) at
    ``plugins_ref`` so the eval job serves the olmo_hybrid_small architecture on
    stock vLLM/transformers. olmo-eval v1.0.0 JSON-decodes the value, so it must
    be a JSON list of strings.
    """
    ref = plugins_ref or _current_git_ref()
    deps = [
        f"vllm-plugin @ {PLUGINS_REPO}@{ref}#subdirectory=plugins/vllm_plugin",
        f"transformers-plugin @ {PLUGINS_REPO}@{ref}#subdirectory=plugins/transformers_plugin",
    ]
    return f"provider.dependencies={json.dumps(deps)}"


# Each stage maps size -> list of HF checkpoint paths.
# Single-model stages have one-element lists; grid searches have multiple.
all_stages: dict[str, dict[str, list[str]]] = {
    "pretraining": {
        "275m": [f"{CKPT_BASE}/hybrid-small-275M-Cx100/step161186-hf/"],
        "275m-transformer": [f"{CKPT_BASE}/hybrid-small-transformer-275M/step161186-hf/"],
        "275m-gdn": [f"{CKPT_BASE}/hybrid-small-gdn-275M/step161186-hf/"],
        "450m": [f"{CKPT_BASE}/hybrid-small-450m-cx100-lr8e-3/step179814-hf/"],
        "810m": [f"{CKPT_BASE}/hybrid-small-810M-Cx100/step269926-hf/"],
        "1.4b": [f"{CKPT_BASE}/hybrid-small-1.4B-Cx100/step308433-hf/"],
        "2.7b": [f"{CKPT_BASE}/hybrid-small-2.7B-Cx100-v2/step604266-hf/"],
    },
    "midtraining": {
        "275m": [f"{CKPT_BASE}/hybrid-small-midtraining-275M-v2-lr1.6e-3/step38147-hf/"],
        "275m-transformer": [f"{CKPT_BASE}/hybrid-small-transformer-275M-midtraining/step38147-hf/"],
        "275m-gdn": [f"{CKPT_BASE}/hybrid-small-gdn-275M-midtraining/step38147-hf/"],
        "450m": [f"{CKPT_BASE}/hybrid-small-midtraining-450m/step38147-hf/"],
        "810m": [f"{CKPT_BASE}/hybrid-small-midtraining-v2-810M-lr4e-4/step23842-hf/"],
        "1.4b": [f"{CKPT_BASE}/hybrid-small-midtraining-v2-1.4b-lr4e-4/step11921-hf/"],
        "2.7b": [f"{CKPT_BASE}/hybrid-small-midtraining-2.7B/step11921-hf/"],
    },
    "long_context": {
        "275m": [f"{CKPT_BASE}/hybrid-small-long-context-v2-275m/step47684-hf/"],
        "275m-transformer": [f"{CKPT_BASE}/hybrid-small-transformer-275M-lc/step47684-hf/"],
        "275m-gdn": [f"{CKPT_BASE}/hybrid-small-gdn-275M-lc/step47684-hf/"],
        "450m": [f"{CKPT_BASE}/hybrid-small-lc-v2-450m/step47684-hf/"],
        "810m": [f"{CKPT_BASE}/hybrid-small-long-context-v2-810m/step23842-hf/"],
        "1.4b": [f"{CKPT_BASE}/hybrid-small-long-context-v2-1.4b/step23842-hf/"],
        "2.7b": [f"{CKPT_BASE}/hybrid-small-lc-2.7B/step23842-hf/"],
    },
}

# External baseline models (HF base checkpoints) to compare against.
# Launch with: --model <hf_id> (e.g. --model Qwen/Qwen3.5-2B-Base).
# ctx = native context length; arch notes matter for serving.
#   - qwen3_5 models are hybrid (Gated DeltaNet linear-attention + periodic Gated
#     Attention, dense FFN — not MoE at these sizes) with a vision encoder;
#     served text-only via provider.kwargs.language_model_only=true (harmless no-op
#     on text-only models, so it is now set globally in build_command).
# status: "ok" = RULER 4096 smoke test passed through our stack.
BASELINES: dict[str, dict[str, str | int]] = {
    # Qwen3 dense transformers (32k ctx, Apr 2025)
    "qwen3-0.6b": {"hf": "Qwen/Qwen3-0.6B-Base", "ctx": 32768, "status": "ok"},
    "qwen3-1.7b": {"hf": "Qwen/Qwen3-1.7B-Base", "ctx": 32768, "status": "ok"},
    # Qwen3.5 hybrid (Gated DeltaNet + Gated Attention), dense FFN (256k native, Feb 2026)
    "qwen3.5-0.8b": {"hf": "Qwen/Qwen3.5-0.8B-Base", "ctx": 262144, "status": "untested"},
    "qwen3.5-2b": {"hf": "Qwen/Qwen3.5-2B-Base", "ctx": 262144, "status": "ok"},
    # Llama 3.2 dense (128k ctx, Sep 2024)
    "llama3.2-1b": {"hf": "meta-llama/Llama-3.2-1B", "ctx": 131072, "status": "ok"},
    "llama3.2-3b": {"hf": "meta-llama/Llama-3.2-3B", "ctx": 131072, "status": "untested"},
    # Gemma 3 dense (32k ctx, 2025)
    "gemma3-270m": {"hf": "google/gemma-3-270m", "ctx": 32768, "status": "ok"},
    "gemma3-1b": {"hf": "google/gemma-3-1b-pt", "ctx": 32768, "status": "untested"},
    # OLMo 2 dense (4k ctx, Apr 2025) — fully-open; weak/short-ctx, downstream baseline
    "olmo2-1b": {"hf": "allenai/OLMo-2-0425-1B", "ctx": 4096, "status": "untested"},
    # Pythia dense (2k ctx, Apr 2023) — fully-open scaling suite; weak/short-ctx baseline
    "pythia-1b": {"hf": "EleutherAI/pythia-1b", "ctx": 2048, "status": "untested"},
    "pythia-1.4b": {"hf": "EleutherAI/pythia-1.4b", "ctx": 2048, "status": "untested"},
    # SmolLM2 dense llama (8k ctx, Feb 2025) — fully-open; downstream-only baseline
    "smollm2-1.7b": {"hf": "HuggingFaceTB/SmolLM2-1.7B", "ctx": 8192, "status": "untested"},
    # SmolLM3 dense GQA+NoPE (64k native, 128k YaRN; 2025) — fully-open; long-ctx baseline
    "smollm3-3b": {"hf": "HuggingFaceTB/SmolLM3-3B-Base", "ctx": 65536, "status": "untested"},
    # Qwen2.5 dense transformers (32k native ctx, Sep 2024) — prior-gen Qwen trendline
    "qwen2.5-0.5b": {"hf": "Qwen/Qwen2.5-0.5B", "ctx": 32768, "status": "untested"},
    "qwen2.5-1.5b": {"hf": "Qwen/Qwen2.5-1.5B", "ctx": 32768, "status": "untested"},
    # Qwen2.5-3B is qwen-research license (NOT Apache like 0.5B/1.5B)
    "qwen2.5-3b": {"hf": "Qwen/Qwen2.5-3B", "ctx": 32768, "status": "untested"},
    # LFM2.5 hybrid (short-conv + GQA) base checkpoints (32k ctx, Jun 2026) — on-device baseline
    "lfm2.5-230m": {"hf": "LiquidAI/LFM2.5-230M-Base", "ctx": 32768, "status": "untested"},
    "lfm2.5-350m": {"hf": "LiquidAI/LFM2.5-350M-Base", "ctx": 32768, "status": "untested"},
}

# Eval suites. Each suite is launched as ONE Beaker job that runs all of its
# tasks together, sharing a single vLLM allocation (repeated -t), instead of one
# job per task. This mirrors the scaling-ladders mainline launcher and keeps the
# job count to (models x selected suites). Tasks are split into suites by GPU
# need so each suite's job requests max(gpus) across its tasks.
OLMOBASE_MATH = [
    ("olmobase:math", 8),
]

OLMOBASE_CODE = [
    ("olmobase:code", 8),
    ("olmobase:code_fim", 8),
]

OLMOBASE_GEN_MC = [
    ("olmobase:gen", 4),
    ("olmobase:mcqa_non_stem", 4),
    ("olmobase:mcqa_stem", 4),
]

OLMOBASE_EASY = [
    ("c4_10k:ppl", 2),
    ("olmobase:easy:qa:rc", 2),
    ("olmobase:easy:qa:bpb", 2),
    ("olmobase:easy:math:bpb", 2),
    ("olmobase:easy:code:bpb", 2),
    ("code_fresh:bpb", 2),
]

OLMOBASE_LC_TASKS = [
    ("ruler_all__4096", 1),
    ("ruler_all__8192", 1),
    ("ruler_all__16384", 1),
    ("ruler_all__32768", 1),
    ("ruler_all__65536", 1),
    ("ruler_all__131072", 1),
]

# Suite name -> task list; the key is what --eval-type selects.
SUITES: dict[str, list[tuple[str, int]]] = {
    # Accuracy suites (headline reported numbers).
    "math": OLMOBASE_MATH,
    "code": OLMOBASE_CODE,
    "gen_mc": OLMOBASE_GEN_MC,
    # Cheap likelihood/BPB/perplexity signals (no sandbox).
    "easy": OLMOBASE_EASY,
    # Long context.
    "lc": OLMOBASE_LC_TASKS,
}

# Default olmo-eval harness preset (vLLM server, no sandbox).
DEFAULT_HARNESS = "default"

# Suites whose scorers execute generated code (pass@1 via CodeExecutionScorer /
# MultiplEScorer) require a sandboxed harness. codex_universal ships the
# sandbox-fusion + bigcodebench Docker sandboxes; launching under it makes the
# CLI switch to the Podman-enabled Beaker image automatically.
SUITE_HARNESS: dict[str, str] = {
    "code": "codex_universal",
}


def harness_for_suite(suite: str) -> str:
    """Harness preset to launch a suite under (sandbox for code-exec suites)."""
    return SUITE_HARNESS.get(suite, DEFAULT_HARNESS)


LC_TASK_NAMES: set[str] = {name for name, _ in OLMOBASE_LC_TASKS}


def resolve_eval_groups(suite_names: list[str]) -> list[tuple[str, list[str], int]]:
    """Return one ``(suite, task_names, gpus)`` run group per named suite.

    Each group becomes a single olmo-eval run (one Beaker job): all of its tasks
    are evaluated together at one GPU count (the max requested within the suite).
    """
    groups: list[tuple[str, list[str], int]] = []
    for suite in suite_names:
        suite_tasks = SUITES[suite]
        if not suite_tasks:
            continue
        names = [name for name, _ in suite_tasks]
        gpus = max(req for _, req in suite_tasks)
        groups.append((suite, names, gpus))
    return groups


def ruler_length(task_name: str) -> int:
    """Extract the sequence length from a RULER task name (ruler_all__4096 -> 4096)."""
    return int(task_name.rsplit("__", 1)[1])


def lc_tasks_for_ctx(ctx_tokens: int) -> list[tuple[str, int]]:
    """Return the RULER tasks whose length fits within the model's native context."""
    return [(t, g) for (t, g) in OLMOBASE_LC_TASKS if ruler_length(t) <= ctx_tokens]


def experiment_name(model_path: str, tasks: list[str]) -> str:
    """Deterministic Beaker experiment name for a (model, tasks) eval job.

    Must stay in sync with the ``-n`` value used in build_command so resume/skip
    logic can match experiments that were already launched.
    """
    # Last two path components joined with underscore, slashes removed.
    parts = model_path.rstrip("/").split("/")
    model_short = "_".join(parts[-2:]).lower()
    tasks_short = "-".join(t.replace(":", "_") for t in tasks[:2])
    if len(tasks) > 2:
        tasks_short += f"-and-{len(tasks) - 2}-more"
    return f"{model_short}-{tasks_short}"


def successful_experiment_names(group: str) -> set[str]:
    """Names of experiments in the Beaker group whose latest run SUCCEEDED.

    Used by the ``--skip-existing`` resume logic: only successfully-completed
    experiments are skipped, so jobs whose last run failed, was canceled, or is
    still in progress get re-launched. The ``status`` reported by
    ``beaker group info`` is the workload's latest-run status, so this already
    reflects "last run only".

    Returns an empty set if the group does not exist yet (nothing launched). The
    group name is auto-qualified with the Beaker username when unqualified.
    """
    if "/" not in group:
        try:
            from beaker import Beaker

            group = f"{Beaker.from_env().user_name}/{group}"
        except Exception:
            pass
    out = subprocess.run(
        ["uv", "run", "olmo-eval", "beaker", "group", "info", group, "-f", "json"],
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        if "not found" in (out.stdout + out.stderr).lower():
            return set()
        raise RuntimeError(f"failed to query group {group}: {out.stderr.strip()}")
    text = out.stdout.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Be tolerant of any leading/trailing noise around the JSON payload.
        start, end = text.find("{"), text.rfind("}")
        data = json.loads(text[start : end + 1])
    # BeakerWorkloadStatus.name for a completed run is "succeeded" (lower-case).
    return {e["name"] for e in data.get("experiments", []) if e.get("status") == "succeeded"}


def build_command(
    model_path: str,
    tasks: list[str],
    num_gpus: int = NUM_GPUS,
    is_lc: bool = False,
    group: str = GROUP,
    max_model_len: int = 131072,
    plugins_ref: str | None = None,
    harness: str = DEFAULT_HARNESS,
) -> list[str]:
    exp_name = experiment_name(model_path, tasks)

    cmd = ["uv", "run", "olmo-eval", "beaker", "launch"]
    cmd += ["-H", harness]
    cmd += ["-n", exp_name]
    cmd += ["-o", f"provider.num_instances={num_gpus}"]
    cmd += ["-o", "provider.kwargs.enforce_eager=true"]
    cmd += ["-o", "provider.kwargs.mamba_ssm_cache_dtype=float32"]
    # Serve vision-language baselines (e.g. qwen3_5) as text-only. Harmless no-op
    # on text-only models, verified via Qwen3-0.6B RULER smoke test.
    cmd += ["-o", "provider.kwargs.language_model_only=true"]
    cmd += ["-o", "provider.add_bos_token=false"]
    cmd += ["-o", "provider.kind=vllm"]
    cmd += ["-o", "provider.package=wheel"]
    cmd += ["-o", _plugin_dependencies_override(plugins_ref)]
    cmd += ["-o", "provider.kwargs.attention_backend=FLASH_ATTN"]
    if is_lc:
        cmd += ["-o", f"provider.max_model_len={max_model_len}"]
    cmd += ["-m", model_path]
    for task in tasks:
        cmd += ["-t", task]
    cmd += ["--gpus", str(num_gpus)]
    cmd += ["--priority", PRIORITY]
    cmd += ["--group", group]
    cmd += ["--cluster", CLUSTER]
    cmd += ["--workspace", WORKSPACE]
    cmd += ["--budget", BUDGET]
    cmd += ["--inspect"]
    cmd += ["--secret-env", "yashasbls_HF_TOKEN:HF_TOKEN"]
    # Auth for cloning the private plugins repo during the provider.dependencies
    # install. Gantry's built-in git-credential setup reads the user-scoped
    # secret ``yashasbls_GITHUB_TOKEN``; map it to GITHUB_TOKEN here too.
    cmd += ["--secret-env", "yashasbls_GITHUB_TOKEN:GITHUB_TOKEN"]
    cmd += ["--env", "VLLM_ALLOW_LONG_MAX_MODEL_LEN=1"]
    cmd += ["--no-follow"]
    cmd += ["-y"]
    return cmd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sizes",
        nargs="+",
        choices=["275m", "275m-transformer", "275m-gdn", "450m", "810m", "1.4b", "2.7b"],
        default=["275m"],
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=list(all_stages.keys()),
        default=["pretraining"],
    )
    parser.add_argument(
        "--eval-type",
        nargs="+",
        choices=list(SUITES.keys()),
        default=["gen_mc"],
    )
    parser.add_argument(
        "--group", type=str, default=GROUP, help="Beaker workgroup for launched jobs."
    )
    parser.add_argument("--gpus", type=int, default=NUM_GPUS, help="Number of GPUs per job.")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Custom model path or HF name (overrides --sizes/--stages)",
    )
    parser.add_argument(
        "--baselines",
        nargs="+",
        choices=list(BASELINES.keys()),
        default=None,
        help="External baseline keys to eval; lc tasks are filtered to each model's native ctx.",
    )
    parser.add_argument(
        "--lc-lengths",
        nargs="+",
        type=int,
        default=None,
        help="Restrict lc (RULER) tasks to these sequence lengths (e.g. 4096). Default: all.",
    )
    parser.add_argument(
        "--use-latest",
        action="store_true",
        help=f"Install the plugins from '{DEFAULT_PLUGINS_REF}' instead of resolving the "
        "current git branch. Use this to run the launcher outside a checkout of this repo.",
    )
    parser.add_argument(
        "--plugins-ref",
        type=str,
        default=None,
        help="Git ref (branch/tag/commit) to install the vllm/transformers plugins from. "
        "Overrides --use-latest and the auto-resolved current branch.",
    )
    parser.add_argument(
        "--delay", type=int, default=0, help="Seconds to wait between launching each eval job"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    num_gpus = args.gpus
    group = args.group

    # Resolve the ref the eval job installs the plugins from: explicit --plugins-ref
    # wins, then --use-latest, else the current branch (requires a git checkout).
    if args.plugins_ref:
        plugins_ref = args.plugins_ref
    elif args.use_latest:
        plugins_ref = DEFAULT_PLUGINS_REF
    else:
        plugins_ref = _current_git_ref()

    # One run group per selected suite; each group is a single job running all
    # of the suite's tasks together (fewer jobs than one-per-task).
    eval_groups = resolve_eval_groups(args.eval_type)

    lc_lengths = set(args.lc_lengths) if args.lc_lengths else None

    def filter_lc_names(names: list[str]) -> list[str]:
        """Drop lc (RULER) task names whose length isn't in --lc-lengths (if set)."""
        if lc_lengths is None:
            return names
        return [n for n in names if n not in LC_TASK_NAMES or ruler_length(n) in lc_lengths]

    launched = 0

    def launch_group(
        model_path, names, gpus, *, label, max_model_len=131072, harness=DEFAULT_HARNESS
    ):
        """Launch one Beaker job running all of `names` together for `model_path`."""
        nonlocal launched
        names = filter_lc_names(names)
        if not names:
            print(f"[skip] {label}: no tasks after filtering")
            return
        # Explicit --gpus overrides the suite's requested count.
        job_gpus = num_gpus if num_gpus != NUM_GPUS else gpus
        is_lc = any(n in LC_TASK_NAMES for n in names)
        cmd = build_command(
            model_path,
            names,
            job_gpus,
            is_lc=is_lc,
            group=group,
            max_model_len=max_model_len,
            plugins_ref=plugins_ref,
            harness=harness,
        )
        mml = max_model_len if is_lc else "-"
        print(
            f"\n=== {label} | {len(names)} task(s) | {job_gpus} GPUs "
            f"| mml={mml} | harness={harness} ==="
        )
        print(" ".join(cmd))
        if not args.dry_run:
            if launched > 0 and args.delay > 0:
                time.sleep(args.delay)
            subprocess.run(cmd, check=True)
        launched += 1

    if args.model:
        # Custom model: run each selected suite as one job.
        for suite, names, gpus in eval_groups:
            launch_group(
                args.model, names, gpus, label=f"custom/{suite}", harness=harness_for_suite(suite)
            )
    elif args.baselines:
        # Baseline models: ctx-aware. lc tasks are capped to each model's native
        # context, and max_model_len is set to that context rather than 131072.
        for key in args.baselines:
            spec = BASELINES[key]
            hf = spec["hf"]
            ctx_tokens = spec["ctx"]
            for suite, names, gpus in eval_groups:
                if suite == "lc":
                    lc_names = [n for n, _ in lc_tasks_for_ctx(ctx_tokens)]
                    if not lc_names:
                        print(f"[skip] {key}/lc: ctx {ctx_tokens} below smallest RULER length")
                        continue
                    launch_group(
                        hf,
                        lc_names,
                        gpus,
                        label=f"baseline {key} ({hf})/lc",
                        max_model_len=ctx_tokens,
                    )
                else:
                    launch_group(
                        hf,
                        names,
                        gpus,
                        label=f"baseline {key} ({hf})/{suite}",
                        harness=harness_for_suite(suite),
                    )
    else:
        for stage in args.stages:
            checkpoints = all_stages[stage]
            for size in args.sizes:
                if size not in checkpoints:
                    print(f"[skip] {stage} has no size {size}")
                    continue
                for model_path in checkpoints[size]:
                    short_name = model_path.rstrip("/").split("/")[-1]
                    for suite, names, gpus in eval_groups:
                        launch_group(
                            model_path,
                            names,
                            gpus,
                            label=f"{stage}/{size}/{short_name}/{suite}",
                            harness=harness_for_suite(suite),
                        )

    print(f"\nLaunched {launched} job(s).")


if __name__ == "__main__":
    main()
