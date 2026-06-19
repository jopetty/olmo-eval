import argparse
import shlex
import subprocess
from pathlib import Path

GROUP = "jacksonp-hybrid-small-downstream-evals"
CLUSTER = "ai2/jupiter"
PRIORITY = "urgent"
NUM_GPUS = 2
WORKSPACE = "ai2/linear-rnns"
BUDGET = "ai2/oe-other"

MODEL_ROOTS = {
    # Converted
    "hybrid": Path(
        "/weka/oe-training-default/ai2-llm/checkpoints/jacksonp/olmo3-hybrid-gdn-deux/1B"
    ),
    "transformer": Path(
        "/weka/oe-training-default/ai2-llm/checkpoints/jacksonp/olmo3-baseline-jacksonp/1B"
    ),
    "gdn": Path(
        "/weka/oe-training-default/ai2-llm/checkpoints/jacksonp/pure-gdn-ladder-gate-neg-eig/1B"
    ),
    "gdn+": Path(
        "/weka/oe-training-default/ai2-llm/checkpoints/jacksonp/pure-gdn-ladder-gate-pos-eig/1B/"
    ),
    "hybrid-small": Path(
        "/weka/oe-training-default/ai2-llm/checkpoints/jacksonp/hybrid-small-Cx100/275M/"
    ),
    "transformer-275M": Path(
        "/weka/oe-training-default/ai2-llm/checkpoints/jacksonp/transformer-small-Cx100/275M/"
    ),
}

# hybrid-gdn and baseline models use the same step numbers; others use a very slightly different
# set, advanced by ~1-4 steps/checkpoint; use latter of two (after decay)
STEPS_A = [0, 7320, 8134, 15454, 16268, 30910, 32537, 61819, 65073, 123639, 130000, 130147]
STEPS_B = [0, 7321, 8135, 15455, 16269, 30911, 32538, 61823, 65077, 123646, 130000, 130154]
STEPS_C = [0, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 100000, 128000, 161186]


CHECKPOINTS = {
    "hybrid": {step: MODEL_ROOTS["hybrid"] / f"step{step}/" for step in STEPS_A},
    "transformer": {step: MODEL_ROOTS["transformer"] / f"step{step}/" for step in STEPS_A},
    "gdn": {step: MODEL_ROOTS["gdn"] / f"step{step}/" for step in STEPS_B},
    "gdn+": {step: MODEL_ROOTS["gdn+"] / f"step{step}/" for step in STEPS_B},
    "hybrid-small": {step: MODEL_ROOTS["hybrid-small"] / f"step{step}/" for step in STEPS_C},
    "transformer-275M": {
        step: MODEL_ROOTS["transformer-275M"] / f"step{step}/" for step in STEPS_C
    },
}

OLMO_3_7B_BASE_ID = "allenai/Olmo-3-1025-7B"

TASKS = [
    # "monoid:commutative:3shot",
    # "monoid:soluble:3shot",
    # "monoid:insoluble:3shot",
    # "grapheme_en",
    "formal_langs_copy",
    "formal_langs_k_dyck",
    "formal_langs_k_shuffle_dyck",
    "formal_langs_var_unique",
    "formal_langs_var_undefined",
    "formal_langs_var_reassign_var",
    "formal_langs_var_reassign_const",
    "formal_langs:v2",
    "formal_langs:v4",
    "formal_langs:v6",
    "formal_langs_cube:v2",
    "formal_langs_cube:v4",
    "formal_langs_cube:v6",
    "formal_langs_k_dyck:k1",
    "formal_langs_k_dyck:k2",
    "formal_langs_k_dyck:k3",
    "formal_langs_k_dyck:k4",
    "formal_langs_k_dyck:k5",
    "formal_langs_k_dyck:k6",
    "formal_langs_k_shuffle_dyck:k1",
    "formal_langs_k_shuffle_dyck:k2",
    "formal_langs_k_shuffle_dyck:k3",
    "formal_langs_k_shuffle_dyck:k4",
    "formal_langs_k_shuffle_dyck:k5",
    "formal_langs_k_shuffle_dyck:k6",
    "formal_langs_reverse_copy",
    "formal_langs_sort",
    "formal_langs_char_shift_1",
    "formal_langs_char_shift_1",
    "formal_langs_char_shift_5",
]

BASE_HARNESS_OVERRIDES = [
    ("provider.num_instances", "{num_gpus}"),
    ("provider.kwargs.enforce_eager", "true"),
    ("provider.kwargs.compilation_config", '{"custom_ops":["-rms_norm"]}'),
    ("provider.add_bos_token", "false"),
    ("provider.prompt_logprobs", "1"),
    ("provider.logprob_temperature", "1.0"),
    ("provider.completion_use_prompt_token_ids", "true"),
    ("provider.completion_client_side_stop_trim", "true"),
    ("provider.completion_sentencepiece_cleanup", "true"),
    (
        "provider.dependencies",
        "[transformers @ git+https://github.com/yashassamaga/transformers.git@hybrid-small-suite]",
    ),
    ("provider.tokenizer", "allenai/Olmo-3-1025-7B"),
]

CUSTOM_CONFIG_HARNESS_OVERRIDES = [
    ("provider.trust_remote_code", "true"),
]

GDN_HARNESS_OVERRIDES = [
    # The GDN conversion script writes a PureGDN config, but the weights follow
    # the OLMo Hybrid HF layout. Use vLLM's native OLMo Hybrid implementation
    # instead of the generic Transformers backend.
    ("provider.kwargs.hf_overrides", '{"architectures":["OlmoHybridForCausalLM"]}'),
]

TRANSFORMER_275M_HARNESS_OVERRIDES = [
    # The checkpoint config advertises a custom class name, but the weights use
    # the standard OLMo 3 transformer layout that vLLM supports natively.
    ("provider.kwargs.hf_overrides", '{"architectures":["Olmo3ForCausalLM"]}'),
]

HYBRID_SMALL_HARNESS_OVERRIDES = [
    ("provider.kind", "vllm"),
    ("provider.package", "wheel"),
    ("provider.kwargs.mamba_ssm_cache_dtype", "float32"),
    ("provider.kwargs.attention_backend", "FLASH_ATTN"),
]

HYBRID_SMALL_IMAGE = "yashasbls/olmo-eval-vllm-g79d31a3f9-tch2100cu128-2026-05-23"
HYBRID_SMALL_ENVS = [
    ("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1"),
]

SECRET_ENVS = [
    "JACKSONP_HF_TOKEN:HF_TOKEN",
    "jacksonp_GITHUB_TOKEN:GITHUB_TOKEN",
]


def resolve_internal_checkpoint(model: str, checkpoint: int) -> int:
    # since there are two slightly different sets of steps, we want to allow for fuzzy indexing.
    if (
        (checkpoint in STEPS_A and checkpoint in STEPS_B)
        or (checkpoint in STEPS_A and model in ["hybrid", "transformer"])
        or (checkpoint in STEPS_B and model in ["mamba", "gdn", "gdn+"])
        or (checkpoint in STEPS_C and model in ["hybrid-small", "transformer-275M"])
    ):
        return checkpoint
    if checkpoint in STEPS_A:
        return STEPS_B[STEPS_A.index(checkpoint)]
    if checkpoint in STEPS_B:
        return STEPS_A[STEPS_B.index(checkpoint)]
    raise ValueError(f"Invalid checkpoint {checkpoint} for model {model}")


def get_model_short_name(model_path: str, revision: str | None = None) -> str:
    parts = model_path.rstrip("/").split("/")
    model_short = "_".join(parts[-3:]).lower()
    if revision is not None:
        model_short = f"{model_short}_{revision.lower()}"
    return model_short.replace("/", "_").replace(":", "_")


def build_command(
    model_path: str,
    model_name: str,
    tasks: list[str],
    num_gpus: int = NUM_GPUS,
    revision: str | None = None,
    tokenizer: str = OLMO_3_7B_BASE_ID,
) -> list[str]:

    model_short = get_model_short_name(model_path, revision)
    tasks_short = "-".join(t.replace(":", "_") for t in tasks[:2])
    if len(tasks) > 2:
        tasks_short += f"-and-{len(tasks) - 2}-more"
    exp_name = f"{model_short}-{tasks_short}"

    cmd = ["uv", "run", "olmo-eval", "beaker", "launch"]
    cmd.extend(["-H", "default", "-n", exp_name])

    harness_overrides = [
        (key, tokenizer if key == "provider.tokenizer" else value)
        for key, value in BASE_HARNESS_OVERRIDES
    ]
    if revision is not None:
        harness_overrides.append(("provider.revision", revision))
    if model_name in ["gdn", "gdn+", "transformer-275M"]:
        harness_overrides.extend(CUSTOM_CONFIG_HARNESS_OVERRIDES)
    if model_name in ["gdn", "gdn+"]:
        harness_overrides.extend(GDN_HARNESS_OVERRIDES)
    if model_name == "transformer-275M":
        harness_overrides.extend(TRANSFORMER_275M_HARNESS_OVERRIDES)
    if model_name == "hybrid-small":
        harness_overrides.extend(HYBRID_SMALL_HARNESS_OVERRIDES)

    for key, value in harness_overrides:
        cmd.extend(["-o", f"{key}={value.replace('{num_gpus}', str(num_gpus))}"])

    cmd.extend(["-m", model_path])
    for task in tasks:
        cmd.extend(["-t", task])

    cmd.extend(
        [
            "--gpus",
            str(num_gpus),
            "--priority",
            PRIORITY,
            "--group",
            GROUP,
            "--cluster",
            CLUSTER,
            "--workspace",
            WORKSPACE,
            "--budget",
            BUDGET,
            "--inspect",
        ]
    )
    if model_name == "hybrid-small":
        cmd.extend(["--image", HYBRID_SMALL_IMAGE])
        for key, value in HYBRID_SMALL_ENVS:
            cmd.extend(["--env", f"{key}={value}"])
    for secret_env in SECRET_ENVS:
        cmd.extend(["--secret-env", secret_env])
    cmd.extend(["--no-follow", "-y"])
    return cmd


def resolve_checkpoints(model: str, requested_checkpoints: list[int] | None) -> list[int]:
    if requested_checkpoints is not None:
        return requested_checkpoints
    return list(CHECKPOINTS[model].keys())


def build_internal_model_path(model: str, checkpoint: int) -> str:
    resolved_ckpt = resolve_internal_checkpoint(model, checkpoint)
    return str(CHECKPOINTS[model][resolved_ckpt])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        choices=["transformer", "transformer-275M", "hybrid", "hybrid-small", "gdn", "gdn+"],
        default="transformer",
        help="Model architecture to use.",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=NUM_GPUS,
        help="Number of GPUs per job.",
    )
    parser.add_argument(
        "--checkpoints",
        "-c",
        nargs="+",
        type=int,
        help="Checkpoint steps to use. Defaults to all checkpoints for the selected model.",
    )
    parser.add_argument(
        "--hf-model",
        type=str,
        help="Hugging Face model ID to evaluate instead of an internal Weka checkpoint.",
    )
    parser.add_argument(
        "--hf-revision",
        type=str,
        help="Hugging Face revision, branch, tag, or checkpoint to use with --hf-model.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.hf_revision is not None and args.hf_model is None:
        parser.error("--hf-revision requires --hf-model")
    if args.hf_model is not None and args.checkpoints is not None:
        parser.error("--checkpoints can only be used with internal Weka models")
    return args


def main() -> None:
    args = parse_args()

    if args.hf_model is not None:
        commands = [
            build_command(
                args.hf_model,
                "hf",
                TASKS,
                args.gpus,
                revision=args.hf_revision,
                tokenizer=args.hf_model,
            )
        ]
    else:
        checkpoints = resolve_checkpoints(args.model, args.checkpoints)
        commands = [
            build_command(
                build_internal_model_path(args.model, checkpoint),
                args.model,
                TASKS,
                args.gpus,
            )
            for checkpoint in checkpoints
        ]

    for cmd in commands:
        if args.dry_run:
            print(shlex.join(cmd))
        else:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
