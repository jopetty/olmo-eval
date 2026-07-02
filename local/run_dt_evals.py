import argparse
import shlex
import subprocess
from pathlib import Path

GROUP = "jacksonp-directly-trained-sensitivity-evals"
CLUSTER = "ai2/jupiter"
PRIORITY = "urgent"
NUM_GPUS = 2
WORKSPACE = "ai2/linear-rnns"
BUDGET = "ai2/oe-other"

MODEL_ROOTS = {
    "hybrid-275M-aperiodic-sup": Path(
        "/weka/oe-training-default/ai2-llm/checkpoints/jacksonp/hybrid-aperiodic_supervised_n10000_v26_a50_m64_z1p2_s3-Cx8/275M"
    ),
    "hybrid-275M-aperiodic-unsup": Path(
        "/weka/oe-training-default/ai2-llm/checkpoints/jacksonp/hybrid-aperiodic_unsupervised_n10000_v26_a50_m64_z1p2_s2-Cx8/275M"
    ),
    "hybrid-275M-periodic-sup": Path(
        "/weka/oe-training-default/ai2-llm/checkpoints/jacksonp/hybrid-periodic_supervised_n10000_v26_a50_m64_z1p2_s5-Cx8/275M"
    ),
    "hybrid-275M-periodic-unsup": Path(
        "/weka/oe-training-default/ai2-llm/checkpoints/jacksonp/hybrid-periodic_unsupervised_n10000_v26_a50_m64_z1p2_s4-Cx8/275M"
    ),
    "hybrid-275M-r-trivial-sup": Path(
        "/weka/oe-training-default/ai2-llm/checkpoints/jacksonp/hybrid-r-trivial_supervised_n10000_v26_a50_m64_z1p2_s1-Cx8/275M"
    ),
    "hybrid-275M-r-trivial-unsup": Path(
        "/weka/oe-training-default/ai2-llm/checkpoints/jacksonp/hybrid-r-trivial_unsupervised_n10000_v26_a50_m64_z1p2_s0-Cx8/275M"
    ),
}

STEPS_STANDARD = [0, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 100000, 128000, 161186]
STEPS_DIRECT_TRAINED = [0, 2901, 3224, 6124, 6447, 12250, 12895, 24500, 25790, 49000, 51000, 51579]

CHECKPOINTS = {
    "transformer-275M": {
        step: MODEL_ROOTS["transformer-275M"] / f"step{step}/" for step in STEPS_STANDARD
    },
    "hybrid-small": {step: MODEL_ROOTS["hybrid-small"] / f"step{step}/" for step in STEPS_STANDARD},
    "hybrid-275M-aperiodic-sup": {
        step: MODEL_ROOTS["hybrid-275M-aperiodic-sup"] / f"step{step}/"
        for step in STEPS_DIRECT_TRAINED
    },
    "hybrid-275M-aperiodic-unsup": {
        step: MODEL_ROOTS["hybrid-275M-aperiodic-unsup"] / f"step{step}/"
        for step in STEPS_DIRECT_TRAINED
    },
    "hybrid-275M-periodic-sup": {
        step: MODEL_ROOTS["hybrid-275M-periodic-sup"] / f"step{step}/"
        for step in STEPS_DIRECT_TRAINED
    },
    "hybrid-275M-periodic-unsup": {
        step: MODEL_ROOTS["hybrid-275M-periodic-unsup"] / f"step{step}/"
        for step in STEPS_DIRECT_TRAINED
    },
    "hybrid-275M-r-trivial-sup": {
        step: MODEL_ROOTS["hybrid-275M-r-trivial-sup"] / f"step{step}/"
        for step in STEPS_DIRECT_TRAINED
    },
    "hybrid-275M-r-trivial-unsup": {
        step: MODEL_ROOTS["hybrid-275M-r-trivial-unsup"] / f"step{step}/"
        for step in STEPS_DIRECT_TRAINED
    },
}

OLMO_3_7B_BASE_ID = "allenai/Olmo-3-1025-7B"
TASKS = ["sensitivity"]

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
    ("provider.tokenizer", OLMO_3_7B_BASE_ID),
]

CUSTOM_CONFIG_HARNESS_OVERRIDES = [
    ("provider.trust_remote_code", "true"),
]

TRANSFORMER_275M_HARNESS_OVERRIDES = [
    ("provider.kwargs.hf_overrides", '{"architectures":["TransformersForCausalLM"]}'),
]

HYBRID_HARNESS_OVERRIDES = [
    ("provider.kind", "vllm"),
    ("provider.package", "wheel"),
    ("provider.kwargs.mamba_ssm_cache_dtype", "float32"),
    ("provider.kwargs.attention_backend", "FLASH_ATTN"),
]

HYBRID_IMAGE = "yashasbls/olmo-eval-vllm-g79d31a3f9-tch2100cu128-2026-05-23"
HYBRID_ENVS = [
    ("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1"),
]

SECRET_ENVS = [
    "JACKSONP_HF_TOKEN:HF_TOKEN",
    "jacksonp_GITHUB_TOKEN:GITHUB_TOKEN",
]


def get_model_short_name(model_path: str) -> str:
    parts = model_path.rstrip("/").split("/")
    return "_".join(parts[-3:]).lower().replace("/", "_").replace(":", "_")


def build_command(
    model_path: str,
    model_name: str,
    tasks: list[str],
    num_gpus: int = NUM_GPUS,
) -> list[str]:
    model_short = get_model_short_name(model_path)
    tasks_short = "-".join(task.replace(":", "_") for task in tasks)
    exp_name = f"{model_short}-{tasks_short}"

    cmd = ["uv", "run", "olmo-eval", "beaker", "launch"]
    cmd.extend(["-H", "default", "-n", exp_name])

    harness_overrides = list(BASE_HARNESS_OVERRIDES)
    if model_name == "transformer-275M":
        harness_overrides.extend(CUSTOM_CONFIG_HARNESS_OVERRIDES)
        harness_overrides.extend(TRANSFORMER_275M_HARNESS_OVERRIDES)
    else:
        harness_overrides.extend(HYBRID_HARNESS_OVERRIDES)

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

    if model_name != "transformer-275M":
        cmd.extend(["--image", HYBRID_IMAGE])
        for key, value in HYBRID_ENVS:
            cmd.extend(["--env", f"{key}={value}"])

    for secret_env in SECRET_ENVS:
        cmd.extend(["--secret-env", secret_env])
    cmd.extend(["--no-follow", "-y"])
    return cmd


def resolve_checkpoints(model: str, requested_checkpoints: list[int] | None) -> list[int]:
    if requested_checkpoints is not None:
        invalid = sorted(set(requested_checkpoints) - set(CHECKPOINTS[model]))
        if invalid:
            valid = ", ".join(str(step) for step in CHECKPOINTS[model])
            raise ValueError(f"Invalid checkpoint(s) for {model}: {invalid}. Valid steps: {valid}")
        return requested_checkpoints
    return list(CHECKPOINTS[model])


def build_internal_model_path(model: str, checkpoint: int) -> str:
    return str(CHECKPOINTS[model][checkpoint])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        choices=sorted(MODEL_ROOTS),
        default="transformer-275M",
        help="Directly trained 275M model to evaluate.",
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
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
