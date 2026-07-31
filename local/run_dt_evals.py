import argparse
import shlex
import subprocess
from pathlib import Path

GROUP = "jacksonp-directly-trained-synthetic-evals-2"
CLUSTER = "ai2/saturn"
PRIORITY = "urgent"
NUM_GPUS = 2
WORKSPACE = "ai2/beyond-state"
BUDGET = "ai2/oe-other"

CHECKPOINT_BASE = Path(
    "/weka/oe-training-default/ai2-llm/model-ladders/synthetic-ladder/60M"
)
MODEL_TYPES = ("hybrid", "transformer")
DATASET_TYPES = ("r-trivial", "aperiodic", "periodic")
SEEDS = (0, 7, 42)

DATASET_DIRS = {
    "r-trivial": "r-trivial_0supervision_n200000000_v26_a50_m64_z1p2_assignments_lt5",
    "aperiodic": "aperiodic_0supervision_n200000000_v26_a50_m64_z1p2_assignments_lt10",
    "periodic": "periodic_0supervision_n200000000_v26_a50_m64_z1p2_assignments_lt10",
}

CHECKPOINT_STEPS = {
    "hybrid": (0, 2034, 2260, 4294, 4521, 8589, 9000, 9041),
    "transformer": (0, 1971, 2190, 4161, 4381, 8000, 8323, 8762),
}

MODEL_ROOTS = {
    (model_type, dataset_type, seed): CHECKPOINT_BASE
    / model_type
    / DATASET_DIRS[dataset_type]
    / "Cx2"
    / f"init_seed{seed}"
    for model_type in MODEL_TYPES
    for dataset_type in DATASET_TYPES
    for seed in SEEDS
}

CHECKPOINTS = {
    model: {
        step: model_root / f"step{step}-hf"
        for step in CHECKPOINT_STEPS[model[0]]
    }
    for model, model_root in MODEL_ROOTS.items()
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

TRANSFORMER_HARNESS_OVERRIDES = [
    *BASE_HARNESS_OVERRIDES,
    ("provider.trust_remote_code", "true"),
    ("provider.kwargs.hf_overrides", '{"architectures":["TransformersForCausalLM"]}'),
]

HYBRID_HARNESS_OVERRIDES = [
    ("provider.num_instances", "{num_gpus}"),
    ("provider.kind", "hf"),
    ("provider.trust_remote_code", "true"),
    ("provider.dtype", "bfloat16"),
    (
        "provider.dependencies",
        "[transformers @ git+https://github.com/yashassamaga/transformers.git@hybrid-small-suite]",
    ),
    ("provider.tokenizer", OLMO_3_7B_BASE_ID),
]

HYBRID_IMAGE = "yashasbls/olmo-eval-vllm-g79d31a3f9-tch2100cu128-2026-05-23"

SECRET_ENVS = [
    "JACKSONP_HF_TOKEN:HF_TOKEN",
    "jacksonp_GITHUB_TOKEN:GITHUB_TOKEN",
]


def get_model_short_name(model_path: str) -> str:
    parts = model_path.rstrip("/").split("/")
    return "_".join(parts[-5:]).lower().replace("/", "_").replace(":", "_")


def build_command(
    model_path: str,
    model_type: str,
    tasks: list[str],
    num_gpus: int = NUM_GPUS,
    cluster: str = CLUSTER,
) -> list[str]:
    model_short = get_model_short_name(model_path)
    tasks_short = "-".join(task.replace(":", "_") for task in tasks)
    exp_name = f"{model_short}-{tasks_short}"

    cmd = ["uv", "run", "olmo-eval", "beaker", "launch"]
    cmd.extend(["-H", "default", "-n", exp_name])

    if model_type == "hybrid":
        harness_overrides = HYBRID_HARNESS_OVERRIDES
    else:
        harness_overrides = TRANSFORMER_HARNESS_OVERRIDES

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
            cluster,
            "--workspace",
            WORKSPACE,
            "--budget",
            BUDGET,
            "--inspect",
        ]
    )

    if model_type == "hybrid":
        cmd.extend(["--image", HYBRID_IMAGE])

    for secret_env in SECRET_ENVS:
        cmd.extend(["--secret-env", secret_env])
    cmd.extend(["--no-follow", "-y"])
    return cmd


def resolve_checkpoints(
    model_type: str,
    dataset_type: str,
    seed: int,
    requested_checkpoints: list[int] | None,
    final: bool = False,
) -> list[int]:
    model = (model_type, dataset_type, seed)
    if model not in CHECKPOINTS:
        available = ", ".join(
            f"{available_model}/{dataset}/seed{available_seed}"
            for available_model, dataset, available_seed in MODEL_ROOTS
        )
        raise ValueError(
            f"No checkpoint is configured for {model_type}/{dataset_type}/seed{seed}."
            f" Available combinations: {available}"
        )
    if final:
        return [max(CHECKPOINTS[model])]
    if requested_checkpoints is not None:
        invalid = sorted(set(requested_checkpoints) - set(CHECKPOINTS[model]))
        if invalid:
            valid = ", ".join(str(step) for step in CHECKPOINTS[model])
            raise ValueError(f"Invalid checkpoint(s) for {model}: {invalid}. Valid steps: {valid}")
        return requested_checkpoints
    return list(CHECKPOINTS[model])


def build_internal_model_path(
    model_type: str,
    dataset_type: str,
    seed: int,
    checkpoint: int,
) -> str:
    model = (model_type, dataset_type, seed)
    return str(CHECKPOINTS[model][checkpoint])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-type",
        choices=MODEL_TYPES,
        default="transformer",
        help="Model architecture to evaluate.",
    )
    parser.add_argument(
        "--dataset-type",
        choices=DATASET_TYPES,
        default="aperiodic",
        help="Dataset variant used to train the model.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        choices=SEEDS,
        default=0,
        help="Model initialization seed.",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=NUM_GPUS,
        help="Number of GPUs per job.",
    )
    parser.add_argument(
        "--cluster",
        default=CLUSTER,
        help="Beaker cluster on which to run the evaluation.",
    )
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument(
        "--checkpoints",
        "-c",
        nargs="+",
        type=int,
        help="Checkpoint steps to use. Defaults to all checkpoints for the selected model.",
    )
    checkpoint_group.add_argument(
        "--final",
        action="store_true",
        help="Evaluate only the final (largest-step) checkpoint for the selected model.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoints = resolve_checkpoints(
        args.model_type,
        args.dataset_type,
        args.seed,
        args.checkpoints,
        args.final,
    )
    commands = [
        build_command(
            build_internal_model_path(
                args.model_type,
                args.dataset_type,
                args.seed,
                checkpoint,
            ),
            args.model_type,
            TASKS,
            args.gpus,
            args.cluster,
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
