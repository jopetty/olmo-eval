import argparse
import shlex
import subprocess
from pathlib import Path

GROUP = "jacksonp-directly-trained-synthetic-evals"
CLUSTER = "ai2/jupiter"
PRIORITY = "urgent"
NUM_GPUS = 2
WORKSPACE = "ai2/beyond-state"
BUDGET = "ai2/oe-other"

CHECKPOINT_BASE = Path("/weka/oe-training-default/ai2-llm/checkpoints/jacksonp")
MODEL_TYPES = ("hybrid", "transformer")
DATASET_TYPES = ("r-trivial", "aperiodic", "periodic")
SUPERVISION_LEVELS = (0, 50, 100)

# These are the directly-trained 200M-token runs currently available in the
# checkpoint store. The omitted combinations do not have corresponding runs.
MODEL_ROOTS = {
    ("hybrid", "aperiodic", 0): CHECKPOINT_BASE
    / "hybrid-aperiodic_0supervision_n200000000_v26_a50_m64_z1p2_assignments_lt10-Cx2"
    / "60M",
    ("hybrid", "aperiodic", 50): CHECKPOINT_BASE
    / "hybrid-aperiodic_50supervision_n200000000_v26_a50_m64_z1p2_assignments_lt10-Cx4-init_seed42"
    / "60M",
    ("hybrid", "aperiodic", 100): CHECKPOINT_BASE
    / "hybrid-aperiodic_100supervision_n200000000_v26_a50_m64_z1p2_assignments_lt10-Cx4"
    / "60M",
    ("hybrid", "periodic", 0): CHECKPOINT_BASE
    / "hybrid-periodic_0supervision_n200000000_v26_a50_m64_z1p2_assignments_lt10-Cx2"
    / "60M",
    ("hybrid", "periodic", 50): CHECKPOINT_BASE
    / "hybrid-periodic_50supervision_n200000000_v26_a50_m64_z1p2_assignments_lt10-Cx4"
    / "60M",
    ("hybrid", "periodic", 100): CHECKPOINT_BASE
    / "hybrid-periodic_100supervision_n200000000_v26_a50_m64_z1p2_assignments_lt10-Cx4"
    / "60M",
    ("hybrid", "r-trivial", 0): CHECKPOINT_BASE
    / "hybrid-r-trivial_0supervision_n200000000_v26_a50_m64_z1p2_assignments_lt5-Cx2"
    / "60M",
    ("hybrid", "r-trivial", 50): CHECKPOINT_BASE
    / "hybrid-r-trivial_50supervision_n200000000_v26_a50_m64_z1p2_assignments_lt5-Cx2"
    / "60M",
    ("hybrid", "r-trivial", 100): CHECKPOINT_BASE
    / "hybrid-r-trivial_100supervision_n200000000_v26_a50_m64_z1p2_assignments_lt5-Cx2"
    / "60M",
    ("transformer", "aperiodic", 0): CHECKPOINT_BASE
    / "transformer-aperiodic_0supervision_n200000000_v26_a50_m64_z1p2_assignments_lt10-Cx2"
    / "60M",
    ("transformer", "aperiodic", 50): CHECKPOINT_BASE
    / "transformer-aperiodic_50supervision_n200000000_v26_a50_m64_z1p2_assignments_lt10-Cx4"
    / "60M",
    ("transformer", "aperiodic", 100): CHECKPOINT_BASE
    / "transformer-aperiodic_100supervision_n200000000_v26_a50_m64_z1p2_assignments_lt10-Cx4"
    / "60M",
    ("transformer", "periodic", 0): CHECKPOINT_BASE
    / "transformer-periodic_0supervision_n200000000_v26_a50_m64_z1p2_assignments_lt10-Cx2-init_seed42"
    / "60M",
    ("transformer", "periodic", 50): CHECKPOINT_BASE
    / "transformer-periodic_50supervision_n200000000_v26_a50_m64_z1p2_assignments_lt10-Cx4"
    / "60M",
    ("transformer", "periodic", 100): CHECKPOINT_BASE
    / "transformer-periodic_100supervision_n200000000_v26_a50_m64_z1p2_assignments_lt10-Cx4"
    / "60M",
    ("transformer", "r-trivial", 0): CHECKPOINT_BASE
    / "transformer-r-trivial_0supervision_n200000000_v26_a50_m64_z1p2_assignments_lt5-Cx2"
    / "60M",
    ("transformer", "r-trivial", 50): CHECKPOINT_BASE
    / "transformer-r-trivial_50supervision_n200000000_v26_a50_m64_z1p2_assignments_lt5-Cx2"
    / "60M",
    ("transformer", "r-trivial", 100): CHECKPOINT_BASE
    # / "transformer-r-trivial_100supervision_n200000000_v26_a50_m64_z1p2_assignments_lt5-Cx2-init_seed0"
    # / "transformer-r-trivial_100supervision_n200000000_v26_a50_m64_z1p2_assignments_lt5-Cx2-init_seed7"
    / "transformer-r-trivial_100supervision_n200000000_v26_a50_m64_z1p2_assignments_lt5-Cx2-init_seed42"
    / "60M",
}

CHECKPOINT_STEPS = {
    ("hybrid", "aperiodic", 0): [0, 2034, 2260, 4294, 4521, 8589, 9000, 9041],
    ("hybrid", "aperiodic", 50): [
        0,
        2034,
        2260,
        4294,
        4521,
        8589,
        9041,
        17177,
        18000,
        18082,
    ],
    ("hybrid", "aperiodic", 100): [
        0,
        2034,
        2260,
        4294,
        4521,
        8589,
        9041,
        17177,
        18000,
        18082,
    ],
    ("hybrid", "periodic", 0): [0, 2034, 2260, 4294, 4521, 8589, 9000, 9041],
    ("hybrid", "periodic", 50): [
        0,
        2034,
        2260,
        4294,
        4521,
        8589,
        9041,
        17177,
        18000,
        18082,
    ],
    ("hybrid", "periodic", 100): [
        0,
        2034,
        2260,
        4294,
        4521,
        8589,
        9041,
        17177,
        18000,
        18082,
    ],
    ("hybrid", "r-trivial", 0): [0, 2034, 2260, 4294, 4521, 8589, 9000, 9041],
    ("hybrid", "r-trivial", 50): [0, 2034, 2260, 4294, 4521, 8589, 9000, 9041],
    ("hybrid", "r-trivial", 100): [0, 2034, 2260, 4294, 4521, 8589, 9000, 9041],
    ("transformer", "aperiodic", 0): [0, 1971, 2190, 4161, 4381, 8000, 8323, 8762],
    ("transformer", "aperiodic", 50): [
        0,
        1971,
        2190,
        4000,
        4161,
        4381,
        8323,
        8762,
        16647,
        17000,
        17524,
    ],
    ("transformer", "aperiodic", 100): [
        0,
        1971,
        2190,
        4161,
        4381,
        8323,
        8762,
        16647,
        17000,
        17524,
    ],
    ("transformer", "periodic", 0): [0, 1971, 2190, 4161, 4381, 8000, 8323, 8762],
    ("transformer", "periodic", 50): [
        0,
        1971,
        2190,
        4161,
        4381,
        8323,
        8762,
        16647,
        17000,
        17524,
    ],
    ("transformer", "periodic", 100): [
        0,
        1971,
        2190,
        4161,
        4381,
        8323,
        8762,
        16647,
        17000,
        17524,
    ],
    ("transformer", "r-trivial", 0): [0, 1971, 2190, 4161, 4381, 8000, 8323, 8762],
    ("transformer", "r-trivial", 50): [0, 1971, 2190, 4161, 4381, 8000, 8323, 8762],
    ("transformer", "r-trivial", 100): [0, 1971, 2190, 4161, 4381, 8000, 8323, 8762],
}

CHECKPOINTS = {
    model: {step: MODEL_ROOTS[model] / f"step{step}/" for step in steps}
    for model, steps in CHECKPOINT_STEPS.items()
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

TRANSFORMER_60M_HARNESS_OVERRIDES = [
    ("provider.kwargs.hf_overrides", '{"architectures":["TransformersForCausalLM"]}'),
]

HYBRID_HARNESS_OVERRIDES = [
    ("provider.kind", "hf"),
    ("provider.trust_remote_code", "true"),
    ("provider.dtype", "bfloat16"),
]

# HYBRID_HARNESS_OVERRIDES = [
#     ("provider.kind", "vllm"),
#     ("provider.package", "wheel"),
#     ("provider.kwargs.mamba_ssm_cache_dtype", "float32"),
#     ("provider.kwargs.attention_backend", "FLASH_ATTN"),
#     ("provider.trust_remote_code", "true"),
# ]

HYBRID_IMAGE = "yashasbls/olmo-eval-vllm-g79d31a3f9-tch2100cu128-2026-05-23"
# HYBRID_ENVS = [
#     ("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1"),
# ]

SECRET_ENVS = [
    "JACKSONP_HF_TOKEN:HF_TOKEN",
    "jacksonp_GITHUB_TOKEN:GITHUB_TOKEN",
]


def get_model_short_name(model_path: str) -> str:
    parts = model_path.rstrip("/").split("/")
    return "_".join(parts[-3:]).lower().replace("/", "_").replace(":", "_")


def build_command(
    model_path: str,
    model_type: str,
    tasks: list[str],
    num_gpus: int = NUM_GPUS,
) -> list[str]:
    model_short = get_model_short_name(model_path)
    tasks_short = "-".join(task.replace(":", "_") for task in tasks)
    exp_name = f"{model_short}-{tasks_short}"

    cmd = ["uv", "run", "olmo-eval", "beaker", "launch"]
    cmd.extend(["-H", "default", "-n", exp_name])

    harness_overrides = list(BASE_HARNESS_OVERRIDES)
    if model_type == "hybrid":
        harness_overrides = [
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
    else:  # transformer
        harness_overrides = list(BASE_HARNESS_OVERRIDES)
        harness_overrides.extend(CUSTOM_CONFIG_HARNESS_OVERRIDES)
        harness_overrides.extend(TRANSFORMER_60M_HARNESS_OVERRIDES)

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

    if model_type == "hybrid":
        cmd.extend(["--image", HYBRID_IMAGE])
    #     for key, value in HYBRID_ENVS:
    #         cmd.extend(["--env", f"{key}={value}"])

    for secret_env in SECRET_ENVS:
        cmd.extend(["--secret-env", secret_env])
    cmd.extend(["--no-follow", "-y"])
    return cmd


def resolve_checkpoints(
    model_type: str,
    dataset_type: str,
    supervision: int,
    requested_checkpoints: list[int] | None,
    final: bool = False,
) -> list[int]:
    model = (model_type, dataset_type, supervision)
    if model not in CHECKPOINTS:
        available = ", ".join(
            f"{model_type}/{dataset}/{level}"
            for model_type_, dataset, level in MODEL_ROOTS
            if model_type_ == model_type
        )
        raise ValueError(
            f"No checkpoint is configured for {model_type}/{dataset_type}/{supervision}."
            f" Available {model_type} combinations: {available}"
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
    model_type: str, dataset_type: str, supervision: int, checkpoint: int
) -> str:
    model = (model_type, dataset_type, supervision)
    return str(CHECKPOINTS[model][checkpoint])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-type",
        type=str,
        choices=MODEL_TYPES,
        default="transformer",
        help="Model architecture to evaluate.",
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        choices=DATASET_TYPES,
        default="aperiodic",
        help="Dataset variant used to train the model.",
    )
    parser.add_argument(
        "--supervision",
        type=int,
        choices=SUPERVISION_LEVELS,
        default=0,
        help="Supervision percentage used to train the model.",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=NUM_GPUS,
        help="Number of GPUs per job.",
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
        args.supervision,
        args.checkpoints,
        args.final,
    )
    commands = [
        build_command(
            build_internal_model_path(
                args.model_type,
                args.dataset_type,
                args.supervision,
                checkpoint,
            ),
            args.model_type,
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
