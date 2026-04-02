from olmo_eval.evals.suites.registry import make_suite

make_suite(
    "aime_pass_at_32_rlzero",
    (
        "aime_2024:pass_at_32_rlzero",
        "aime_2025:pass_at_32_rlzero",
    ),
    description="AIME 2024+2025 pass@32 evaluation for RL-zero models",
)
