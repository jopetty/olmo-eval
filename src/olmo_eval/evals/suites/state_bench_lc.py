from olmo_eval.evals.suites.registry import AggregationStrategy, make_suite
from olmo_eval.evals.tasks.state_bench_lc import STATE_BENCH_LC_CONFIGS, STATE_BENCH_LC_TASKS


def _task_name(config_name: str) -> str:
    normalized = config_name.replace("--", "_").replace("-", "_")
    return f"state_bench_lc_{normalized}"


make_suite(
    name="state_bench_lc",
    tasks=STATE_BENCH_LC_TASKS,
    aggregation=AggregationStrategy.AVERAGE,
    description="Long-context StateBench tasks across all formats and complexity classes.",
)

for _complexity in ("aperiodic", "periodic", "r-trivial"):
    make_suite(
        name=f"state_bench_lc:{_complexity.replace('-', '_')}",
        tasks=tuple(
            _task_name(config_name)
            for config_name in STATE_BENCH_LC_CONFIGS
            if config_name.endswith(f"--{_complexity}")
        ),
        aggregation=AggregationStrategy.AVERAGE,
        description=f"Long-context StateBench tasks in the {_complexity} complexity class.",
    )

for _formatter in (
    "cube-painting",
    "integer-code",
    "people-in-rooms",
    "spreadsheet-cells",
    "status-lights",
):
    make_suite(
        name=f"state_bench_lc:{_formatter.replace('-', '_')}",
        tasks=tuple(
            _task_name(config_name)
            for config_name in STATE_BENCH_LC_CONFIGS
            if config_name.startswith(f"{_formatter}--")
        ),
        aggregation=AggregationStrategy.AVERAGE,
        description=f"Long-context StateBench tasks using the {_formatter} format.",
    )
