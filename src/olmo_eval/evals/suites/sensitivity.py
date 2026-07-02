from olmo_eval.evals.suites.registry import AggregationStrategy, make_suite
from olmo_eval.evals.tasks.sensitivity import SENSITIVITY_TASKS

make_suite(
    name="sensitivity",
    tasks=SENSITIVITY_TASKS,
    aggregation=AggregationStrategy.AVERAGE,
    description="Sensitivity string-completion tasks.",
)
