from olmo_eval.evals.suites.registry import AggregationStrategy, make_suite
from olmo_eval.evals.tasks.minerva_math import MATH_SUBSETS

MINERVA_MATH_TASKS = tuple(f"minerva_math_{subset}" for subset in MATH_SUBSETS)

MINERVA_MATH_ALL_4SHOT = make_suite(
    "minerva_math_all:4shot",
    tuple(f"{t}:4shot" for t in MINERVA_MATH_TASKS),
    aggregation=AggregationStrategy.AVERAGE,
    description="All 7 Minerva MATH subsets with 4-shot prompting",
)