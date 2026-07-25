from olmo_eval.evals.suites.registry import (
    AggregationStrategy,
    make_suite,
)
from olmo_eval.evals.tasks.monoid import (
    COMMUTATIVE_MONOIDS,
    INSOLUBLE_MONOIDS,
    NONCOMMUTATIVE_SOLUBLE_MONOIDS,
)

make_suite(
    name="monoid:commutative",
    tasks=tuple(COMMUTATIVE_MONOIDS),
    aggregation=AggregationStrategy.AVERAGE,
)

make_suite(
    name="monoid:commutative:3shot",
    tasks=tuple(f"{task}:3shot" for task in COMMUTATIVE_MONOIDS),
    aggregation=AggregationStrategy.AVERAGE,
)

make_suite(
    name="monoid:soluble",
    tasks=tuple(NONCOMMUTATIVE_SOLUBLE_MONOIDS),
    aggregation=AggregationStrategy.AVERAGE,
)

make_suite(
    name="monoid:soluble:3shot",
    tasks=tuple(f"{task}:3shot" for task in NONCOMMUTATIVE_SOLUBLE_MONOIDS),
    aggregation=AggregationStrategy.AVERAGE,
)

make_suite(
    name="monoid:insoluble",
    tasks=tuple(INSOLUBLE_MONOIDS),
    aggregation=AggregationStrategy.AVERAGE,
)

make_suite(
    name="monoid:insoluble:3shot",
    tasks=tuple(f"{task}:3shot" for task in INSOLUBLE_MONOIDS),
    aggregation=AggregationStrategy.AVERAGE,
)
