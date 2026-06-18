from olmo_eval.evals.suites.registry import AggregationStrategy, make_suite
from olmo_eval.evals.tasks.formal_langs import (
    ASSIGNMENT_COUNT_BINS,
    BINDING_VARIABLE_COUNTS,
    DYCK_PARENTHESIS_TYPE_COUNTS,
    FORMAL_LANG_BINDING_TASKS,
    FORMAL_LANG_CORE_TASKS,
    FORMAL_LANG_CUBE_BINDING_TASKS,
    FORMAL_LANG_DYCK_TASKS,
    FORMAL_LANG_TASKS,
)

make_suite(
    name="formal_langs",
    tasks=FORMAL_LANG_TASKS,
    aggregation=AggregationStrategy.AVERAGE,
    description="Formal-language string-completion tasks.",
)

for _num_variables in BINDING_VARIABLE_COUNTS:
    make_suite(
        name=f"formal_langs:v{_num_variables}",
        tasks=(
            *FORMAL_LANG_CORE_TASKS,
            *(f"{task_name}:v{_num_variables}" for task_name in FORMAL_LANG_BINDING_TASKS),
        ),
        aggregation=AggregationStrategy.AVERAGE,
        description=(
            "Formal-language string-completion tasks with variable-binding "
            f"tasks filtered to {_num_variables} variables."
        ),
    )

for _num_parenthesis_types in DYCK_PARENTHESIS_TYPE_COUNTS:
    make_suite(
        name=f"formal_langs:k{_num_parenthesis_types}",
        tasks=tuple(
            f"{task_name}:k{_num_parenthesis_types}" for task_name in FORMAL_LANG_DYCK_TASKS
        ),
        aggregation=AggregationStrategy.AVERAGE,
        description=(
            f"Formal-language Dyck tasks filtered to {_num_parenthesis_types} parenthesis types."
        ),
    )

for _min_assignments, _max_assignments in ASSIGNMENT_COUNT_BINS:
    make_suite(
        name=f"formal_langs:a{_min_assignments}-{_max_assignments}",
        tasks=tuple(
            f"{task_name}:a{_min_assignments}-{_max_assignments}"
            for task_name in FORMAL_LANG_BINDING_TASKS
        ),
        aggregation=AggregationStrategy.AVERAGE,
        description=(
            "Formal-language variable-binding tasks filtered to "
            f"{_min_assignments}-{_max_assignments} assignments."
        ),
    )

make_suite(
    name="formal_langs_cube",
    tasks=(
        *FORMAL_LANG_CUBE_BINDING_TASKS,
        *(
            f"{task_name}:v{num_variables}"
            for task_name in FORMAL_LANG_CUBE_BINDING_TASKS
            for num_variables in BINDING_VARIABLE_COUNTS
        ),
    ),
    aggregation=AggregationStrategy.AVERAGE,
    description="Cube-painting formal-language variable-binding tasks.",
)

for _num_variables in BINDING_VARIABLE_COUNTS:
    make_suite(
        name=f"formal_langs_cube:v{_num_variables}",
        tasks=tuple(
            f"{task_name}:v{_num_variables}" for task_name in FORMAL_LANG_CUBE_BINDING_TASKS
        ),
        aggregation=AggregationStrategy.AVERAGE,
        description=(
            "Cube-painting formal-language variable-binding tasks "
            f"filtered to {_num_variables} variables."
        ),
    )

for _min_assignments, _max_assignments in ASSIGNMENT_COUNT_BINS:
    make_suite(
        name=f"formal_langs_cube:a{_min_assignments}-{_max_assignments}",
        tasks=tuple(
            f"{task_name}:a{_min_assignments}-{_max_assignments}"
            for task_name in FORMAL_LANG_CUBE_BINDING_TASKS
        ),
        aggregation=AggregationStrategy.AVERAGE,
        description=(
            "Cube-painting formal-language variable-binding tasks filtered to "
            f"{_min_assignments}-{_max_assignments} assignments."
        ),
    )
