"""Long-context state-tracking tasks from StateBench."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from olmo_eval.common.metrics import LogprobPerTokenMCAccuracyMetric
from olmo_eval.common.types import Instance, LMRequest, RequestType
from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.common import Task, register

STATE_BENCH_REPO = "jacksonp-ai2/state-bench-lc"

STATE_BENCH_CONFIGS = (
    "cube-painting--aperiodic",
    "cube-painting--periodic",
    "cube-painting--r-trivial",
    "integer-code--aperiodic",
    "integer-code--periodic",
    "integer-code--r-trivial",
    "people-in-rooms--aperiodic",
    "people-in-rooms--periodic",
    "people-in-rooms--r-trivial",
    "spreadsheet-cells--aperiodic",
    "spreadsheet-cells--periodic",
    "spreadsheet-cells--r-trivial",
    "status-lights--aperiodic",
    "status-lights--periodic",
    "status-lights--r-trivial",
)

STATE_BENCH_TOKEN_STRATA = (
    "tokens_0_100k",
    "tokens_100k_500k",
    "tokens_500k_1m",
    "tokens_1m_plus",
)

_STATE_BENCH_1M_PLUS_CONFIGS = frozenset(
    {
        "cube-painting--aperiodic",
        "cube-painting--periodic",
        "people-in-rooms--periodic",
        "spreadsheet-cells--aperiodic",
        "spreadsheet-cells--periodic",
        "status-lights--aperiodic",
        "status-lights--periodic",
    }
)

STATE_BENCH_STRATA_BY_CONFIG = {
    config_name: STATE_BENCH_TOKEN_STRATA
    if config_name in _STATE_BENCH_1M_PLUS_CONFIGS
    else STATE_BENCH_TOKEN_STRATA[:-1]
    for config_name in STATE_BENCH_CONFIGS
}


def state_bench_task_name(config_name: str, token_stratum: str) -> str:
    """Return the task name for a dataset config and token-length stratum."""
    normalized = config_name.replace("--", "_").replace("-", "_")
    return f"state_bench_{normalized}_{token_stratum}"


STATE_BENCH_TASKS = tuple(
    state_bench_task_name(config_name, token_stratum)
    for config_name in STATE_BENCH_CONFIGS
    for token_stratum in STATE_BENCH_STRATA_BY_CONFIG[config_name]
)


class StateBench(Task):
    """Rank candidate final states after a long sequence of assignments."""

    metrics = (LogprobPerTokenMCAccuracyMetric(),)
    dataset_split: str

    @property
    def instances(self) -> Iterator[Instance]:
        yield from self._load_instances_cached(split=self.dataset_split)

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        prefix = doc.get("prefix")
        choices = doc.get("choices")
        gold_idx = doc.get("correct_choice_index")
        if not isinstance(prefix, str) or not isinstance(choices, list) or not choices:
            return None
        if not isinstance(gold_idx, int) or not 0 <= gold_idx < len(choices):
            return None
        if not all(isinstance(choice, str) for choice in choices):
            return None

        gold_answer = choices[gold_idx]
        return Instance(
            question=prefix,
            gold_answer=gold_answer,
            choices=tuple(choices),
            metadata={
                "id": doc.get("example_id", index),
                "instance_id": doc.get("instance_id"),
                "index": index,
                "dataset": "state_bench",
                "formatter": doc.get("formatter"),
                "complexity": doc.get("complexity"),
                "gold_idx": gold_idx,
                "gold_text": gold_answer,
                "num_variables": doc.get("num_variables"),
                "num_values": doc.get("num_values"),
                "num_assignments": doc.get("num_assignments"),
                "num_extra_assignments": doc.get("num_extra_assignments"),
                "target_extra_assignments": doc.get("target_extra_assignments"),
                "filler_proportion": doc.get("filler_proportion"),
                "actual_filler_proportion": doc.get("actual_filler_proportion"),
                "filler_distraction": doc.get("filler_distraction"),
                "num_filler_lines": doc.get("num_filler_lines"),
                "token_length": doc.get("token_length"),
                "token_length_is_estimate": doc.get("token_length_is_estimate"),
                "token_stratum": doc.get("token_stratum"),
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        return LMRequest(
            request_type=RequestType.LOGLIKELIHOOD,
            prompt=instance.question,
            continuations=tuple(f" {choice}" for choice in (instance.choices or ())),
            max_length=self.config.max_length,
        )


for _config_name in STATE_BENCH_CONFIGS:
    for _token_stratum in STATE_BENCH_STRATA_BY_CONFIG[_config_name]:
        _name = state_bench_task_name(_config_name, _token_stratum)
        _class_name = (
            "StateBench_" + _config_name.title().replace("-", "_") + "_" + _token_stratum.title()
        )
        _dataset_split = f"test__{_token_stratum}"
        _class = type(
            _class_name,
            (StateBench,),
            {
                "data_source": DataSource(
                    STATE_BENCH_REPO,
                    subset=_config_name,
                    split=_dataset_split,
                ),
                "dataset_split": _dataset_split,
                "__module__": __name__,
                "__qualname__": _class_name,
            },
        )
        globals()[_class_name] = register(_name)(_class)
