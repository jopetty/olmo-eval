"""Long-context state-tracking tasks from StateBench."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from olmo_eval.common.metrics import LogprobPerTokenMCAccuracyMetric
from olmo_eval.common.types import Instance, LMRequest, RequestType
from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.common import Task, register

STATE_BENCH_LC_REPO = "jacksonp-ai2/state-bench-lc"

STATE_BENCH_LC_CONFIGS = (
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


def _task_name(config_name: str) -> str:
    normalized = config_name.replace("--", "_").replace("-", "_")
    return f"state_bench_lc_{normalized}"


STATE_BENCH_LC_TASKS = tuple(_task_name(config_name) for config_name in STATE_BENCH_LC_CONFIGS)


class StateBenchLongContext(Task):
    """Rank candidate final states after a long sequence of assignments."""

    metrics = (LogprobPerTokenMCAccuracyMetric(),)

    @property
    def instances(self) -> Iterator[Instance]:
        yield from self._load_instances_cached()

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
                "dataset": "state_bench_lc",
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
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        return LMRequest(
            request_type=RequestType.LOGLIKELIHOOD,
            prompt=instance.question,
            continuations=tuple(f" {choice}" for choice in (instance.choices or ())),
            max_length=self.config.max_length,
        )


for _config_name in STATE_BENCH_LC_CONFIGS:
    _name = _task_name(_config_name)
    _class_name = "StateBenchLongContext_" + _config_name.title().replace("-", "_")
    _class = type(
        _class_name,
        (StateBenchLongContext,),
        {
            "data_source": DataSource(
                STATE_BENCH_LC_REPO,
                subset=_config_name,
                split="test",
            ),
            "__module__": __name__,
            "__qualname__": _class_name,
        },
    )
    globals()[_class_name] = register(_name)(_class)
