"""Sensitivity string-completion tasks."""

from __future__ import annotations

import random
import re
import sys
from collections.abc import Iterator
from os.path import commonprefix
from typing import Any

from olmo_eval.common.metrics import LogprobPerTokenMCAccuracyMetric
from olmo_eval.common.types import Instance, LMRequest, RequestType, Split
from olmo_eval.data import DataLoader, DataSource
from olmo_eval.evals.tasks.common import Task, register

SENSITIVITY_REPO = "jacksonp-ai2/sensitivity-data"
SENSITIVITY_DATASETS = (
    "aperiodic_supervised_n100000_v26_a50_m64_z1p2_s3",
    "aperiodic_unsupervised_n100000_v26_a50_m64_z1p2_s2",
    "periodic_supervised_n100000_v26_a50_m64_z1p2_s5",
    "periodic_unsupervised_n100000_v26_a50_m64_z1p2_s4",
    "r-trivial_supervised_n100000_v26_a50_m64_z1p2_s1",
    "r-trivial_unsupervised_n100000_v26_a50_m64_z1p2_s0",
)


def _file_prefix_from_dataset(dataset: str) -> str:
    match = re.search(r"_n\d+(?:_|$)", dataset)
    if match is None:
        return dataset
    return dataset[: match.start()]


SENSITIVITY_FILE_PREFIXES = tuple(
    _file_prefix_from_dataset(dataset) for dataset in SENSITIVITY_DATASETS
)
SENSITIVITY_TASKS = tuple(
    f"sensitivity_{file_prefix.replace('-', '_')}" for file_prefix in SENSITIVITY_FILE_PREFIXES
)


def _format_text(value: Any) -> str:
    return "" if value is None else str(value)


def _alt_sort_key(key: str) -> tuple[int, int | str]:
    suffix = key.removeprefix("alt_")
    if suffix.isdigit():
        return (0, int(suffix))
    return (1, suffix)


def _alt_keys(doc: dict[str, Any]) -> list[str]:
    return sorted((key for key in doc if key.startswith("alt_")), key=_alt_sort_key)


def _unique(values: list[str]) -> list[str]:
    unique_values: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique_values.append(value)
    return unique_values


def _shared_prefix(candidates: list[str]) -> str:
    prefix = commonprefix(candidates)
    return prefix


def _completion(candidate: str, prefix: str) -> str:
    return candidate[len(prefix) :]


def _class_name_from_file_prefix(file_prefix: str) -> str:
    parts = re.split(r"[-_]+", file_prefix)
    return "".join(part[:1].upper() + part[1:] for part in parts if part)


def _task_name_from_file_prefix(file_prefix: str) -> str:
    return f"sensitivity_{file_prefix.replace('-', '_')}"


class SensitivityCompletion(Task):
    """Base class for sensitivity completion tasks."""

    data_source = DataSource(SENSITIVITY_REPO, split="train")
    metrics = (LogprobPerTokenMCAccuracyMetric(),)
    num_fewshot = 0
    limit = 1000
    seed = 42
    split = Split.TRAIN

    file_prefix: str = ""
    eval_name: str = ""

    @property
    def instances(self) -> Iterator[Instance]:
        if self._instances_cache is None:
            self._instances_cache = [
                self.process_doc(doc, index)
                for index, doc in enumerate(self._load_sensitivity_docs())
            ]
        yield from self._instances_cache

    def _load_sensitivity_docs(self) -> Iterator[dict[str, Any]]:
        loader = DataLoader()
        yield from loader.load(self.config.get_data_source())

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance:
        text = _format_text(doc["text"])
        alternatives = [_format_text(doc[key]) for key in _alt_keys(doc)]
        candidates = _unique([text, *alternatives])
        prefix = _shared_prefix(candidates)
        correct_completion = _completion(text, prefix)
        choices = [_completion(candidate, prefix) for candidate in candidates]

        rng = random.Random(self.config.seed + index)
        rng.shuffle(choices)
        gold_idx = choices.index(correct_completion)

        metadata = {
            key: value for key, value in doc.items() if key != "text" and not key.startswith("alt_")
        }
        metadata.update(
            {
                "id": doc.get("id", index),
                "index": index,
                "dataset": "sensitivity",
                "eval_name": self.eval_name,
                "file_prefix": self.file_prefix,
                "gold_idx": gold_idx,
                "gold_text": text,
                "gold_completion": correct_completion,
                "prefix": prefix,
                "alternatives": tuple(alternatives),
                "distractors": tuple(choice for choice in choices if choice != correct_completion),
            }
        )

        return Instance(
            question=prefix,
            gold_answer=correct_completion,
            choices=tuple(choices),
            metadata=metadata,
        )

    def format_request(self, instance: Instance) -> LMRequest:
        return LMRequest(
            request_type=RequestType.LOGLIKELIHOOD,
            prompt=instance.question,
            continuations=instance.choices or (),
            max_length=self.config.max_length,
        )


for _dataset in SENSITIVITY_DATASETS:
    _file_prefix = _file_prefix_from_dataset(_dataset)
    _task_name = _task_name_from_file_prefix(_file_prefix)
    _class_name = _class_name_from_file_prefix(_file_prefix)
    _cls = type(
        _class_name,
        (SensitivityCompletion,),
        {
            "__module__": __name__,
            "__qualname__": _class_name,
            "data_source": DataSource(SENSITIVITY_REPO, subset=_dataset, split="train"),
            "file_prefix": _file_prefix,
            "eval_name": _class_name,
        },
    )
    setattr(sys.modules[__name__], _class_name, _cls)
    register(_task_name)(_cls)
