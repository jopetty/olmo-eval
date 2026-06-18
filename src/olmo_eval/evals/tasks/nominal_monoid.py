"""Nominal monoid composition tasks."""

from __future__ import annotations

import os
import random
import sys
from collections.abc import Iterator, Sequence
from typing import Any

from olmo_eval.common.metrics import GreedyAccuracyMetric, LogprobPerTokenMCAccuracyMetric
from olmo_eval.common.types import Instance, LMRequest, RequestType
from olmo_eval.data import DataLoader, DataSource
from olmo_eval.evals.suites.registry import AggregationStrategy, make_suite
from olmo_eval.evals.tasks.common import Task, register, register_variant

NOMINAL_MONOID_DATA_DIR = os.environ.get(
    "OLMO_EVAL_NOMINAL_MONOID_DATA_DIR",
    "/weka/oe-training-default/jacksonp/datasets",
)

NOMINAL_MONOID_VARIANTS = ("S5_20", "S20_20", "Z5_20", "Z20_20")


def _data_path(variant: str, split: str) -> str:
    return os.path.join(NOMINAL_MONOID_DATA_DIR, f"nominal-{variant}_{split}.jsonl")


def _format_input(input_sequence: Sequence[Any]) -> str:
    return " * ".join(str(element).strip() for element in input_sequence)


def _format_target(target: Any) -> str:
    if isinstance(target, str):
        return target.strip()
    return " ".join(str(element) for element in target) or "ν"


def _format_example(doc: dict[str, Any]) -> str:
    return f"{_format_input(doc['input'])} = {_format_target(doc['target'])}"


def _unique_wrong_targets(
    docs: Sequence[dict[str, Any]],
    gold_answer: str,
    excluded_answers: set[str],
) -> list[str]:
    seen: set[str] = {gold_answer, *excluded_answers}
    wrong_answers: list[str] = []

    for doc in docs:
        answer = _format_target(doc["target"])
        if answer in seen:
            continue
        seen.add(answer)
        wrong_answers.append(answer)

    return wrong_answers


class NominalMonoidComposition(Task):
    """Base class for nominal monoid composition tasks."""

    data_source = DataSource(path=_data_path("S5_20", "test"))
    fewshot_source = DataSource(path=_data_path("S5_20", "train"))
    metrics = (GreedyAccuracyMetric(),)
    primary_metric = GreedyAccuracyMetric()
    num_fewshot = 5
    fewshot_seed = 42
    n_alternatives = 3

    @property
    def instances(self) -> Iterator[Instance]:
        if self._instances_cache is None:
            self._instances_cache = list(self._load_nominal_monoid_instances())
        yield from self._instances_cache

    def _is_mc(self) -> bool:
        return any(
            isinstance(metric, LogprobPerTokenMCAccuracyMetric) for metric in self.config.metrics
        )

    def _load_docs(self, source: DataSource) -> list[tuple[int, dict[str, Any]]]:
        loader = DataLoader()
        return list(enumerate(loader.load(source)))

    def _load_eval_docs(self) -> list[tuple[int, dict[str, Any]]]:
        source = (
            self.config.data_source
            if isinstance(self.config.data_source, DataSource)
            else self.config.get_data_source()
        )
        return self._load_docs(source)

    def _load_fewshot_docs(self) -> list[tuple[int, dict[str, Any]]]:
        source = self.config.get_fewshot_source(split="train")
        if source is None:
            return []
        return self._load_docs(source)

    def _sample_fewshot_docs(
        self,
        docs: Sequence[tuple[int, dict[str, Any]]],
        index: int,
    ) -> list[dict[str, Any]]:
        if self.config.num_fewshot <= 0 or not docs:
            return []

        rng = random.Random(self.config.fewshot_seed + index)
        return rng.sample(
            [doc for _, doc in docs],
            min(self.config.num_fewshot, len(docs)),
        )

    def _sample_wrong_answers(
        self,
        docs: Sequence[dict[str, Any]],
        gold_answer: str,
        index: int,
        sequence_n_end: Any,
        fewshot_docs: Sequence[dict[str, Any]] = (),
    ) -> list[str]:
        docs = [doc for doc in docs if doc.get("sequence_n_end") == sequence_n_end]
        fewshot_answers = {_format_target(doc["target"]) for doc in fewshot_docs}
        wrong_answers = _unique_wrong_targets(docs, gold_answer, fewshot_answers)
        if not wrong_answers:
            return []

        rng = random.Random(self.config.seed + index)
        return rng.sample(wrong_answers, min(self.n_alternatives, len(wrong_answers)))

    def _load_nominal_monoid_instances(self) -> Iterator[Instance]:
        eval_docs = self._load_eval_docs()
        fewshot_pool = self._load_fewshot_docs()
        raw_eval_docs = [doc for _, doc in eval_docs]

        for index, doc in eval_docs:
            fewshot_docs = self._sample_fewshot_docs(fewshot_pool, index)
            yield self.process_doc(
                doc,
                index,
                fewshot_docs=fewshot_docs,
                candidate_docs=raw_eval_docs,
            )

    def process_doc(
        self,
        doc: dict[str, Any],
        index: int = 0,
        fewshot_docs: list[dict[str, Any]] | None = None,
        candidate_docs: Sequence[dict[str, Any]] | None = None,
    ) -> Instance:
        answer_text = _format_target(doc["target"])
        fewshot_examples = [_format_example(fewshot_doc) for fewshot_doc in fewshot_docs or []]
        prompt = "\n".join([*fewshot_examples, f"{_format_input(doc['input'])} ="])

        if self._is_mc():
            wrong_answers = self._sample_wrong_answers(
                candidate_docs or (),
                answer_text,
                index,
                doc.get("sequence_n_end"),
                fewshot_docs=fewshot_docs or (),
            )
            choices = [answer_text, *wrong_answers]
            rng = random.Random(self.config.seed + index)
            rng.shuffle(choices)
            gold_idx = choices.index(answer_text)
        else:
            choices = [answer_text]
            gold_idx = 0

        return Instance(
            question=prompt,
            gold_answer=answer_text,
            choices=tuple(choices),
            metadata={
                "id": doc.get("id", index),
                "index": index,
                "gold_idx": gold_idx,
                "gold_text": answer_text,
                "fewshot_examples": fewshot_examples,
                "input": doc.get("input"),
                "target": doc.get("target"),
                **({"seq_length": doc["seq_length"]} if "seq_length" in doc else {}),
                **({"base_group": doc["base_group"]} if "base_group" in doc else {}),
                **({"n_0": doc["n_0"]} if "n_0" in doc else {}),
                **({"sequence_n_end": doc["sequence_n_end"]} if "sequence_n_end" in doc else {}),
                **({"dataset_n_end": doc["dataset_n_end"]} if "dataset_n_end" in doc else {}),
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        if self._is_mc():
            continuations = tuple(f" {choice}" for choice in (instance.choices or ()))
        else:
            continuations = (
                (f" {instance.gold_answer}",) if instance.gold_answer is not None else ("",)
            )

        return LMRequest(
            request_type=RequestType.LOGLIKELIHOOD,
            prompt=instance.question,
            continuations=continuations,
        )


_NOMINAL_MONOID_TASKS: list[str] = []

for _variant in NOMINAL_MONOID_VARIANTS:
    _task_name = f"nominal_monoid_{_variant.lower()}"
    _class_name = f"NominalMonoid{_variant.title().replace('_', '')}"
    _cls = type(
        _class_name,
        (NominalMonoidComposition,),
        {
            "data_source": DataSource(path=_data_path(_variant, "test")),
            "fewshot_source": DataSource(path=_data_path(_variant, "train")),
            "__module__": __name__,
            "__qualname__": _class_name,
        },
    )
    setattr(sys.modules[__name__], _class_name, _cls)
    register(_task_name)(_cls)
    register_variant(
        _task_name,
        "greedy",
        metrics=(GreedyAccuracyMetric(),),
        primary_metric=GreedyAccuracyMetric(),
        num_fewshot=5,
    )
    register_variant(
        _task_name,
        "mc",
        metrics=(LogprobPerTokenMCAccuracyMetric(),),
        primary_metric=LogprobPerTokenMCAccuracyMetric(),
        num_fewshot=5,
    )
    _NOMINAL_MONOID_TASKS.append(_task_name)


NOMINAL_MONOID = make_suite(
    "nominal_monoid",
    tuple(_NOMINAL_MONOID_TASKS),
    aggregation=AggregationStrategy.AVERAGE,
    description="Nominal monoid composition tasks with greedy decoding accuracy.",
)

NOMINAL_MONOID_MC = make_suite(
    "nominal_monoid:mc",
    tuple(f"{task}:mc" for task in _NOMINAL_MONOID_TASKS),
    aggregation=AggregationStrategy.AVERAGE,
    description="Nominal monoid composition tasks with target-sequence MC distractors.",
)
