"""Elemental computational tasks from Liu et al. (2026).

Defines evaluations used to measure the emergency of basic computational skills
proposed in Liu et al. (2026) "What Do Language Models Learn and When? The Implicit Curriculum
Hypothesis". The `compositional` tasks are built by composing the simple ones.
"""

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

ELEMENTAL_DATA_PATH = os.environ.get(
    "OLMO_EVAL_ELEMENTAL_DATA_PATH",
    "/weka/oe-training-default/jacksonp/datasets/elemental_simple.csv",
)
ELEMENTAL_COMPOSITIONAL_DATA_PATH = os.environ.get(
    "OLMO_EVAL_ELEMENTAL_COMPOSITIONAL_DATA_PATH",
    "/weka/oe-training-default/jacksonp/datasets/elemental_compositional.csv",
)

ELEMENTAL_CATEGORIES = (
    "uppercase",
    "lowercase",
    "first_letter",
    "last_letter",
    "translate_eng_fr",
    "translate_fr_eng",
    "translate_eng_sp",
    "translate_sp_eng",
    "present_to_gerund",
    "singular_to_plural",
    "country_to_capital",
    "country_to_currency",
)

ELEMENTAL_COMPOSITIONAL_CATEGORIES = (
    "upper_reverse",
    "lower_reverse",
    "upper_first",
    "lower_first",
    "upper_last",
    "lower_last",
    "reverse_first",
    "reverse_last",
    "translate_eng_fr_upper",
    "translate_eng_fr_lower",
    "translate_eng_fr_reverse",
    "translate_eng_fr_first",
    "translate_eng_fr_last",
    "translate_eng_sp_upper",
    "translate_eng_sp_lower",
    "translate_eng_sp_reverse",
    "translate_eng_sp_first",
    "translate_eng_sp_last",
    "translate_fr_eng_upper",
    "translate_fr_eng_lower",
    "translate_fr_eng_reverse",
    "translate_fr_eng_first",
    "translate_fr_eng_last",
    "translate_sp_eng_upper",
    "translate_sp_eng_lower",
    "translate_sp_eng_reverse",
    "translate_sp_eng_first",
    "translate_sp_eng_last",
    "gerund_upper",
    "gerund_lower",
    "gerund_reverse",
    "gerund_first",
    "plural_upper",
    "plural_lower",
    "plural_reverse",
    "plural_first",
    "gerund_upper_reverse",
    "plural_upper_reverse",
    "translate_eng_fr_upper_reverse",
    "translate_eng_sp_upper_reverse",
)


def _format_example(
    doc: dict[str, Any],
    question_field: str = "question",
    answer_field: str = "answer",
) -> str:
    return f"{doc[question_field]} -> {doc[answer_field]}"


def _unique_wrong_answers(
    docs: Sequence[dict[str, Any]],
    gold_answer: str,
    answer_field: str = "answer",
) -> list[str]:
    seen: set[str] = {gold_answer}
    wrong_answers: list[str] = []
    for doc in docs:
        answer = str(doc[answer_field])
        if answer in seen:
            continue
        seen.add(answer)
        wrong_answers.append(answer)
    return wrong_answers


class ElementalTask(Task):
    data_source = DataSource(path=ELEMENTAL_DATA_PATH, split="test")
    metrics = (GreedyAccuracyMetric(),)
    primary_metric = GreedyAccuracyMetric()
    num_fewshot = 5

    category_name: str = ""
    n_alternatives: int = 3
    question_field: str = "question"
    answer_field: str = "answer"

    @property
    def instances(self) -> Iterator[Instance]:
        if self._instances_cache is None:
            self._instances_cache = list(self._load_elemental_instances())
        yield from self._instances_cache

    def _is_mc(self) -> bool:
        return any(
            isinstance(metric, LogprobPerTokenMCAccuracyMetric) for metric in self.config.metrics
        )

    def _load_elemental_docs(self) -> list[tuple[int, dict[str, Any]]]:
        loader = DataLoader()
        source = (
            self.config.data_source
            if isinstance(self.config.data_source, DataSource)
            else self.config.get_data_source()
        )
        return list(enumerate(loader.load(source)))

    def _sample_fewshot_docs(
        self,
        docs: Sequence[tuple[int, dict[str, Any]]],
        index: int,
    ) -> list[dict[str, Any]]:
        if self.config.num_fewshot <= 0:
            return []

        candidates = [doc for doc_index, doc in docs if doc_index != index]
        if not candidates:
            return []

        rng = random.Random(self.config.fewshot_seed + index)
        return rng.sample(candidates, min(self.config.num_fewshot, len(candidates)))

    def _sample_wrong_answers(
        self,
        category_docs: Sequence[dict[str, Any]],
        gold_answer: str,
        index: int,
        fewshot_docs: Sequence[dict[str, Any]] = (),
    ) -> list[str]:
        wrong_answers = _unique_wrong_answers(category_docs, gold_answer, self.answer_field)
        if not wrong_answers:
            return []

        fewshot_answers = {str(doc[self.answer_field]) for doc in fewshot_docs}
        preferred_wrong_answers = [
            answer for answer in wrong_answers if answer not in fewshot_answers
        ]
        if len(preferred_wrong_answers) >= min(self.n_alternatives, len(wrong_answers)):
            wrong_answers = preferred_wrong_answers

        rng = random.Random(self.config.seed + index)
        return rng.sample(wrong_answers, min(self.n_alternatives, len(wrong_answers)))

    def _load_elemental_instances(self) -> Iterator[Instance]:
        docs = self._load_elemental_docs()
        category_docs = [
            (index, doc) for index, doc in docs if doc.get("category_name") == self.category_name
        ]
        raw_category_docs = [doc for _, doc in category_docs]

        for index, doc in category_docs:
            fewshot_docs = self._sample_fewshot_docs(category_docs, index)
            yield self.process_doc(
                doc,
                index,
                fewshot_docs=fewshot_docs,
                category_docs=raw_category_docs,
            )

    def process_doc(
        self,
        doc: dict[str, Any],
        index: int = 0,
        fewshot_docs: list[dict[str, Any]] | None = None,
        category_docs: Sequence[dict[str, Any]] | None = None,
    ) -> Instance:
        category_name = str(doc.get("category_name") or self.category_name)
        question_text = str(doc[self.question_field])
        answer_text = str(doc[self.answer_field])

        fewshot_examples = [
            _format_example(fewshot_doc, self.question_field, self.answer_field)
            for fewshot_doc in fewshot_docs or []
        ]
        prompt = "\n".join([*fewshot_examples, f"{question_text} ->"])

        if self._is_mc():
            wrong_answers = self._sample_wrong_answers(
                category_docs or (),
                answer_text,
                index,
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
                "id": doc.get("index", index),
                "index": index,
                "category_name": category_name,
                "gold_idx": gold_idx,
                "gold_text": answer_text,
                "fewshot_examples": fewshot_examples,
                **({"operations": doc["operations"]} if "operations" in doc else {}),
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


class ElementalCompositionalTask(ElementalTask):
    data_source = DataSource(path=ELEMENTAL_COMPOSITIONAL_DATA_PATH, split="test")
    question_field = "input"
    answer_field = "output"


_ELEMENTAL_TASKS: list[str] = []
_ELEMENTAL_COMPOSITIONAL_TASKS: list[str] = []

for _category in ELEMENTAL_CATEGORIES:
    _task_name = f"elemental_{_category}"
    _class_name = f"Elemental{_category.title().replace('_', '')}"
    _cls = type(
        _class_name,
        (ElementalTask,),
        {
            "category_name": _category,
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
    _ELEMENTAL_TASKS.append(_task_name)

for _category in ELEMENTAL_COMPOSITIONAL_CATEGORIES:
    _task_name = f"elemental_compositional_{_category}"
    _class_name = f"ElementalCompositional{_category.title().replace('_', '')}"
    _cls = type(
        _class_name,
        (ElementalCompositionalTask,),
        {
            "category_name": _category,
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
    _ELEMENTAL_COMPOSITIONAL_TASKS.append(_task_name)


ELEMENTAL = make_suite(
    "elemental",
    tuple(_ELEMENTAL_TASKS),
    aggregation=AggregationStrategy.AVERAGE,
    description="Elemental simple transformation tasks with greedy decoding accuracy.",
)

ELEMENTAL_MC = make_suite(
    "elemental:mc",
    tuple(f"{task}:mc" for task in _ELEMENTAL_TASKS),
    aggregation=AggregationStrategy.AVERAGE,
    description="Elemental simple transformation tasks with within-category MC distractors.",
)

ELEMENTAL_COMPOSITIONAL = make_suite(
    "elemental:compositional",
    tuple(_ELEMENTAL_COMPOSITIONAL_TASKS),
    aggregation=AggregationStrategy.AVERAGE,
    description="Elemental compositional transformation tasks with greedy decoding accuracy.",
)

ELEMENTAL_COMPOSITIONAL_MC = make_suite(
    "elemental:compositional:mc",
    tuple(f"{task}:mc" for task in _ELEMENTAL_COMPOSITIONAL_TASKS),
    aggregation=AggregationStrategy.AVERAGE,
    description=(
        "Elemental compositional transformation tasks with within-category MC distractors."
    ),
)
