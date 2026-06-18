"""Formal-language string-completion tasks."""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence
from typing import Any

from olmo_eval.common.metrics import LogprobPerTokenMCAccuracyMetric
from olmo_eval.common.types import Instance, LMRequest, RequestType
from olmo_eval.data import DataLoader, DataSource
from olmo_eval.evals.tasks.common import Task, register

FORMAL_LANGS_REPO = "jacksonp-ai2/formal-langs"
BINDING_VARIABLE_COUNTS = (2, 4, 6)
DYCK_PARENTHESIS_TYPE_COUNTS = tuple(range(1, 7))
ASSIGNMENT_COUNT_BINS = (
    (1, 10),
    (11, 20),
    (21, 30),
    (31, 40),
    (41, 50),
)
FORMAL_LANG_CORE_TASKS = (
    "formal_langs_copy",
    "formal_langs_reverse_copy",
    "formal_langs_sort",
    "formal_langs_char_shift_1",
    "formal_langs_char_shift_2",
    "formal_langs_char_shift_5",
    "formal_langs_k_dyck",
    "formal_langs_k_shuffle_dyck",
)
FORMAL_LANG_DYCK_TASKS = (
    "formal_langs_k_dyck",
    "formal_langs_k_shuffle_dyck",
)
FORMAL_LANG_BASE_TASKS = (
    *FORMAL_LANG_CORE_TASKS,
    "formal_langs_var_unique",
    "formal_langs_var_undefined",
    "formal_langs_var_reassign_var",
    "formal_langs_var_reassign_const",
    "formal_langs_cube_unique",
    "formal_langs_cube_undefined",
    "formal_langs_cube_reassign_var",
    "formal_langs_cube_reassign_const",
)
FORMAL_LANG_SYMBOLIC_BINDING_TASKS = (
    "formal_langs_var_unique",
    "formal_langs_var_undefined",
    "formal_langs_var_reassign_var",
    "formal_langs_var_reassign_const",
)
FORMAL_LANG_CUBE_BINDING_TASKS = (
    "formal_langs_cube_unique",
    "formal_langs_cube_undefined",
    "formal_langs_cube_reassign_var",
    "formal_langs_cube_reassign_const",
)
FORMAL_LANG_BINDING_TASKS = (*FORMAL_LANG_SYMBOLIC_BINDING_TASKS, *FORMAL_LANG_CUBE_BINDING_TASKS)
FORMAL_LANG_VARIABLE_COUNT_TASKS = tuple(
    f"{task_name}:v{num_variables}"
    for task_name in FORMAL_LANG_BINDING_TASKS
    for num_variables in BINDING_VARIABLE_COUNTS
)
FORMAL_LANG_DYCK_PARENTHESIS_TYPE_TASKS = tuple(
    f"{task_name}:k{num_parenthesis_types}"
    for task_name in FORMAL_LANG_DYCK_TASKS
    for num_parenthesis_types in DYCK_PARENTHESIS_TYPE_COUNTS
)
FORMAL_LANG_ASSIGNMENT_COUNT_TASKS = tuple(
    f"{task_name}:a{min_assignments}-{max_assignments}"
    for task_name in FORMAL_LANG_BINDING_TASKS
    for min_assignments, max_assignments in ASSIGNMENT_COUNT_BINS
)
FORMAL_LANG_TASKS = (
    *FORMAL_LANG_BASE_TASKS,
    *FORMAL_LANG_VARIABLE_COUNT_TASKS,
    *FORMAL_LANG_DYCK_PARENTHESIS_TYPE_TASKS,
    *FORMAL_LANG_ASSIGNMENT_COUNT_TASKS,
)


def _format_text(value: Any) -> str:
    return "" if value is None else str(value)


def _is_cube_doc(doc: dict[str, Any]) -> bool:
    return _format_text(doc.get("binding_language")).startswith("cube-")


def _first_int(doc: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        value = doc.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _doc_parenthesis_type_count(doc: dict[str, Any]) -> int | None:
    return _first_int(doc, ("num_parenthesis_types", "num_parens", "k"))


def _doc_assignment_count(doc: dict[str, Any]) -> int | None:
    return _first_int(doc, ("num_assignments", "num_total_assignments", "total_assignments"))


def _format_example(doc: dict[str, Any]) -> str:
    separator = " " if _is_cube_doc(doc) else "="
    return f"{_format_text(doc['input'])}{separator}{_format_text(doc['correct'])}"


def _format_query(doc: dict[str, Any]) -> str:
    separator = " " if _is_cube_doc(doc) else "="
    return f"{_format_text(doc['input'])}{separator}"


def _distractor_sort_key(key: str) -> tuple[int, int | str]:
    suffix = key.removeprefix("distractor_")
    if suffix.isdigit():
        return (0, int(suffix))
    return (1, suffix)


def _choices_from_doc(doc: dict[str, Any]) -> list[str]:
    correct = _format_text(doc["correct"])
    choices = [correct]
    seen = {correct}

    distractor_keys = sorted(
        (key for key in doc if key.startswith("distractor_")),
        key=_distractor_sort_key,
    )
    for key in distractor_keys:
        choice = _format_text(doc[key])
        if choice in seen:
            continue
        seen.add(choice)
        choices.append(choice)

    return choices


class FormalLanguageCompletion(Task):
    """Base class for formal-language completion tasks."""

    data_source = DataSource(FORMAL_LANGS_REPO, split="test")
    fewshot_source = DataSource(FORMAL_LANGS_REPO, split="train")
    metrics = (LogprobPerTokenMCAccuracyMetric(),)
    num_fewshot = 3
    fewshot_seed = 42
    fewshot_split = "train"
    num_variables: int | None = None
    num_parenthesis_types: int | None = None
    assignment_count_range: tuple[int, int] | None = None

    @property
    def instances(self) -> Iterator[Instance]:
        if self._instances_cache is None:
            self._instances_cache = list(self._load_formal_language_instances())
        yield from self._instances_cache

    def _load_docs(self, source: DataSource) -> list[tuple[int, dict[str, Any]]]:
        loader = DataLoader()
        return list(enumerate(loader.load(source)))

    def _load_eval_docs(self) -> list[tuple[int, dict[str, Any]]]:
        return self._filter_docs(self._load_docs(self.config.get_data_source()))

    def _load_fewshot_docs(self) -> list[tuple[int, dict[str, Any]]]:
        source = self.config.get_fewshot_source(split=self.fewshot_split)
        if source is None:
            return []
        return self._filter_docs(self._load_docs(source))

    def _filter_docs(
        self,
        docs: list[tuple[int, dict[str, Any]]],
    ) -> list[tuple[int, dict[str, Any]]]:
        if (
            self.num_variables is None
            and self.num_parenthesis_types is None
            and self.assignment_count_range is None
        ):
            return docs

        filtered = docs
        filters: list[str] = []
        if self.num_variables is not None:
            filtered = [
                (index, doc)
                for index, doc in filtered
                if _first_int(doc, ("num_variables",)) == self.num_variables
            ]
            filters.append(f"num_variables={self.num_variables}")

        if self.num_parenthesis_types is not None:
            filtered = [
                (index, doc)
                for index, doc in filtered
                if _doc_parenthesis_type_count(doc) == self.num_parenthesis_types
            ]
            filters.append(f"num_parenthesis_types={self.num_parenthesis_types}")

        if self.assignment_count_range is not None:
            min_assignments, max_assignments = self.assignment_count_range
            filtered = [
                (index, doc)
                for index, doc in filtered
                if (
                    (assignment_count := _doc_assignment_count(doc)) is not None
                    and min_assignments <= assignment_count <= max_assignments
                )
            ]
            filters.append(f"num_assignments={min_assignments}-{max_assignments}")

        if not filtered:
            raise ValueError(f"{self.config.name} found no documents matching {', '.join(filters)}")
        return filtered

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

    def _load_formal_language_instances(self) -> Iterator[Instance]:
        fewshot_pool = self._load_fewshot_docs()
        for index, doc in self._load_eval_docs():
            fewshot_docs = self._sample_fewshot_docs(fewshot_pool, index)
            yield self.process_doc(doc, index, fewshot_docs=fewshot_docs)

    def process_doc(
        self,
        doc: dict[str, Any],
        index: int = 0,
        fewshot_docs: list[dict[str, Any]] | None = None,
    ) -> Instance:
        input_text = _format_text(doc["input"])
        correct = _format_text(doc["correct"])
        choices = _choices_from_doc(doc)

        rng = random.Random(self.config.seed + index)
        rng.shuffle(choices)
        gold_idx = choices.index(correct)

        fewshot_examples = [_format_example(fewshot_doc) for fewshot_doc in fewshot_docs or []]
        question = "\n".join([*fewshot_examples, _format_query(doc)])

        return Instance(
            question=question,
            gold_answer=correct,
            choices=tuple(choices),
            metadata={
                "id": doc.get("id", index),
                "index": index,
                "dataset": "formal_langs",
                "gold_idx": gold_idx,
                "gold_text": correct,
                "input": input_text,
                "correct": correct,
                "distractors": tuple(choice for choice in choices if choice != correct),
                "fewshot_examples": fewshot_examples,
                **({"n": doc["n"]} if "n" in doc else {}),
                **({"input_length": doc["input_length"]} if "input_length" in doc else {}),
                **({"k": doc["k"]} if "k" in doc else {}),
                **({"shift": doc["shift"]} if "shift" in doc else {}),
                **({"num_examples": doc["num_examples"]} if "num_examples" in doc else {}),
                **({"num_parens": doc["num_parens"]} if "num_parens" in doc else {}),
                **(
                    {"num_parenthesis_types": doc["num_parenthesis_types"]}
                    if "num_parenthesis_types" in doc
                    else {}
                ),
                **({"a": doc["a"]} if "a" in doc else {}),
                **(
                    {"binding_language": doc["binding_language"]}
                    if "binding_language" in doc
                    else {}
                ),
                **({"num_variables": doc["num_variables"]} if "num_variables" in doc else {}),
                **({"num_assignments": doc["num_assignments"]} if "num_assignments" in doc else {}),
                **(
                    {"num_total_assignments": doc["num_total_assignments"]}
                    if "num_total_assignments" in doc
                    else {}
                ),
                **(
                    {"total_assignments": doc["total_assignments"]}
                    if "total_assignments" in doc
                    else {}
                ),
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        return LMRequest(
            request_type=RequestType.LOGLIKELIHOOD,
            prompt=instance.question,
            continuations=instance.choices or (),
            max_length=self.config.max_length,
        )


@register("formal_langs_copy")
class FormalLanguageCopy(FormalLanguageCompletion):
    data_source = DataSource(FORMAL_LANGS_REPO, split="test", subset="copy")
    fewshot_source = DataSource(FORMAL_LANGS_REPO, split="train", subset="copy")


@register("formal_langs_reverse_copy")
class FormalLanguageReverseCopy(FormalLanguageCompletion):
    data_source = DataSource(FORMAL_LANGS_REPO, split="test", subset="reverse-copy")
    fewshot_source = DataSource(FORMAL_LANGS_REPO, split="train", subset="reverse-copy")


@register("formal_langs_sort")
class FormalLanguageSort(FormalLanguageCompletion):
    data_source = DataSource(FORMAL_LANGS_REPO, split="test", subset="sort")
    fewshot_source = DataSource(FORMAL_LANGS_REPO, split="train", subset="sort")


@register("formal_langs_char_shift_1")
class FormalLanguageCharShift1(FormalLanguageCompletion):
    data_source = DataSource(FORMAL_LANGS_REPO, split="test", subset="char-shift-1")
    fewshot_source = DataSource(FORMAL_LANGS_REPO, split="train", subset="char-shift-1")


@register("formal_langs_char_shift_2")
class FormalLanguageCharShift2(FormalLanguageCompletion):
    data_source = DataSource(FORMAL_LANGS_REPO, split="test", subset="char-shift-2")
    fewshot_source = DataSource(FORMAL_LANGS_REPO, split="train", subset="char-shift-2")


@register("formal_langs_char_shift_5")
class FormalLanguageCharShift5(FormalLanguageCompletion):
    data_source = DataSource(FORMAL_LANGS_REPO, split="test", subset="char-shift-5")
    fewshot_source = DataSource(FORMAL_LANGS_REPO, split="train", subset="char-shift-5")


@register("formal_langs_k_dyck")
class FormalLanguageKDyck(FormalLanguageCompletion):
    data_source = DataSource(FORMAL_LANGS_REPO, split="test", subset="k-dyck")
    fewshot_source = DataSource(FORMAL_LANGS_REPO, split="train", subset="k-dyck")


@register("formal_langs_k_shuffle_dyck")
class FormalLanguageKShuffleDyck(FormalLanguageCompletion):
    data_source = DataSource(FORMAL_LANGS_REPO, split="test", subset="k-shuffle-dyck")
    fewshot_source = DataSource(FORMAL_LANGS_REPO, split="train", subset="k-shuffle-dyck")


@register("formal_langs_var_unique")
class FormalLanguageVarUnique(FormalLanguageCompletion):
    data_source = DataSource(FORMAL_LANGS_REPO, split="test", subset="var-unique")
    fewshot_source = DataSource(FORMAL_LANGS_REPO, split="train", subset="var-unique")


@register("formal_langs_var_undefined")
class FormalLanguageVarUndefined(FormalLanguageCompletion):
    data_source = DataSource(FORMAL_LANGS_REPO, split="test", subset="var-undefined")
    fewshot_source = DataSource(FORMAL_LANGS_REPO, split="train", subset="var-undefined")


@register("formal_langs_var_reassign_var")
class FormalLanguageVarReassignVar(FormalLanguageCompletion):
    data_source = DataSource(FORMAL_LANGS_REPO, split="test", subset="var-reassign-var")
    fewshot_source = DataSource(FORMAL_LANGS_REPO, split="train", subset="var-reassign-var")


@register("formal_langs_var_reassign_const")
class FormalLanguageVarReassignConst(FormalLanguageCompletion):
    data_source = DataSource(FORMAL_LANGS_REPO, split="test", subset="var-reassign-const")
    fewshot_source = DataSource(FORMAL_LANGS_REPO, split="train", subset="var-reassign-const")


class FormalLanguageCubeCompletion(FormalLanguageCompletion):
    num_fewshot = 5


@register("formal_langs_cube_unique")
class FormalLanguageCubeUnique(FormalLanguageCubeCompletion):
    data_source = DataSource(FORMAL_LANGS_REPO, split="test", subset="cube-unique")
    fewshot_source = DataSource(FORMAL_LANGS_REPO, split="train", subset="cube-unique")


@register("formal_langs_cube_undefined")
class FormalLanguageCubeUndefined(FormalLanguageCubeCompletion):
    data_source = DataSource(FORMAL_LANGS_REPO, split="test", subset="cube-undefined")
    fewshot_source = DataSource(FORMAL_LANGS_REPO, split="train", subset="cube-undefined")


@register("formal_langs_cube_reassign_var")
class FormalLanguageCubeReassignVar(FormalLanguageCubeCompletion):
    data_source = DataSource(FORMAL_LANGS_REPO, split="test", subset="cube-reassign-var")
    fewshot_source = DataSource(FORMAL_LANGS_REPO, split="train", subset="cube-reassign-var")


@register("formal_langs_cube_reassign_const")
class FormalLanguageCubeReassignConst(FormalLanguageCubeCompletion):
    data_source = DataSource(FORMAL_LANGS_REPO, split="test", subset="cube-reassign-const")
    fewshot_source = DataSource(FORMAL_LANGS_REPO, split="train", subset="cube-reassign-const")


def _register_binding_variable_count_tasks() -> None:
    task_classes = {
        "formal_langs_var_unique": FormalLanguageVarUnique,
        "formal_langs_var_undefined": FormalLanguageVarUndefined,
        "formal_langs_var_reassign_var": FormalLanguageVarReassignVar,
        "formal_langs_var_reassign_const": FormalLanguageVarReassignConst,
        "formal_langs_cube_unique": FormalLanguageCubeUnique,
        "formal_langs_cube_undefined": FormalLanguageCubeUndefined,
        "formal_langs_cube_reassign_var": FormalLanguageCubeReassignVar,
        "formal_langs_cube_reassign_const": FormalLanguageCubeReassignConst,
    }

    for task_name, base_class in task_classes.items():
        for num_variables in BINDING_VARIABLE_COUNTS:
            class_name = f"{base_class.__name__}V{num_variables}"
            cls = type(
                class_name,
                (base_class,),
                {
                    "__module__": __name__,
                    "__qualname__": class_name,
                    "num_variables": num_variables,
                },
            )
            globals()[class_name] = cls
            register(f"{task_name}:v{num_variables}")(cls)


def _register_dyck_parenthesis_type_tasks() -> None:
    task_classes = {
        "formal_langs_k_dyck": FormalLanguageKDyck,
        "formal_langs_k_shuffle_dyck": FormalLanguageKShuffleDyck,
    }

    for task_name, base_class in task_classes.items():
        for num_parenthesis_types in DYCK_PARENTHESIS_TYPE_COUNTS:
            class_name = f"{base_class.__name__}K{num_parenthesis_types}"
            cls = type(
                class_name,
                (base_class,),
                {
                    "__module__": __name__,
                    "__qualname__": class_name,
                    "num_parenthesis_types": num_parenthesis_types,
                },
            )
            globals()[class_name] = cls
            register(f"{task_name}:k{num_parenthesis_types}")(cls)


def _register_assignment_count_tasks() -> None:
    task_classes = {
        "formal_langs_var_unique": FormalLanguageVarUnique,
        "formal_langs_var_undefined": FormalLanguageVarUndefined,
        "formal_langs_var_reassign_var": FormalLanguageVarReassignVar,
        "formal_langs_var_reassign_const": FormalLanguageVarReassignConst,
        "formal_langs_cube_unique": FormalLanguageCubeUnique,
        "formal_langs_cube_undefined": FormalLanguageCubeUndefined,
        "formal_langs_cube_reassign_var": FormalLanguageCubeReassignVar,
        "formal_langs_cube_reassign_const": FormalLanguageCubeReassignConst,
    }

    for task_name, base_class in task_classes.items():
        for min_assignments, max_assignments in ASSIGNMENT_COUNT_BINS:
            class_name = f"{base_class.__name__}A{min_assignments}To{max_assignments}"
            cls = type(
                class_name,
                (base_class,),
                {
                    "__module__": __name__,
                    "__qualname__": class_name,
                    "assignment_count_range": (min_assignments, max_assignments),
                },
            )
            globals()[class_name] = cls
            register(f"{task_name}:a{min_assignments}-{max_assignments}")(cls)


_register_binding_variable_count_tasks()
_register_dyck_parenthesis_type_tasks()
_register_assignment_count_tasks()
