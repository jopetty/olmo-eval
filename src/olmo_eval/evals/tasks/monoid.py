"""Monoid composition tasks.

Monoid compositions tasks are useful for evaluating the state-tracking
capabilities of language models. A model must solve the word problem for a
monoid conditioned on the monoid's Cayley table. The difficulty of the task is
modulated by:
    - the length of the sequence to compose
    - the number of elements in the group which derives the monoid
    - the complexity of the monoid's algebraic structure; at a high level,
      the easiest groups are those which are commutative (abeliean), followed by
      those which are non-commutative but still soluble, followed by
      those which are insoluble. Insoluble problems are provably "hard" for
      transformer-based language models.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from functools import cache
from itertools import islice
from typing import Any

from olmo_eval.common.metrics import (
    LogprobPerTokenMCAccuracyMetric,
)
from olmo_eval.common.types import Instance, LMRequest, RequestType
from olmo_eval.data import DataLoader, DataSource
from olmo_eval.evals.tasks.common import Task, register, register_variant

COMMUTATIVE_MONOIDS = [
    "monoid_s2",
    "monoid_a3",
    "monoid_z6",
    "monoid_z12",
    "monoid_z24",
    "monoid_z60",
    "monoid_z120",
    "monoid_z360",
    "monoid_z720",
]

NONCOMMUTATIVE_SOLUBLE_MONOIDS = [
    "monoid_s3",
    "monoid_s4",
    "monoid_a4",
]

INSOLUBLE_MONOIDS = [
    "monoid_s5",
    "monoid_s6",
    "monoid_a5",
    "monoid_a6",
]


@cache
def _get_cayley_table(group: str) -> Any:
    """Build the Cayley table for a supported group name."""
    try:
        from abstract_algebra.finite_algebras import (  # noqa: F401
            Group,
            generate_cyclic_group,
            generate_symmetric_group,
        )
    except ImportError as err:
        raise ImportError(
            "The monoid task requires abstract-algebra. "
            "It is declared as a task runtime dependency, but for local runs install "
            "'git+https://github.com/jopetty/abstract_algebra'."
        ) from err

    if group.startswith("Z"):
        algebra = generate_cyclic_group(int(group[1:]))
    elif group.startswith("S"):
        algebra = generate_symmetric_group(int(group[1:]))
    elif group.startswith("A"):
        symmetric_group = generate_symmetric_group(int(group[1:]))
        if not isinstance(symmetric_group, Group):
            raise ValueError(f"Expected a Group, got {type(symmetric_group).__name__}")
        algebra = symmetric_group.commutator_subalgebra()
        algebra.name = group
    else:
        raise ValueError(f"Unsupported monoid group: {group}")

    if not isinstance(algebra, Group):
        raise ValueError(f"Expected a Group, got {type(algebra).__name__}")
    return algebra.table.table


def _format_cayley_table(group: str, sep: str = ", ") -> str:
    table = _get_cayley_table(group)
    return sep.join(
        f"{left}*{right}={table[left, right]}"
        for left in range(table.shape[0])
        for right in range(table.shape[1])
    )


def _format_sequence(sequence: list[int]) -> str:
    return " * ".join(str(element) for element in sequence)


def _format_fewshot_doc(doc: dict[str, Any]) -> str:
    return f"{_format_sequence(doc['sequence'])} = {doc['product']}"


class MonoidComposition(Task):
    """Base class for monoid composition tasks."""

    data_source = DataSource("jowenpetty/monoids-100")
    dependencies = ["git+https://github.com/jopetty/abstract_algebra"]
    metrics = (LogprobPerTokenMCAccuracyMetric(),)
    num_fewshot = 0
    fewshot_seed = 42
    n_alternatives = 3
    n_samples_per_group = 100
    max_sequence_length = 10
    includes_cayley_table = True

    def _sample_excluding(
        self, n_samples: int, upper_bound: int, excluding: int, rng: random.Random
    ) -> list[int]:
        return rng.sample([i for i in range(upper_bound) if i != excluding], n_samples)

    def _sample_fewshot_docs(
        self,
        docs: list[tuple[int, dict[str, Any]]],
        index: int,
    ) -> list[dict[str, Any]]:
        if self.config.num_fewshot <= 0:
            return []

        candidates = [doc for doc_index, doc in docs if doc_index != index]
        if not candidates:
            return []

        rng = random.Random(self.config.fewshot_seed + index)
        n_fewshot = min(self.config.num_fewshot, len(candidates))
        return rng.sample(candidates, n_fewshot)

    def _load_group_instances(self) -> Iterator[Instance]:
        loader = DataLoader()
        source = (
            self.config.data_source
            if isinstance(self.config.data_source, DataSource)
            else self.config.get_data_source()
        )
        docs: list[tuple[int, dict[str, Any]]] = []

        for index, doc in enumerate(loader.load(source)):
            sequence = doc.get("sequence", ())
            if self.max_sequence_length is not None and len(sequence) > self.max_sequence_length:
                continue
            docs.append((index, doc))

        for index, doc in docs:
            fewshot_docs = self._sample_fewshot_docs(docs, index)
            yield self.process_doc(doc, index, fewshot_docs=fewshot_docs)

    @property
    def instances(self) -> Iterator[Instance]:
        if self._instances_cache is None:
            instances = self._load_group_instances()
            if self.n_samples_per_group is not None:
                instances = islice(instances, self.n_samples_per_group)
            self._instances_cache = list(instances)
        yield from self._instances_cache

    def process_doc(
        self,
        doc: dict[str, Any],
        index: int = 0,
        fewshot_docs: list[dict[str, Any]] | None = None,
    ) -> Instance:
        group = doc["group"]
        element_sequence = doc["sequence"]
        product = int(doc["product"])
        table = _get_cayley_table(group)
        group_size = int(table.shape[0])
        n_alternatives = min(self.n_alternatives, group_size - 1)

        rng = random.Random(index)
        alternative_choices = self._sample_excluding(n_alternatives, group_size, product, rng)
        choices = [product, *alternative_choices]
        rng.shuffle(choices)
        gold_idx = choices.index(product)

        fewshot_examples = [_format_fewshot_doc(fewshot_doc) for fewshot_doc in fewshot_docs or []]
        formatted_sequence = _format_sequence(element_sequence)
        if self.includes_cayley_table:
            cayley_table = _format_cayley_table(group)
            context_parts = [cayley_table, *fewshot_examples]
            question = f"Given {' and '.join(context_parts)} then {formatted_sequence} ="
        elif fewshot_examples:
            question = f"Given {' and '.join(fewshot_examples)} then {formatted_sequence} ="
        else:
            question = f"{formatted_sequence} ="

        return Instance(
            question=question,
            gold_answer=str(product),
            choices=tuple(str(choice) for choice in choices),
            metadata={
                "group": group,
                "group_size": group_size,
                "gold_idx": gold_idx,
                "gold_text": str(product),
                "index": index,
                "sequence": element_sequence,
                "fewshot_examples": fewshot_examples,
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        return LMRequest(
            request_type=RequestType.LOGLIKELIHOOD,
            prompt=instance.question,
            continuations=tuple(f" {choice}" for choice in (instance.choices or ())),
        )


@register("monoid_s2")
class MonoidCompositionS2(MonoidComposition):
    data_source = DataSource("jowenpetty/monoids-100", split="S2")


@register("monoid_s3")
class MonoidCompositionS3(MonoidComposition):
    data_source = DataSource("jowenpetty/monoids-100", split="S3")


@register("monoid_s4")
class MonoidCompositionS4(MonoidComposition):
    data_source = DataSource("jowenpetty/monoids-100", split="S4")


@register("monoid_s5")
class MonoidCompositionS5(MonoidComposition):
    data_source = DataSource("jowenpetty/monoids-100", split="S5")


@register("monoid_s6")
class MonoidCompositionS6(MonoidComposition):
    data_source = DataSource("jowenpetty/monoids-100", split="S6")


@register("monoid_a3")
class MonoidCompositionA3(MonoidComposition):
    data_source = DataSource("jowenpetty/monoids-100", split="A3")


@register("monoid_a4")
class MonoidCompositionA4(MonoidComposition):
    data_source = DataSource("jowenpetty/monoids-100", split="A4")


@register("monoid_a5")
class MonoidCompositionA5(MonoidComposition):
    data_source = DataSource("jowenpetty/monoids-100", split="A5")


@register("monoid_a6")
class MonoidCompositionA6(MonoidComposition):
    data_source = DataSource("jowenpetty/monoids-100", split="A6")


@register("monoid_z6")
class MonoidCompositionZ6(MonoidComposition):
    data_source = DataSource("jowenpetty/monoids-100", split="Z6")


@register("monoid_z12")
class MonoidCompositionZ12(MonoidComposition):
    data_source = DataSource("jowenpetty/monoids-100", split="Z12")


@register("monoid_z24")
class MonoidCompositionZ24(MonoidComposition):
    data_source = DataSource("jowenpetty/monoids-100", split="Z24")


@register("monoid_z60")
class MonoidCompositionZ60(MonoidComposition):
    data_source = DataSource("jowenpetty/monoids-100", split="Z60")


@register("monoid_z120")
class MonoidCompositionZ120(MonoidComposition):
    data_source = DataSource("jowenpetty/monoids-100", split="Z120")


@register("monoid_z360")
class MonoidCompositionZ360(MonoidComposition):
    data_source = DataSource("jowenpetty/monoids-100", split="Z360")


@register("monoid_z720")
class MonoidCompositionZ720(MonoidComposition):
    data_source = DataSource("jowenpetty/monoids-100", split="Z720")


register_variant("monoid_s2", "3shot", num_fewshot=3)
register_variant("monoid_s3", "3shot", num_fewshot=3)
register_variant("monoid_s4", "3shot", num_fewshot=3)
register_variant("monoid_s5", "3shot", num_fewshot=3)
register_variant("monoid_s6", "3shot", num_fewshot=3)

register_variant("monoid_a3", "3shot", num_fewshot=3)
register_variant("monoid_a4", "3shot", num_fewshot=3)
register_variant("monoid_a5", "3shot", num_fewshot=3)
register_variant("monoid_a6", "3shot", num_fewshot=3)

register_variant("monoid_z6", "3shot", num_fewshot=3)
register_variant("monoid_z12", "3shot", num_fewshot=3)
register_variant("monoid_z24", "3shot", num_fewshot=3)
register_variant("monoid_z60", "3shot", num_fewshot=3)
register_variant("monoid_z120", "3shot", num_fewshot=3)
register_variant("monoid_z360", "3shot", num_fewshot=3)
register_variant("monoid_z720", "3shot", num_fewshot=3)
