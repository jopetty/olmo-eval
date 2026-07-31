from __future__ import annotations

import random
from collections.abc import Iterator
from typing import Any

from olmo_eval.common.metrics import GreedyAccuracyMetric
from olmo_eval.common.types import Instance, LMRequest, RequestType
from olmo_eval.data import DataLoader, DataSource
from olmo_eval.evals.tasks.common import Task, register


def _format_example(doc: dict[str, Any]) -> str:
    return f"{doc['mutated']} -> {doc['original']}"


class Grapheme(Task):
    metrics = (GreedyAccuracyMetric(),)
    num_fewshot = 3

    @property
    def instances(self) -> Iterator[Instance]:
        if self._instances_cache is None:
            self._instances_cache = list(self._load_grapheme_instances())
        yield from self._instances_cache

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
        return rng.sample(candidates, min(self.config.num_fewshot, len(candidates)))

    def _load_grapheme_instances(self) -> Iterator[Instance]:
        loader = DataLoader()
        source = (
            self.config.data_source
            if isinstance(self.config.data_source, DataSource)
            else self.config.get_data_source()
        )
        docs = list(enumerate(loader.load(source)))

        for index, doc in docs:
            fewshot_docs = self._sample_fewshot_docs(docs, index)
            yield self.process_doc(doc, index, fewshot_docs=fewshot_docs)

    def process_doc(
        self,
        doc: dict[str, Any],
        index: int = 0,
        fewshot_docs: list[dict[str, Any]] | None = None,
    ) -> Instance:
        mutated = str(doc["mutated"])
        original = str(doc["original"])

        fewshot_examples = [_format_example(fewshot_doc) for fewshot_doc in fewshot_docs or []]
        prompt_parts = [*fewshot_examples, f"{mutated} ->"]
        question = "\n".join(prompt_parts)

        return Instance(
            question=question,
            gold_answer=original,
            choices=(original,),
            metadata={
                "id": doc.get("id", index),
                "index": index,
                "gold_idx": 0,
                "gold_text": original,
                "mutated": mutated,
                "fewshot_examples": fewshot_examples,
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        continuation = f" {instance.gold_answer}" if instance.gold_answer is not None else ""
        return LMRequest(
            request_type=RequestType.LOGLIKELIHOOD,
            prompt=instance.question,
            continuations=(continuation,),
        )


@register("grapheme_ar")
class GraphemeAR(Grapheme):
    data_source = DataSource("jacksonp-ai2/grapheme", split="test", subset="ar")


@register("grapheme_de")
class GraphemeDE(Grapheme):
    data_source = DataSource("jacksonp-ai2/grapheme", split="test", subset="de")


@register("grapheme_el")
class GraphemeEL(Grapheme):
    data_source = DataSource("jacksonp-ai2/grapheme", split="test", subset="el")


@register("grapheme_en")
class GraphemeEN(Grapheme):
    data_source = DataSource("jacksonp-ai2/grapheme", split="test", subset="en")


@register("grapheme_es")
class GraphemeES(Grapheme):
    data_source = DataSource("jacksonp-ai2/grapheme", split="test", subset="es")


@register("grapheme_he")
class GraphemeHE(Grapheme):
    data_source = DataSource("jacksonp-ai2/grapheme", split="test", subset="he")


@register("grapheme_hi")
class GraphemeHI(Grapheme):
    data_source = DataSource("jacksonp-ai2/grapheme", split="test", subset="hi")


@register("grapheme_ko")
class GraphemeKO(Grapheme):
    data_source = DataSource("jacksonp-ai2/grapheme", split="test", subset="ko")


@register("grapheme_nl")
class GraphemeNL(Grapheme):
    data_source = DataSource("jacksonp-ai2/grapheme", split="test", subset="nl")


@register("grapheme_pt")
class GraphemePT(Grapheme):
    data_source = DataSource("jacksonp-ai2/grapheme", split="test", subset="pt")


@register("grapheme_zh")
class GraphemeZH(Grapheme):
    data_source = DataSource("jacksonp-ai2/grapheme", split="test", subset="zh")
