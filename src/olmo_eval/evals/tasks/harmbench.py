"""
HarmBench Safety Evaluation Task

This module implements the HarmBench Safety evaluation task.

Usage:

    olmo-eval run -m llama3.1-8b -t harmbench:openai
"""

import logging
from collections.abc import Iterator, Sequence
from typing import Any

from olmo_eval.common.formatters import ChatFormatter
from olmo_eval.common.metrics import AccuracyMetric
from olmo_eval.common.scorers import SafetyScorer
from olmo_eval.common.types import Instance, LMRequest, RequestType, Response, SamplingParams
from olmo_eval.data import DataLoader, DataSource
from olmo_eval.evals.extract import extract_think_answer
from olmo_eval.evals.tasks.common import Task, register, register_variant

logger = logging.getLogger(__name__)


@register("harmbench")
class Harmbench(Task):
    """Harmbench safety evaluation task."""

    data_source = DataSource("allenai/tulu-3-harmbench-eval")
    formatter = ChatFormatter()
    sampling_params = SamplingParams(temperature=0.6, top_p=0.95)

    @property
    def instances(self) -> Iterator[Instance]:
        """Yield instances from the dataset."""
        if self._instances_cache is None:
            self._instances_cache = []
            loader = DataLoader()
            source = self.config.get_data_source()

            for idx, doc in enumerate(loader.load(source)):
                instance = self.process_doc(doc, idx)
                if instance is not None:
                    self._instances_cache.append(instance)

        yield from self._instances_cache

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance:
        """Convert a dataset document to an Instance."""

        functional_cat = doc.get("FunctionalCategory")
        semantic_cat = doc.get("SemanticCategory")

        if functional_cat == "contextual":
            question = doc.get("ContextString") + "\n\n---\n\n" + doc.get("Behavior")
        else:
            question = doc.get("Behavior")

        return Instance(
            question=question,
            metadata={
                "id": doc.get("BehaviorID"),
                "index": index,
                "functional_cat": functional_cat,
                "semantic_cat": semantic_cat,
                "gold_answer": "no",
            },
        )

    @property
    def request_type(self) -> RequestType:
        """Return the request type for this task."""
        if self.config.formatter is not None:
            return self.config.formatter.request_type
        return RequestType.CHAT

    def format_request(self, instance: Instance) -> LMRequest:
        """Format an instance into an LM request.

        Delegates to the configured formatter (ChatFormatter by default).
        """
        if self.config.formatter is not None:
            return self.config.formatter.format(instance)
        # Fallback: create a simple chat request
        return LMRequest(
            request_type=RequestType.CHAT,
            messages=({"role": "user", "content": instance.question},),
        )

    def _extract_answers(self, responses: Sequence[Response]) -> None:
        """Extract the answers from model. Reasoning logic from oe-eval"""
        print(self.config.name)
        for response in responses:
            for output in response.outputs:
                output.extracted_answer = extract_think_answer(self.extract_answer(output))


# =============================================================================
# Variant Registrations
# =============================================================================

# Judge variant - uses LLM-as-judge scoring for factual accuracy
register_variant(
    "harmbench",
    "openai",
    metrics=(AccuracyMetric(scorer=SafetyScorer),),
)
