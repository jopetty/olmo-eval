from collections.abc import Iterator
from typing import Any

from olmo_eval.common.formatters import ChatFormatter
from olmo_eval.common.metrics import AccuracyMetric, PassAtKMetric
from olmo_eval.common.scorers import MinervaMathScorer
from olmo_eval.common.types import Instance, LMOutput, LMRequest, SamplingParams
from olmo_eval.data import DataSource
from olmo_eval.evals.extract import MathExtractor
from olmo_eval.evals.tasks.common import Task, register, register_variant

AIME_SYSTEM_PROMPT = (
    "You are a helpful assistant that solves math competition problems. "
    "Show your work step-by-step, then put your final integer answer in \\boxed{}."
)

ZS_COT_R1_SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


@register("aime")
class AIMETask(Task):
    data_source = DataSource(path="allenai/aime-2021-2025")
    formatter = ChatFormatter(
        system_prompt=AIME_SYSTEM_PROMPT,
        user_template="{question}",
    )
    metrics = (AccuracyMetric(scorer=MinervaMathScorer),)
    num_fewshot = 0
    sampling_params = SamplingParams(
        max_tokens=16384,
        temperature=0.0,
    )
    dependencies = ["lark>=1.0"]

    # Default: all years
    years: list[int] | None = None

    @property
    def instances(self) -> Iterator[Instance]:
        yield from self._load_instances_cached()

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        year = doc.get("year")
        if self.years is not None and year not in self.years:
            return None

        question = doc["problem"]
        gold_answer = str(doc["answer"])

        # AIME answers are integers 0-999; normalize by stripping leading zeros
        gold_normalized = gold_answer.lstrip("0") or "0"

        return Instance(
            question=question,
            gold_answer=gold_normalized,
            metadata={
                "id": doc.get("id", index),
                "year": year,
                "problem_number": doc.get("problem_number"),
                "all_gold_answers": [gold_normalized],
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        assert self.config.formatter is not None
        return self.config.formatter.format(instance, self.get_fewshot())

    def extract_answer(self, output: LMOutput) -> str | None:
        answers = MathExtractor.extract_answer(output.text)
        # Store all extracted answers in metadata for flexible scorers
        output.metadata["all_extracted_answers"] = answers if answers else []
        return answers[0] if answers else None


# --- Variant: zs_cot_r1 (zero-shot chain-of-thought, R1-style) ---
# Base R1-style reasoning variant
register_variant(
    "aime",
    "zs_cot_r1",
    formatter=ChatFormatter(
        system_prompt=ZS_COT_R1_SYSTEM_PROMPT,
        user_template="{question}",
    ),
    sampling_params=SamplingParams(
        max_tokens=16384,
        temperature=0.6,
        top_p=0.95,
    ),
)


# --- Year-filtered task classes ---
@register("aime_2024")
class AIME2024Task(AIMETask):
    years = [2024]


@register("aime_2025")
class AIME2025Task(AIMETask):
    years = [2025]


# --- pass_at_32 variants for rlzero evaluation ---
_PASS_AT_32_METRICS = (
    AccuracyMetric(scorer=MinervaMathScorer),
    PassAtKMetric(k=1, scorer=MinervaMathScorer),
    PassAtKMetric(k=4, scorer=MinervaMathScorer),
    PassAtKMetric(k=8, scorer=MinervaMathScorer),
    PassAtKMetric(k=16, scorer=MinervaMathScorer),
    PassAtKMetric(k=32, scorer=MinervaMathScorer),
)

_PASS_AT_32_SAMPLING = SamplingParams(
    max_tokens=16384,
    temperature=0.6,
    top_p=0.95,
    num_samples=32,
)

_PASS_AT_32_R1_FORMATTER = ChatFormatter(
    system_prompt=ZS_COT_R1_SYSTEM_PROMPT,
    user_template="{question}",
)

# aime:zs_cot_r1::pass_at_32_2024_rlzero
# Accessed as: aime_2024:zs_cot_r1:pass_at_32_rlzero
register_variant(
    "aime_2024",
    "zs_cot_r1",
    formatter=_PASS_AT_32_R1_FORMATTER,
    sampling_params=SamplingParams(
        max_tokens=16384,
        temperature=0.6,
        top_p=0.95,
    ),
)

register_variant(
    "aime_2024",
    "pass_at_32_rlzero",
    formatter=_PASS_AT_32_R1_FORMATTER,
    metrics=_PASS_AT_32_METRICS,
    primary_metric=PassAtKMetric(k=1, scorer=MinervaMathScorer),
    sampling_params=_PASS_AT_32_SAMPLING,
)

# aime:zs_cot_r1::pass_at_32_2025_rlzero
# Accessed as: aime_2025:zs_cot_r1:pass_at_32_rlzero
register_variant(
    "aime_2025",
    "zs_cot_r1",
    formatter=_PASS_AT_32_R1_FORMATTER,
    sampling_params=SamplingParams(
        max_tokens=16384,
        temperature=0.6,
        top_p=0.95,
    ),
)

register_variant(
    "aime_2025",
    "pass_at_32_rlzero",
    formatter=_PASS_AT_32_R1_FORMATTER,
    metrics=_PASS_AT_32_METRICS,
    primary_metric=PassAtKMetric(k=1, scorer=MinervaMathScorer),
    sampling_params=_PASS_AT_32_SAMPLING,
)

# --- pass_at_32 variants for deepseek evaluation ---
# Same config as rlzero (temp=0.6, top_p=0.95, 32 samples, pass@k)
register_variant(
    "aime_2024",
    "pass_at_32_deepseek",
    formatter=_PASS_AT_32_R1_FORMATTER,
    metrics=_PASS_AT_32_METRICS,
    primary_metric=PassAtKMetric(k=1, scorer=MinervaMathScorer),
    sampling_params=_PASS_AT_32_SAMPLING,
)

register_variant(
    "aime_2025",
    "pass_at_32_deepseek",
    formatter=_PASS_AT_32_R1_FORMATTER,
    metrics=_PASS_AT_32_METRICS,
    primary_metric=PassAtKMetric(k=1, scorer=MinervaMathScorer),
    sampling_params=_PASS_AT_32_SAMPLING,
)
