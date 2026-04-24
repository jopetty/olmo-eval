"""SciCode external evaluation.

Implements the SciCode sub-step accuracy benchmark as an ``ExternalEval`` so
that sequential per-sub-step generation runs against the same ``InferenceProvider``
that served the main request — no auxiliary provider required.

Reference: https://scicode-bench.github.io/
Dataset: https://huggingface.co/datasets/SciCode1/SciCode
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from olmo_eval.common.types import LMRequest, RequestType, SamplingParams
from olmo_eval.evals.external.base import ExternalEval
from olmo_eval.evals.external.result import ExternalEvalResult

from . import loader as scicode_loader
from . import prompts as scicode_prompts
from . import verifier as scicode_verifier

if TYPE_CHECKING:
    from olmo_eval.inference.base import InferenceProvider


DEFAULT_H5PY_HOST_PATH = "/weka/oe-adapt-default/finbarrt/scicode/test_data.h5"
DEFAULT_H5PY_CONTAINER_PATH = "/workspace/scicode_test_data.h5"

_SANDBOX_LOCK_DIR = Path(__file__).parent / "sandbox"
_SANDBOX_PYPROJECT = (_SANDBOX_LOCK_DIR / "pyproject.toml").read_text()
_SANDBOX_UV_LOCK = (_SANDBOX_LOCK_DIR / "uv.lock").read_text()


@dataclass
class SciCodeArgs:
    """Arguments for SciCode external evaluation."""

    split: str = "test"
    problem_ids: list[str] | None = None
    with_background: bool = True
    enable_thinking: bool = False
    max_tokens: int = 16384
    temperature: float = 0.6
    max_concurrency: int = 4
    command_timeout: float = 600.0
    startup_timeout: float = 300.0
    h5py_host_path: str = DEFAULT_H5PY_HOST_PATH
    h5py_container_path: str = DEFAULT_H5PY_CONTAINER_PATH
    sandbox_image: str = "ghcr.io/astral-sh/uv:python3.12-bookworm-slim"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SciCodeArgs:
        problem_ids = data.get("problem_ids")
        if isinstance(problem_ids, str):
            problem_ids = [p.strip() for p in problem_ids.split(",") if p.strip()]
        return cls(
            split=data.get("split", "test"),
            problem_ids=problem_ids,
            with_background=bool(data.get("with_background", True)),
            enable_thinking=bool(data.get("enable_thinking", False)),
            max_tokens=int(data.get("max_tokens", 16384)),
            temperature=float(data.get("temperature", 0.6)),
            max_concurrency=int(data.get("max_concurrency", 4)),
            command_timeout=float(data.get("command_timeout", 600.0)),
            startup_timeout=float(data.get("startup_timeout", 300.0)),
            h5py_host_path=data.get("h5py_host_path", DEFAULT_H5PY_HOST_PATH),
            h5py_container_path=data.get("h5py_container_path", DEFAULT_H5PY_CONTAINER_PATH),
            sandbox_image=data.get(
                "sandbox_image", "ghcr.io/astral-sh/uv:python3.12-bookworm-slim"
            ),
        )


@dataclass
class _ProblemResult:
    problem_id: str
    problem_name: str
    step_results: list[bool]
    step_codes: dict[int, str]
    step_texts: dict[int, str]
    total_scorable: int

    @property
    def passed(self) -> int:
        return sum(1 for r in self.step_results if r)

    @property
    def all_passed(self) -> bool:
        return self.total_scorable > 0 and self.passed == self.total_scorable


class SciCodeExternalEval(ExternalEval):
    """SciCode multi-step scientific code generation as an external evaluation.

    For each main problem, generates code for each sub-step sequentially via the
    provided ``InferenceProvider`` (hardcoded snippets for problems 13.6, 62.1,
    and 76.3 are injected verbatim without generation). All generated code is
    then concatenated and tested one sub-step at a time in a Python sandbox
    that has ``numpy``, ``scipy``, ``sympy``, ``h5py``, and the SciCode numeric
    reference ``test_data.h5`` available.
    """

    @property
    def name(self) -> str:
        return "scicode"

    @property
    def description(self) -> str:
        return "SciCode: multi-step scientific code generation (65 problems, 288 sub-steps)."

    @property
    def timeout_seconds(self) -> float:
        return 36000.0

    @property
    def arguments(self) -> dict[str, tuple[str, Any | None]]:
        return {
            "split": ("Dataset split (test or validation)", "test"),
            "problem_ids": ("Comma-separated problem IDs (default: all)", None),
            "with_background": ("Inject scientist-annotated step backgrounds", True),
            "enable_thinking": (
                "Set chat_template_kwargs.enable_thinking=true on the provider",
                False,
            ),
            "max_tokens": ("Max generation tokens per sub-step", 16384),
            "temperature": ("Sampling temperature", 0.6),
            "max_concurrency": ("Max parallel problems", 4),
            "command_timeout": ("Per-sub-step sandbox execution timeout", 600.0),
            "h5py_host_path": ("Host path to SciCode test_data.h5", DEFAULT_H5PY_HOST_PATH),
            "h5py_container_path": (
                "Mount path inside the sandbox",
                DEFAULT_H5PY_CONTAINER_PATH,
            ),
        }

    async def execute(
        self,
        provider: InferenceProvider,
        args: dict[str, Any],
        output_dir: str | None = None,
        container_runtime: str = "podman",
    ) -> ExternalEvalResult:
        start_time = time.time()
        sc_args = SciCodeArgs.from_dict(args)

        if sc_args.enable_thinking:
            provider.chat_template_kwargs = {
                **(provider.chat_template_kwargs or {}),
                "enable_thinking": True,
            }

        problems = scicode_loader.load_problems(
            split=sc_args.split, problem_ids=sc_args.problem_ids
        )
        if not problems:
            return self._error_result(
                "No SciCode problems loaded",
                start_time,
                raw_output=(f"split={sc_args.split}, problem_ids={sc_args.problem_ids}"),
            )

        sampling_params = SamplingParams(
            max_tokens=sc_args.max_tokens, temperature=sc_args.temperature
        )

        semaphore = asyncio.Semaphore(sc_args.max_concurrency)

        async def run_problem(problem: scicode_loader.SciCodeProblem) -> _ProblemResult:
            async with semaphore:
                return await self._run_problem(
                    problem=problem,
                    provider=provider,
                    sampling_params=sampling_params,
                    sc_args=sc_args,
                    container_runtime=container_runtime,
                )

        problem_results: list[_ProblemResult] = await asyncio.gather(
            *[run_problem(p) for p in problems]
        )

        total_sub = sum(pr.total_scorable for pr in problem_results)
        passed_sub = sum(pr.passed for pr in problem_results)
        main_passed = sum(1 for pr in problem_results if pr.all_passed)

        metrics: dict[str, float] = {
            "sub_step_accuracy": (passed_sub / total_sub) if total_sub else 0.0,
            "main_problem_accuracy": (
                (main_passed / len(problem_results)) if problem_results else 0.0
            ),
            "num_problems": float(len(problem_results)),
            "num_sub_steps": float(total_sub),
            "passed_sub_steps": float(passed_sub),
        }

        predictions = [
            {
                "problem_id": pr.problem_id,
                "problem_name": pr.problem_name,
                "passed": pr.passed,
                "total": pr.total_scorable,
                "all_passed": pr.all_passed,
                "step_results": pr.step_results,
                "step_codes": pr.step_codes,
                "step_texts": pr.step_texts,
            }
            for pr in problem_results
        ]

        result = ExternalEvalResult(
            name=self.name,
            success=True,
            metrics=metrics,
            metadata={
                "model_name": getattr(provider, "model_name", None),
                "split": sc_args.split,
                "with_background": sc_args.with_background,
                "max_tokens": sc_args.max_tokens,
                "temperature": sc_args.temperature,
            },
            duration_seconds=time.time() - start_time,
            predictions=predictions,
        )

        if output_dir:
            self._save_results(result, output_dir)
        return result

    async def _run_problem(
        self,
        problem: scicode_loader.SciCodeProblem,
        provider: InferenceProvider,
        sampling_params: SamplingParams,
        sc_args: SciCodeArgs,
        container_runtime: str,
    ) -> _ProblemResult:
        hardcoded = scicode_prompts.hardcoded_for_problem(problem.problem_id)
        sub_steps = problem.sub_steps
        previous_llm_code: list[str | None] = [None] * len(sub_steps)
        for idx, snippet in hardcoded.items():
            if idx < len(sub_steps):
                previous_llm_code[idx] = snippet

        step_codes: dict[int, str] = {}
        step_texts: dict[int, str] = {}

        for idx in range(len(sub_steps)):
            if idx in hardcoded:
                continue
            prompt = scicode_prompts.build_step_prompt(
                sub_steps=sub_steps,
                required_dependencies=problem.required_dependencies,
                step_idx=idx,
                previous_llm_code=previous_llm_code,
                with_background=sc_args.with_background,
            )
            request = LMRequest(
                request_type=RequestType.CHAT,
                messages=({"role": "user", "content": prompt},),
            )
            results = await provider.agenerate([request], sampling_params)
            text = results[0][0].text
            code = scicode_prompts.extract_step_code(text)
            step_texts[idx] = text
            step_codes[idx] = code
            previous_llm_code[idx] = code

        scorable_indices = [i for i in range(len(sub_steps)) if i not in hardcoded]
        total_scorable = len(scorable_indices)

        if total_scorable == 0:
            return _ProblemResult(
                problem_id=problem.problem_id,
                problem_name=problem.problem_name,
                step_results=[],
                step_codes=step_codes,
                step_texts=step_texts,
                total_scorable=0,
            )

        hardcoded_prelude = "\n\n".join(
            hardcoded[i] for i in sorted(hardcoded) if i < len(sub_steps)
        )
        full_code = "\n\n".join(step_codes[i] for i in sorted(step_codes) if step_codes[i])

        step_results = await self._verify(
            problem=problem,
            scorable_indices=scorable_indices,
            full_code=full_code,
            hardcoded_prelude=hardcoded_prelude,
            sc_args=sc_args,
            container_runtime=container_runtime,
        )

        return _ProblemResult(
            problem_id=problem.problem_id,
            problem_name=problem.problem_name,
            step_results=step_results,
            step_codes=step_codes,
            step_texts=step_texts,
            total_scorable=total_scorable,
        )

    async def _verify(
        self,
        problem: scicode_loader.SciCodeProblem,
        scorable_indices: list[int],
        full_code: str,
        hardcoded_prelude: str,
        sc_args: SciCodeArgs,
        container_runtime: str,
    ) -> list[bool]:
        from olmo_eval.harness.sandbox import SandboxManager
        from olmo_eval.harness.sandbox.config import (
            Capability,
            ContainerRuntime,
            SandboxConfig,
            SandboxMode,
        )

        runtime = cast(ContainerRuntime, container_runtime)
        sandbox_config = SandboxConfig(
            image=sc_args.sandbox_image,
            mode=SandboxMode.DOCKER,
            container_runtime=runtime,
            capabilities=Capability.PYTHON,
            startup_timeout=sc_args.startup_timeout,
            command_timeout=sc_args.command_timeout,
            inject_swerex=True,
            dockerfile_extra=_sandbox_dockerfile_extra(),
            volumes=((sc_args.h5py_host_path, sc_args.h5py_container_path),),
        )
        sandbox_manager = SandboxManager([sandbox_config], owner=f"scicode-{problem.problem_id}")

        results: list[bool] = []
        try:
            await sandbox_manager.start()
            for idx in scorable_indices:
                step = problem.sub_steps[idx]
                script = scicode_verifier.build_step_script(
                    step=step,
                    required_dependencies=problem.required_dependencies,
                    full_code=full_code,
                    hardcoded_prelude=hardcoded_prelude,
                    h5py_file=sc_args.h5py_container_path,
                )
                exec_result = await sandbox_manager.execute_code(
                    script, language="python", timeout=sc_args.command_timeout
                )
                results.append(bool(exec_result.success))
        finally:
            await sandbox_manager.stop()
        return results


def _sandbox_dockerfile_extra() -> tuple[str, ...]:
    """Build Dockerfile steps that install SciCode sandbox deps from uv.lock.

    Materializes the checked-in ``pyproject.toml`` and ``uv.lock`` inside the
    image, exports the lock to a pinned requirements file via ``uv export``,
    then installs into the existing ``/root/python`` standalone interpreter
    (which is not a venv, so ``uv sync`` can't target it directly). Embedding
    the lock contents in the Dockerfile string means any lock change
    invalidates the swerex image cache automatically.
    """
    return (
        "WORKDIR /opt/scicode",
        f"RUN cat > pyproject.toml <<'SCICODE_PYPROJECT_EOF'\n"
        f"{_SANDBOX_PYPROJECT}"
        f"SCICODE_PYPROJECT_EOF",
        f"RUN cat > uv.lock <<'SCICODE_UVLOCK_EOF'\n{_SANDBOX_UV_LOCK}SCICODE_UVLOCK_EOF",
        "RUN /root/python/bin/uv export --frozen --no-dev --no-hashes "
        "--format requirements-txt -o /tmp/scicode-requirements.txt",
        "RUN /root/python/bin/uv pip install --system --no-cache -r /tmp/scicode-requirements.txt",
    )
