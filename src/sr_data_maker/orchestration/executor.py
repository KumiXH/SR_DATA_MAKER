from __future__ import annotations

from pathlib import Path
from typing import Any

from sr_data_maker.config.validator import validate_config
from sr_data_maker.dataset.writer import DatasetWriter
from sr_data_maker.orchestration.state_store import RunStateStore
from sr_data_maker.plugins import GENERATORS, RUNNERS, SOURCE_READERS, register_builtins


class PipelineExecutor:
    def run(self, config: dict[str, Any]) -> dict[str, int]:
        validate_config(config)
        register_builtins()

        output_root = Path(config["paths"]["output_root"])
        writer = DatasetWriter(output_root=output_root)
        state = RunStateStore(output_root=output_root)
        source_reader = SOURCE_READERS.build(config["source"], root=config["paths"]["input_root"])
        task_handlers = self._build_task_handlers(config["tasks"], config["runtime"])

        summary = {"succeeded": 0, "skipped_tasks": 0, "failed": 0}
        for source in source_reader.iter_sources():
            for task, generator in task_handlers:
                if not task.get("enabled", True):
                    summary["skipped_tasks"] += 1
                    continue

                key = f"{task['name']}::{source.rel_path}"
                if config["runtime"].get("resume") and state.has(key):
                    summary["skipped_tasks"] += 1
                    continue

                try:
                    samples = generator.generate(source, context=None)
                    for sample in samples:
                        writer.write_image(sample.output_path, sample.image)
                        writer.append_sample(sample.manifest)
                    state.add(key)
                    summary["succeeded"] += len(samples)
                except Exception as exc:  # pragma: no cover - exercised later by failure cases
                    writer.append_failure({"task": task["name"], "source": source.rel_path, "error": str(exc)})
                    summary["failed"] += 1

        writer.write_summary(summary)
        return summary

    def _build_task_handlers(self, tasks: list[dict[str, Any]], runtime: dict[str, Any]) -> list[tuple[dict[str, Any], Any]]:
        handlers: list[tuple[dict[str, Any], Any]] = []
        for task in tasks:
            if task.get("enabled", True):
                runner = self._build_runner(task, runtime)
                generator = self._build_generator(task, runner)
                handlers.append((task, generator))
                continue
            handlers.append((task, None))
        return handlers

    @staticmethod
    def _build_runner(task: dict[str, Any], runtime: dict[str, Any]):
        runner_config = dict(task["runner"])
        runner_type = runner_config.pop("type")
        params = {}
        params.update(task.get("degradation", {}))
        params.update(task.get("model", {}))
        if runner_type == "RealESRGANRunner" and "device" not in params and runtime.get("device"):
            params["device"] = runtime["device"]
        return RUNNERS.build({"type": runner_type, **params})

    @staticmethod
    def _build_generator(task: dict[str, Any], runner: Any):
        generator_type = task["type"]
        params = {"name": task["name"], "runner": runner, "output": task.get("output", {})}
        if "model" in task:
            params["model"] = task["model"]
        return GENERATORS.build({"type": generator_type, **params})
