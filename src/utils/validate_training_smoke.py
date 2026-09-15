"""Validate that a bounded distributed training smoke test produced a real adapter."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


class SmokeValidationError(RuntimeError):
    """Raised when a training smoke test is incomplete, inert, or too slow."""


def validate_smoke(
    metadata_path: str | Path,
    adapter_dir: str | Path,
    expected_world_size: int,
    max_seconds_per_step: float,
) -> dict[str, float | int | str]:
    metadata_file = Path(metadata_path)
    adapter_path = Path(adapter_dir)
    if not metadata_file.is_file():
        raise SmokeValidationError(f"Missing smoke metadata: {metadata_file}")
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    world_size = int(metadata.get("distributed", {}).get("world_size", 0))
    if world_size != expected_world_size:
        raise SmokeValidationError(
            f"Smoke used {world_size} workers, expected {expected_world_size}."
        )
    if not bool(metadata.get("distributed", {}).get("fsdp", False)):
        raise SmokeValidationError("Smoke did not activate FSDP.")
    optimizer_steps = int(metadata.get("optimizer_steps", 0))
    if optimizer_steps < 1:
        raise SmokeValidationError("Smoke completed no optimizer steps.")
    metrics = metadata.get("training_metrics", {})
    runtime = float(metrics.get("train_runtime", 0.0))
    loss = float(metrics.get("train_loss", math.nan))
    if runtime <= 0.0 or not math.isfinite(loss):
        raise SmokeValidationError(
            f"Invalid smoke metrics: train_runtime={runtime}, train_loss={loss}."
        )
    seconds_per_step = runtime / optimizer_steps
    startup_allowance = 3600.0
    guarded_seconds_per_step = max(0.0, runtime - startup_allowance) / optimizer_steps
    if guarded_seconds_per_step > max_seconds_per_step:
        raise SmokeValidationError(
            f"Smoke took {guarded_seconds_per_step:.1f} guarded seconds per optimizer "
            f"step after the {startup_allowance:.0f}-second distributed-startup allowance, "
            f"above the {max_seconds_per_step:.1f}-second guard."
        )
    if not (adapter_path / "adapter_config.json").is_file():
        raise SmokeValidationError(f"Missing adapter_config.json in {adapter_path}")
    weights = list(adapter_path.rglob("adapter_model*.safetensors"))
    weights.extend(adapter_path.rglob("adapter_model*.bin"))
    if not weights or not any(path.stat().st_size > 0 for path in weights):
        raise SmokeValidationError(f"No non-empty adapter weights found in {adapter_path}")
    return {
        "metadata": str(metadata_file),
        "adapter_dir": str(adapter_path),
        "world_size": world_size,
        "optimizer_steps": optimizer_steps,
        "train_loss": loss,
        "train_runtime_seconds": runtime,
        "seconds_per_optimizer_step": seconds_per_step,
        "guarded_seconds_per_optimizer_step": guarded_seconds_per_step,
        "distributed_startup_allowance_seconds": startup_allowance,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--expected-world-size", type=int, default=4)
    parser.add_argument("--max-seconds-per-step", type=float, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            validate_smoke(
                args.metadata,
                args.adapter_dir,
                args.expected_world_size,
                args.max_seconds_per_step,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
