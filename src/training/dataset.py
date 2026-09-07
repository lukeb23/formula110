"""Load append-only Formula 110 demonstrations through the shared encoder."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from controllers.observation import OBSERVATION_SIZE, encode_observation


@dataclass(frozen=True)
class DemonstrationDataset:
    observations: NDArray[np.float32]
    actions: NDArray[np.float32]
    session_ids: tuple[str, ...]
    source_paths: tuple[str, ...]


def load_demonstrations(paths: list[Path]) -> DemonstrationDataset:
    """Load human observation/action rows; actions use [steer, throttle]."""
    observations: list[NDArray[np.float32]] = []
    actions: list[tuple[float, float]] = []
    sessions: list[str] = []
    source_paths: list[str] = []
    for path in sorted({item.resolve() for item in paths}):
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                loaded: object = json.loads(line)
                if not isinstance(loaded, dict):
                    raise ValueError(f"malformed record at {path}:{line_number}")
                record = cast(dict[str, object], loaded)
                if record.get("record_type") != "human_control_step":
                    continue
                sensors = record.get("sensors")
                command = record.get("command")
                if not isinstance(sensors, dict) or not isinstance(command, dict):
                    raise ValueError(f"malformed record at {path}:{line_number}")
                typed_sensors = cast(dict[str, object], sensors)
                typed_command = cast(dict[str, object], command)
                steer = _finite_action(typed_command.get("steer"), path, line_number)
                throttle = _finite_action(typed_command.get("throttle"), path, line_number)
                observations.append(encode_observation(typed_sensors))
                actions.append((steer, throttle))
                sessions.append(str(record.get("session_id", "")))
        source_paths.append(str(path))
    if not observations:
        raise ValueError("no human_control_step records found")
    return DemonstrationDataset(
        observations=np.stack(observations).astype(np.float32, copy=False).reshape(-1, OBSERVATION_SIZE),
        actions=np.asarray(actions, dtype=np.float32),
        session_ids=tuple(sessions),
        source_paths=tuple(source_paths),
    )


def discover_human_trials(root: Path) -> list[Path]:
    return sorted(root.glob("**/trial-*.jsonl"))


def _finite_action(value: object, path: Path, line_number: int) -> float:
    try:
        result = float(cast(float | int | str, value))
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid action at {path}:{line_number}") from error
    if not np.isfinite(result):
        raise ValueError(f"non-finite action at {path}:{line_number}")
    return max(-1.0, min(1.0, result))
