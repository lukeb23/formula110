from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from racing.game import controller_trials


def test_simulator_command_records_the_requested_controller() -> None:
    command = controller_trials.simulator_command(
        output=Path("artifacts/controller.jsonl"),
        seed=110,
        student_module="path/to/controller.py",
        racing_args=("--", "--muted"),
    )

    assert command[1:] == [
        "-m",
        "racing",
        "--seed",
        "110",
        "--student-module",
        "path/to/controller.py",
        "--control-function",
        "control",
        "--record-controller",
        "artifacts/controller.jsonl",
        "--muted",
    ]


def test_capture_trials_creates_controller_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_dir = tmp_path / "controller-driving"
    controller_path = tmp_path / "controller.py"
    controller_path.write_text("def control(sensors): pass\n", encoding="utf-8")
    monkeypatch.setattr(controller_trials, "_new_run_id", lambda: "run-test")

    def fake_run(command: list[str], *, check: bool) -> subprocess.CompletedProcess[str]:
        trial_path = Path(command[command.index("--record-controller") + 1])
        trial_path.write_text(
            json.dumps(
                {
                    "record_type": "controller_control_step",
                    "session_id": "controller-session",
                    "control_source": {
                        "type": "student_controller",
                        "module": str(controller_path),
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(controller_trials.subprocess, "run", fake_run)

    summaries = controller_trials.capture_trials(
        trials=1,
        seed=110,
        output_dir=output_dir,
        student_module=str(controller_path),
    )

    assert summaries[0].path.name == "trial-001-seed-110.jsonl"
    manifest = json.loads((output_dir / "run-test" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["record_type"] == "controller_driving_run"
    assert manifest["control_source"]["type"] == "student_controller"
    assert manifest["control_source"]["module"] == str(controller_path)
    assert manifest["control_source"]["function"] == "control"
    assert len(manifest["control_source"]["sha256"]) == 64
    assert manifest["trials"][0]["session_ids"] == ["controller-session"]
