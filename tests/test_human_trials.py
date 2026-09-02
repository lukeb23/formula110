from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from racing.game import human_trials


def test_simulator_command_uses_current_python_and_requested_recording() -> None:
    command = human_trials.simulator_command(
        output=Path("artifacts/demos.jsonl"),
        seed=110,
        racing_args=("--", "--muted"),
    )

    assert command[1:] == [
        "-m",
        "racing",
        "--seed",
        "110",
        "--record-human",
        "artifacts/demos.jsonl",
        "--muted",
    ]


def test_capture_trials_creates_separate_files_and_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "human-driving"
    calls: list[list[str]] = []
    monkeypatch.setattr(human_trials, "_new_run_id", lambda: "run-test")

    def fake_run(command: list[str], *, check: bool) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        trial = len(calls)
        trial_path = Path(command[command.index("--record-human") + 1])
        record = {
            "record_type": "human_control_step",
            "session_id": f"session-{trial}",
            "sensors": {"tick": 0},
            "command": {"throttle": 1.0, "steer": 0.0},
        }
        with trial_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(human_trials.subprocess, "run", fake_run)

    summaries = human_trials.capture_trials(trials=2, seed=110, output_dir=output_dir)

    run_dir = output_dir / "run-test"
    assert [summary.path.name for summary in summaries] == [
        "trial-001-seed-110.jsonl",
        "trial-002-seed-110.jsonl",
    ]
    assert [summary.session_ids for summary in summaries] == [("session-1",), ("session-2",)]
    assert all(summary.path.read_text(encoding="utf-8").count("\n") == 1 for summary in summaries)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["requested_trials"] == 2
    assert [trial["path"] for trial in manifest["trials"]] == [summary.path.name for summary in summaries]
    assert [trial["record_count"] for trial in manifest["trials"]] == [1, 1]


def test_capture_trials_can_vary_seed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(human_trials, "_new_run_id", lambda: "run-test")

    def fake_run(command: list[str], *, check: bool) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(human_trials.subprocess, "run", fake_run)

    summaries = human_trials.capture_trials(
        trials=3,
        seed=110,
        output_dir=tmp_path / "human-driving",
        vary_seed=True,
    )

    seeds = [command[command.index("--seed") + 1] for command in commands]
    assert seeds == ["110", "111", "112"]
    assert [summary.path.name for summary in summaries] == [
        "trial-001-seed-110.jsonl",
        "trial-002-seed-111.jsonl",
        "trial-003-seed-112.jsonl",
    ]


def test_failed_trial_is_recorded_in_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_dir = tmp_path / "human-driving"
    monkeypatch.setattr(human_trials, "_new_run_id", lambda: "run-test")
    monkeypatch.setattr(
        human_trials.subprocess,
        "run",
        lambda command, check: subprocess.CompletedProcess(command, 7),
    )

    with pytest.raises(RuntimeError, match="status 7"):
        human_trials.capture_trials(trials=1, seed=110, output_dir=output_dir)

    manifest = json.loads((output_dir / "run-test" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["failed_trial"] == 1
    assert manifest["exit_code"] == 7


def test_capture_trials_rejects_nonpositive_trial_count(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        human_trials.capture_trials(trials=0, seed=110, output_dir=tmp_path / "human-driving")


def test_simulator_command_rejects_conflicting_forwarded_arguments() -> None:
    with pytest.raises(ValueError, match="--record-human"):
        human_trials.simulator_command(
            output=Path("human.jsonl"),
            seed=110,
            racing_args=("--record-human=other.jsonl",),
        )
