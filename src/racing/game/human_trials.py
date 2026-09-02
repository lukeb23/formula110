"""Capture multiple manual-driving sessions as separate training trajectories."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

DEFAULT_HUMAN_TRIAL_COUNT = 3
DEFAULT_HUMAN_TRIAL_SEED = 110
DEFAULT_HUMAN_TRIAL_OUTPUT_DIR = Path("artifacts/human-driving")
HUMAN_TRIAL_MANIFEST_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class CapturedTrial:
    """A summary of one independently stored simulator launch."""

    trial_number: int
    seed: int
    path: Path
    record_count: int
    session_ids: tuple[str, ...]


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for multi-trial human capture."""
    parser = argparse.ArgumentParser(
        description=(
            "Run multiple manual driving trials, storing each trajectory in its own JSONL file. "
            "Close the simulator window to begin the next trial."
        )
    )
    parser.add_argument("--trials", type=int, default=DEFAULT_HUMAN_TRIAL_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_HUMAN_TRIAL_SEED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_HUMAN_TRIAL_OUTPUT_DIR)
    parser.add_argument(
        "--vary-seed",
        action="store_true",
        help="increment the seed for each trial instead of replaying the same starting position",
    )
    parser.add_argument(
        "racing_args",
        nargs=argparse.REMAINDER,
        help="additional arguments for racing, placed after -- (for example: -- --muted)",
    )
    return parser


def simulator_command(*, output: Path, seed: int, racing_args: Sequence[str] = ()) -> list[str]:
    """Build the single-trial command executed in the current Python environment."""
    forwarded = list(racing_args)
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    _validate_forwarded_arguments(forwarded)
    return [
        sys.executable,
        "-m",
        "racing",
        "--seed",
        str(seed),
        "--record-human",
        str(output),
        *forwarded,
    ]


def capture_trials(
    *,
    trials: int,
    seed: int,
    output_dir: Path,
    vary_seed: bool = False,
    racing_args: Sequence[str] = (),
) -> list[CapturedTrial]:
    """Run interactive trials and store each one in a new run directory."""
    if trials < 1:
        raise ValueError("--trials must be at least 1")

    output_dir = output_dir.resolve()
    run_id = _new_run_id()
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {
        "schema_version": HUMAN_TRIAL_MANIFEST_SCHEMA_VERSION,
        "record_type": "human_driving_run",
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "recording",
        "requested_trials": trials,
        "base_seed": seed,
        "vary_seed": vary_seed,
        "trials": [],
    }
    _write_manifest(run_dir, manifest)

    summaries: list[CapturedTrial] = []
    for trial_number in range(1, trials + 1):
        trial_seed = seed + trial_number - 1 if vary_seed else seed
        trial_path = run_dir / f"trial-{trial_number:03d}-seed-{trial_seed}.jsonl"
        print(f"\nTrial {trial_number}/{trials} (seed {trial_seed})")
        print("Drive, then close the simulator window to continue.")
        try:
            result = subprocess.run(
                simulator_command(output=trial_path, seed=trial_seed, racing_args=racing_args),
                check=False,
            )
        except KeyboardInterrupt:
            manifest["status"] = "interrupted"
            _write_manifest(run_dir, manifest)
            raise
        if result.returncode != 0:
            manifest["status"] = "failed"
            manifest["failed_trial"] = trial_number
            manifest["exit_code"] = result.returncode
            _write_manifest(run_dir, manifest)
            raise RuntimeError(f"trial {trial_number} exited with status {result.returncode}")

        records = _read_records(trial_path)
        session_ids = tuple(
            dict.fromkeys(
                session_id
                for record in records
                if isinstance((session_id := record.get("session_id")), str)
            )
        )
        summary = CapturedTrial(
            trial_number=trial_number,
            seed=trial_seed,
            path=trial_path,
            record_count=len(records),
            session_ids=session_ids,
        )
        summaries.append(summary)
        manifest["trials"].append(_trial_manifest_record(summary))
        _write_manifest(run_dir, manifest)
        if records:
            session_label = ", ".join(session_ids) if session_ids else "missing session_id"
            print(f"Captured {len(records)} records ({session_label}) in {trial_path.name}.")
        else:
            print(f"Captured 0 records in {trial_path.name}; the window was closed before a control tick.")

    manifest["status"] = "complete"
    manifest["completed_at"] = datetime.now(UTC).isoformat()
    _write_manifest(run_dir, manifest)
    return summaries


def main(argv: Sequence[str] | None = None) -> int:
    """Run the multi-trial recorder command."""
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    try:
        summaries = capture_trials(
            trials=args.trials,
            seed=args.seed,
            output_dir=args.output_dir,
            vary_seed=args.vary_seed,
            racing_args=args.racing_args,
        )
    except KeyboardInterrupt:
        print("\nCapture interrupted; completed trial files and the manifest were preserved.")
        return 130
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))

    total_records = sum(summary.record_count for summary in summaries)
    run_dir = summaries[0].path.parent
    print(f"\nFinished: {total_records} records across {len(summaries)} trial files in {run_dir}")
    return 0


def _new_run_id() -> str:
    timestamp = datetime.now(UTC).strftime("run-%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid4().hex[:8]}"


def _trial_manifest_record(summary: CapturedTrial) -> dict[str, Any]:
    return {
        "trial_number": summary.trial_number,
        "seed": summary.seed,
        "path": summary.path.name,
        "record_count": summary.record_count,
        "session_ids": list(summary.session_ids),
        "status": "captured" if summary.record_count else "empty",
    }


def _write_manifest(run_dir: Path, manifest: dict[str, Any]) -> None:
    manifest_path = run_dir / "manifest.json"
    temporary_path = run_dir / ".manifest.json.tmp"
    temporary_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_path.replace(manifest_path)


def _validate_forwarded_arguments(arguments: Sequence[str]) -> None:
    forbidden = ("--record-human", "--seed", "--student-module")
    for argument in arguments:
        if argument == "h2h" or any(argument == option or argument.startswith(option + "=") for option in forbidden):
            raise ValueError(f"multi-trial capture does not allow forwarded argument {argument!r}")


def _read_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"recording line {line_number} is not valid JSON: {error.msg}") from error
        if not isinstance(record, dict):
            raise ValueError(f"recording line {line_number} is not a JSON object")
        records.append(record)
    return records


if __name__ == "__main__":
    raise SystemExit(main())
