"""Capture repeated student-controller runs as separate training trajectories."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from racing.game.human_trials import CapturedTrial, _new_run_id, _read_records, _trial_manifest_record, _write_manifest

DEFAULT_CONTROLLER_TRIAL_COUNT = 3
DEFAULT_CONTROLLER_TRIAL_SEED = 110
DEFAULT_CONTROLLER_TRIAL_OUTPUT_DIR = Path("artifacts/controller-driving")
CONTROLLER_TRIAL_MANIFEST_SCHEMA_VERSION = 1


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the controller-trial command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Run a student controller repeatedly, storing each trajectory in its own JSONL file. "
            "Close the simulator window to begin the next trial."
        )
    )
    parser.add_argument("--student-module", required=True)
    parser.add_argument("--control-function", default="control")
    parser.add_argument("--trials", type=int, default=DEFAULT_CONTROLLER_TRIAL_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_CONTROLLER_TRIAL_SEED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_CONTROLLER_TRIAL_OUTPUT_DIR)
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


def simulator_command(
    *,
    output: Path,
    seed: int,
    student_module: str,
    control_function: str = "control",
    racing_args: Sequence[str] = (),
) -> list[str]:
    """Build one recorded student-controller simulator command."""
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
        "--student-module",
        student_module,
        "--control-function",
        control_function,
        "--record-controller",
        str(output),
        *forwarded,
    ]


def capture_trials(
    *,
    trials: int,
    seed: int,
    output_dir: Path,
    student_module: str,
    control_function: str = "control",
    vary_seed: bool = False,
    racing_args: Sequence[str] = (),
) -> list[CapturedTrial]:
    """Run controller trials and create an incrementally updated manifest."""
    if trials < 1:
        raise ValueError("--trials must be at least 1")

    output_dir = output_dir.resolve()
    run_id = _new_run_id()
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    control_source: dict[str, Any] = {
        "type": "student_controller",
        "module": student_module,
        "function": control_function,
    }
    source_sha256 = _controller_source_sha256(student_module)
    if source_sha256 is not None:
        control_source["sha256"] = source_sha256
    manifest: dict[str, Any] = {
        "schema_version": CONTROLLER_TRIAL_MANIFEST_SCHEMA_VERSION,
        "record_type": "controller_driving_run",
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "recording",
        "requested_trials": trials,
        "base_seed": seed,
        "vary_seed": vary_seed,
        "control_source": control_source,
        "trials": [],
    }
    _write_manifest(run_dir, manifest)

    summaries: list[CapturedTrial] = []
    for trial_number in range(1, trials + 1):
        trial_seed = seed + trial_number - 1 if vary_seed else seed
        trial_path = run_dir / f"trial-{trial_number:03d}-seed-{trial_seed}.jsonl"
        print(f"\nController trial {trial_number}/{trials} (seed {trial_seed})")
        print("Close the simulator window to continue to the next trial.")
        try:
            result = subprocess.run(
                simulator_command(
                    output=trial_path,
                    seed=trial_seed,
                    student_module=student_module,
                    control_function=control_function,
                    racing_args=racing_args,
                ),
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
        summary = CapturedTrial(trial_number, trial_seed, trial_path, len(records), session_ids)
        summaries.append(summary)
        manifest["trials"].append(_trial_manifest_record(summary))
        _write_manifest(run_dir, manifest)
        print(f"Captured {len(records)} records in {trial_path.name}.")

    manifest["status"] = "complete"
    manifest["completed_at"] = datetime.now(UTC).isoformat()
    _write_manifest(run_dir, manifest)
    return summaries


def main(argv: Sequence[str] | None = None) -> int:
    """Run the controller-trial recorder command."""
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    try:
        summaries = capture_trials(
            trials=args.trials,
            seed=args.seed,
            output_dir=args.output_dir,
            student_module=args.student_module,
            control_function=args.control_function,
            vary_seed=args.vary_seed,
            racing_args=args.racing_args,
        )
    except KeyboardInterrupt:
        print("\nCapture interrupted; completed trial files and the manifest were preserved.")
        return 130
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))

    total_records = sum(summary.record_count for summary in summaries)
    print(f"\nFinished: {total_records} records across {len(summaries)} trial files in {summaries[0].path.parent}")
    return 0


def _validate_forwarded_arguments(arguments: Sequence[str]) -> None:
    forbidden = (
        "--record-human",
        "--record-controller",
        "--seed",
        "--student-module",
        "--control-function",
    )
    for argument in arguments:
        if argument == "h2h" or any(argument == option or argument.startswith(option + "=") for option in forbidden):
            raise ValueError(f"controller-trial capture does not allow forwarded argument {argument!r}")


def _controller_source_sha256(module_reference: str) -> str | None:
    source_path = Path(module_reference)
    if not source_path.is_file():
        return None
    return hashlib.sha256(source_path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
