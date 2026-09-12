"""Re-evaluate saved evolution champions under one consistent protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from controllers.policy_network import flatten_parameters, load_policy_checkpoint
from training.evaluate import FitnessConfig, evaluate_policy


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as file:
        return json.load(file)  # type: ignore[no-any-return]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evolution-dir", type=Path, default=Path("artifacts/evolution"))
    parser.add_argument("--generations", type=int, nargs="+", default=(0, 62, 124))
    parser.add_argument("--simulator-seeds", type=int, nargs="+", default=(110, 111, 112, 271, 997))
    parser.add_argument("--round-seconds", type=float, default=30.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/evolution/consistent-rescore/results.json"),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="preserve existing results and evaluate only requested generations that are missing",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    evolution_dir = args.evolution_dir.resolve()
    output_path = args.output.resolve()
    if output_path.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite existing rescore: {output_path}")
    if args.round_seconds <= 0.0:
        raise ValueError("round seconds must be positive")

    phase_9 = _load_json(evolution_dir / "phase-009.json")
    fitness_config_data = phase_9["config"]["fitness"]
    fitness_config = FitnessConfig(**fitness_config_data)
    if output_path.exists():
        document = _load_json(output_path)
        if float(document["round_seconds"]) != args.round_seconds:
            raise ValueError("existing rescore uses a different round duration")
        if list(document["simulator_seeds"]) != list(args.simulator_seeds):
            raise ValueError("existing rescore uses different simulator seeds")
        if document["fitness_config"] != fitness_config_data:
            raise ValueError("existing rescore uses a different fitness configuration")
        rescored_generations = list(document["generations"])
    else:
        document = {
            "schema_version": 1,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "purpose": "consistent physical-metric comparison of representative generation champions",
            "round_seconds": args.round_seconds,
            "simulator_seeds": list(args.simulator_seeds),
            "fitness_config": fitness_config_data,
            "generations": [],
        }
        rescored_generations = []

    completed_generations = {int(item["generation"]) for item in rescored_generations}
    missing_generations = [generation for generation in args.generations if generation not in completed_generations]

    for generation in missing_generations:
        generation_dir = evolution_dir / f"generation-{generation:03d}"
        historical_result = _load_json(generation_dir / "results.json")
        checkpoint_name = str(historical_result["best_checkpoint"])
        checkpoint_path = generation_dir / checkpoint_name
        policy, _ = load_policy_checkpoint(checkpoint_path)
        weights = flatten_parameters(policy)
        seed_results = [
            evaluate_policy(
                weights,
                seed,
                round_seconds=args.round_seconds,
                fitness_config=fitness_config,
            ).to_dict()
            for seed in args.simulator_seeds
        ]
        rescored_generations.append(
            {
                "generation": generation,
                "checkpoint": checkpoint_name,
                "checkpoint_sha256": _sha256(checkpoint_path),
                "seed_results": seed_results,
            }
        )

    if not missing_generations:
        print(f"All requested generations are already present in {output_path}")
        return

    document["updated_at_utc"] = datetime.now(UTC).isoformat()
    document["generations"] = sorted(rescored_generations, key=lambda item: int(item["generation"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".json.tmp")
    with temporary_path.open("x") as file:
        json.dump(document, file, indent=2)
        file.write("\n")
    temporary_path.replace(output_path)
    print(f"Wrote consistent rescore to {output_path}")


if __name__ == "__main__":
    main()
