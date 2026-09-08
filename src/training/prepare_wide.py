"""Prepare a wider 20-policy continuation without running evolution."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import cast

import torch

from controllers.observation import OBSERVATION_SIZE
from controllers.policy_network import (
    checkpoint_payload,
    load_policy_checkpoint,
    policy_metadata,
    widen_policy,
)
from training.evaluate import FitnessConfig
from training.evolution import CandidateEvaluation, EvolutionConfig, create_next_generation

# pyright: reportUnknownMemberType=false


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("artifacts/evolution/generation-124"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/evolution-wide128-20"))
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--training-seed", type=int, default=91508)
    args = parser.parse_args()
    source = cast(Path, args.source).resolve()
    root = cast(Path, args.output).resolve()
    ranking = json.loads((source / "results.json").read_text())["ranking"]
    # Match the unchanged phase-9 objective explicitly in the launch command.
    config = EvolutionConfig(
        generations=25,
        elite_count=2,
        training_seed=args.training_seed,
        mutation_std=0.0003,
        simulator_seeds=(110, 111, 112, 271, 997),
        round_seconds=30,
        worst_seed_weight=0.25,
        fitness=FitnessConfig(
            distance_weight=1.5,
            lap_completion_bonus=100,
            fast_lap_bonus=300,
            health_exponent=1,
            damage_penalty=0,
            wall_contact_penalty_per_s=20,
            low_progress_penalty_per_s=10,
        ),
    )
    if root.exists():
        raise FileExistsError(f"refusing to overwrite {root}")
    torch.manual_seed(args.training_seed)
    parents = root / "parents"
    parents.mkdir(parents=True)
    candidates: list[CandidateEvaluation] = []
    checks: list[dict[str, object]] = []
    for index, row in enumerate(ranking[:2]):
        checkpoint = source / row["checkpoint"]
        policy, _ = load_policy_checkpoint(checkpoint)
        wide = widen_policy(policy, args.hidden_size)
        observations = torch.rand((2048, OBSERVATION_SIZE), generator=torch.Generator().manual_seed(110)) * 2 - 1
        with torch.inference_mode():
            error = float((policy(observations) - wide(observations)).abs().max())
        if error > 1e-5:
            raise ValueError(f"widening changed outputs by {error}")
        name = f"parent-{index:03d}.pt"
        provenance = {
            "source_checkpoint": str(checkpoint),
            "source_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "source_hidden_size": policy.hidden_size,
            "operation": "neuron replication with divided outgoing weights",
            "max_absolute_output_error": error,
            "not_yet_simulator_validated": True,
        }
        torch.save(checkpoint_payload(wide, metadata=provenance), parents / name)
        # Source scores establish parent ordering only. New policies will be evaluated afresh.
        candidates.append(CandidateEvaluation(index, name, "widened_parent", None, float(row["fitness"]), ()))
        checks.append({**provenance, **policy_metadata(wide)})
    initial = create_next_generation(
        candidates,
        source_dir=parents,
        output_dir=root / "generation-000",
        generation_index=0,
        population_size=20,
        config=config,
    )
    (root / "preparation.json").write_text(
        json.dumps(
            {
                "source": str(source),
                "population_size": 20,
                "status": "prepared_not_started",
                "config": asdict(config),
                "checks": checks,
                "note": "Widening preserves outputs to float precision, not necessarily entire simulated trajectories. No improvement claimed.",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Prepared {initial}: 20 policies, hidden layers {args.hidden_size}, evolution not started.")


if __name__ == "__main__":
    main()
