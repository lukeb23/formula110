from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
import torch

from controllers.policy_network import (
    PolicyNetwork,
    checkpoint_payload,
    flatten_parameters,
    load_policy_checkpoint,
    parameter_mutation_mask,
)
from racing import HeadToHeadTeamRaceStats
from training.evaluate import FitnessConfig, FitnessResult, fitness_from_stats
from training.evolution import CandidateEvaluation, EvolutionConfig, create_next_generation, run_new_phase

# PyTorch and pytest expose a few intentionally dynamic helpers.
# pyright: reportUnknownMemberType=false


def test_fitness_rewards_distance_fast_laps_and_health() -> None:
    healthy_fast = fitness_from_stats(
        _stats(distance=120.0, laps=1, lap_time=12.0, damage=0.0, wall_contact=0.0),
        seed=110,
        round_seconds=20.0,
    )
    healthy_slow = fitness_from_stats(
        _stats(distance=120.0, laps=1, lap_time=18.0, damage=0.0, wall_contact=0.0),
        seed=110,
        round_seconds=20.0,
    )
    damaged_fast = fitness_from_stats(
        _stats(distance=120.0, laps=1, lap_time=12.0, damage=0.8, wall_contact=2.0),
        seed=110,
        round_seconds=20.0,
    )

    assert healthy_fast.fitness > healthy_slow.fitness
    assert healthy_fast.fitness > damaged_fast.fitness
    assert healthy_fast.distance_component == 120.0
    assert damaged_fast.health_factor == pytest.approx(0.04)
    assert damaged_fast.damage_component == 0.0
    assert damaged_fast.wall_contact_component == -60.0


def test_fitness_weights_are_configurable() -> None:
    result = fitness_from_stats(
        _stats(distance=10.0, laps=0, lap_time=None, damage=0.5, wall_contact=1.0),
        seed=1,
        round_seconds=10.0,
        config=FitnessConfig(
            distance_weight=2.0,
            health_exponent=0.0,
            damage_penalty=4.0,
            wall_contact_penalty_per_s=3.0,
        ),
    )

    assert result.fitness == pytest.approx(15.0)


def test_fitness_penalizes_remaining_at_low_progress() -> None:
    moving = fitness_from_stats(
        _stats(distance=100.0, laps=0, lap_time=None, damage=0.0, wall_contact=0.0),
        seed=1,
        round_seconds=30.0,
        config=FitnessConfig(low_progress_penalty_per_s=10.0),
    )
    stopped = fitness_from_stats(
        _stats(distance=100.0, laps=0, lap_time=None, damage=0.0, wall_contact=0.0, low_progress=3.0),
        seed=1,
        round_seconds=30.0,
        config=FitnessConfig(low_progress_penalty_per_s=10.0),
    )

    assert stopped.fitness == moving.fitness - 30.0


def test_next_generation_preserves_elites_and_mutates_children(tmp_path: Path) -> None:
    source_dir = tmp_path / "generation-000"
    source_dir.mkdir()
    policies: list[PolicyNetwork] = []
    evaluations: list[CandidateEvaluation] = []
    for index, fitness in enumerate((20.0, 10.0)):
        torch.manual_seed(index)
        policy = PolicyNetwork()
        policies.append(policy)
        checkpoint = f"policy-{index:03d}.pt"
        torch.save(checkpoint_payload(policy, metadata={"origin": "test"}), source_dir / checkpoint)
        evaluations.append(
            CandidateEvaluation(
                index=index,
                checkpoint=checkpoint,
                origin="test",
                parent_index=None,
                fitness=fitness,
                seed_results=(_fitness_result(fitness),),
            )
        )
    output_dir = tmp_path / "generation-001"

    create_next_generation(
        evaluations,
        source_dir=source_dir,
        output_dir=output_dir,
        generation_index=1,
        population_size=4,
        config=EvolutionConfig(generations=2, elite_count=2, mutation_std=0.05),
    )

    elite_zero = load_policy_checkpoint(output_dir / "policy-000-elite.pt")[0]
    elite_one = load_policy_checkpoint(output_dir / "policy-001-elite.pt")[0]
    child = load_policy_checkpoint(output_dir / "policy-002-mutated.pt")[0]
    assert torch.equal(flatten_parameters(elite_zero), flatten_parameters(policies[0]))
    assert torch.equal(flatten_parameters(elite_one), flatten_parameters(policies[1]))
    assert not torch.equal(flatten_parameters(child), flatten_parameters(policies[0]))


def test_next_generation_allocates_mixed_mutation_scales(tmp_path: Path) -> None:
    source_dir = tmp_path / "generation-000"
    source_dir.mkdir()
    evaluations: list[CandidateEvaluation] = []
    for index in range(3):
        checkpoint = f"policy-{index:03d}.pt"
        torch.save(checkpoint_payload(PolicyNetwork(), metadata={"origin": "test"}), source_dir / checkpoint)
        evaluations.append(
            CandidateEvaluation(
                index=index,
                checkpoint=checkpoint,
                origin="test",
                parent_index=None,
                fitness=float(3 - index),
                seed_results=(_fitness_result(float(3 - index)),),
            )
        )

    output_dir = tmp_path / "generation-001"
    create_next_generation(
        evaluations,
        source_dir=source_dir,
        output_dir=output_dir,
        generation_index=1,
        population_size=20,
        config=EvolutionConfig(
            generations=1,
            elite_count=3,
            mutation_stds=(0.004, 0.015, 0.04),
            mutation_weights=(0.7, 0.2, 0.1),
        ),
    )

    manifest = cast(dict[str, object], json.loads((output_dir / "manifest.json").read_text()))
    candidates = cast(list[dict[str, object]], manifest["candidates"])
    mutation_stds = [
        load_policy_checkpoint(output_dir / cast(str, candidate["path"]))[1]["mutation_std"]
        for candidate in candidates[3:]
    ]
    assert mutation_stds.count(0.004) == 12
    assert mutation_stds.count(0.015) == 3
    assert mutation_stds.count(0.04) == 2


@pytest.mark.parametrize(
    ("scope", "expected_count"),
    (("all", 1538), ("output", 66), ("steer_output", 33), ("throttle_output", 33)),
)
def test_parameter_mutation_mask_selects_expected_policy_parameters(scope: str, expected_count: int) -> None:
    mask = parameter_mutation_mask(PolicyNetwork(), scope)

    assert mask.dtype == torch.bool
    assert mask.shape == (1538,)
    assert int(mask.sum()) == expected_count


def test_next_generation_applies_scoped_mutations_only(tmp_path: Path) -> None:
    source_dir = tmp_path / "generation-000"
    source_dir.mkdir()
    policy = PolicyNetwork()
    checkpoint = "policy-000.pt"
    torch.save(checkpoint_payload(policy, metadata={"origin": "test"}), source_dir / checkpoint)
    evaluation = CandidateEvaluation(
        index=0,
        checkpoint=checkpoint,
        origin="test",
        parent_index=None,
        fitness=1.0,
        seed_results=(_fitness_result(1.0),),
    )

    output_dir = tmp_path / "generation-001"
    create_next_generation(
        [evaluation],
        source_dir=source_dir,
        output_dir=output_dir,
        generation_index=1,
        population_size=2,
        config=EvolutionConfig(
            generations=1,
            elite_count=1,
            mutation_stds=(0.01,),
            mutation_weights=(1.0,),
            mutation_scopes=("throttle_output",),
        ),
    )

    child, metadata = load_policy_checkpoint(output_dir / "policy-001-mutated.pt")
    delta = flatten_parameters(child) - flatten_parameters(policy)
    mask = parameter_mutation_mask(policy, "throttle_output")
    assert torch.count_nonzero(delta[mask]) > 0
    assert torch.count_nonzero(delta[~mask]) == 0
    assert metadata["mutation_scope"] == "throttle_output"


def test_new_phase_rescores_source_and_continues_numbering_with_larger_population(tmp_path: Path) -> None:
    source_dir = tmp_path / "generation-024"
    source_dir.mkdir()
    candidates: list[dict[str, object]] = []
    for index in range(3):
        checkpoint = f"policy-{index:03d}.pt"
        torch.save(checkpoint_payload(PolicyNetwork(), metadata={"origin": "source"}), source_dir / checkpoint)
        candidates.append({"index": index, "path": checkpoint, "origin": "source"})
    (source_dir / "manifest.json").write_text(
        json.dumps({"population_size": 3, "candidates": candidates}), encoding="utf-8"
    )

    def fake_evaluator(generation_dir: Path, _config: EvolutionConfig) -> list[CandidateEvaluation]:
        manifest = cast(dict[str, object], json.loads((generation_dir / "manifest.json").read_text()))
        entries = cast(list[dict[str, object]], manifest["candidates"])
        return [
            CandidateEvaluation(
                index=cast(int, entry["index"]),
                checkpoint=cast(str, entry["path"]),
                origin=cast(str, entry["origin"]),
                parent_index=None,
                fitness=float(len(entries) - position),
                seed_results=(_fitness_result(float(len(entries) - position)),),
            )
            for position, entry in enumerate(entries)
        ]

    phase = run_new_phase(
        source_dir,
        EvolutionConfig(generations=1, elite_count=2, mutation_std=0.01),
        population_size=6,
        generation_evaluator=fake_evaluator,
    )

    next_manifest = cast(dict[str, object], json.loads((tmp_path / "generation-025/manifest.json").read_text()))
    assert next_manifest["population_size"] == 6
    assert phase["source_generation"] == 24
    assert phase["final_generation"] == 25
    assert (tmp_path / "phase-001-source-rescore.json").exists()
    assert (tmp_path / "generation-025/results.json").exists()


def _stats(
    *,
    distance: float,
    laps: int,
    lap_time: float | None,
    damage: float,
    wall_contact: float,
    low_progress: float = 0.0,
) -> HeadToHeadTeamRaceStats:
    return HeadToHeadTeamRaceStats(
        distances_m=(distance,),
        raw_distances_m=(distance,),
        lap_counts=(laps,),
        best_lap_times_seconds=(lap_time,),
        damages=(damage,),
        wall_contact_seconds=(wall_contact,),
        car_contact_seconds=(0.0,),
        max_speeds_mps=(20.0,),
        low_progress_seconds=(low_progress,),
    )


def _fitness_result(fitness: float) -> FitnessResult:
    return FitnessResult(
        seed=110,
        fitness=fitness,
        scored_distance_m=fitness,
        raw_distance_m=fitness,
        lap_count=0,
        best_lap_time_s=None,
        max_speed_mps=0.0,
        damage=0.0,
        wall_contact_s=0.0,
        car_contact_s=0.0,
        low_progress_s=0.0,
        marshal_count=0,
        progress_before_health=fitness,
        health_factor=1.0,
        distance_component=fitness,
        completion_component=0.0,
        time_component=0.0,
        damage_component=0.0,
        wall_contact_component=0.0,
        low_progress_component=0.0,
    )
