"""Simple elitist Gaussian neuroevolution for shared Formula 110 policies."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import fmean
from typing import cast

import numpy as np
import torch
from torch import Tensor

from controllers.observation import observation_metadata
from controllers.policy_network import (
    PARAMETER_MUTATION_SCOPES,
    PolicyNetwork,
    checkpoint_payload,
    flatten_parameters,
    load_parameter_vector,
    load_policy_checkpoint,
    parameter_mutation_mask,
    policy_metadata,
)
from training.evaluate import FitnessConfig, FitnessResult, evaluate_policy

# PyTorch random helpers contain intentionally dynamic annotations.
# pyright: reportUnknownMemberType=false

EVOLUTION_SCHEMA_VERSION = 1
Evaluator = Callable[[Tensor, int], FitnessResult]
GenerationEvaluator = Callable[[Path, "EvolutionConfig"], list["CandidateEvaluation"]]
ProgressReporter = Callable[[dict[str, object]], None]


@dataclass(frozen=True)
class EvolutionConfig:
    generations: int = 25
    elite_count: int = 2
    mutation_std: float = 0.05
    mutation_stds: tuple[float, ...] = ()
    mutation_weights: tuple[float, ...] = ()
    mutation_scopes: tuple[str, ...] = ()
    training_seed: int = 110
    simulator_seeds: tuple[int, ...] = (110, 111, 112)
    round_seconds: float = 20.0
    worst_seed_weight: float = 0.3
    fitness: FitnessConfig = field(default_factory=FitnessConfig)


@dataclass(frozen=True)
class CandidateEvaluation:
    index: int
    checkpoint: str
    origin: str
    parent_index: int | None
    fitness: float
    seed_results: tuple[FitnessResult, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "checkpoint": self.checkpoint,
            "origin": self.origin,
            "parent_index": self.parent_index,
            "fitness": self.fitness,
            "seed_results": [result.to_dict() for result in self.seed_results],
        }


@dataclass(frozen=True)
class MutationProfile:
    std: float
    scope: str


def run_evolution(
    initial_population_dir: Path,
    config: EvolutionConfig,
    *,
    resume: bool = False,
) -> dict[str, object]:
    """Evaluate and evolve a population, preserving elites exactly."""
    _validate_config(config)
    _seed_everything(config.training_seed)
    root = initial_population_dir.parent
    experiment_path = root / "experiment.json"
    if experiment_path.exists() and not resume:
        raise FileExistsError(f"refusing to overwrite experiment: {experiment_path}")
    initial_manifest = _load_json(initial_population_dir / "manifest.json")
    population_size = _required_int(initial_manifest["population_size"], "population_size")
    if config.elite_count >= population_size:
        raise ValueError("elite_count must be smaller than population size")

    experiment: dict[str, object]
    if resume:
        if not experiment_path.exists():
            raise FileNotFoundError(f"no experiment to resume: {experiment_path}")
        experiment = _load_json(experiment_path)
        generation_summaries = cast(list[dict[str, object]], experiment.get("generations", []))
        completed_count = len(generation_summaries)
        if config.generations <= completed_count:
            raise ValueError(f"resume target must exceed {completed_count} completed generations")
        _validate_resume_config(experiment, config, initial_population_dir, population_size)
        previous_dir = root / f"generation-{completed_count - 1:03d}"
        previous_ranked = _load_ranking(previous_dir / "results.json")
        generation_dir = create_next_generation(
            previous_ranked,
            source_dir=previous_dir,
            output_dir=root / f"generation-{completed_count:03d}",
            generation_index=completed_count,
            population_size=population_size,
            config=config,
        )
        experiment_config = cast(dict[str, object], experiment["config"])
        experiment_config["generations"] = config.generations
        experiment["status"] = "running"
        start_generation = completed_count
    else:
        generation_summaries = []
        generation_dir = initial_population_dir
        start_generation = 0
        experiment = {
            "schema_version": EVOLUTION_SCHEMA_VERSION,
            "status": "running",
            "initial_population": str(initial_population_dir.resolve()),
            "population_size": population_size,
            "config": {
                **asdict(config),
                "simulator_seeds": list(config.simulator_seeds),
                "fitness": asdict(config.fitness),
            },
            "policy": {**policy_metadata(), "observation": observation_metadata()},
            "fitness_definition": (
                "progress=(distance_weight*scored_distance + lap_completion_bonus*laps + "
                "fast_lap_bonus*(round_seconds-best_lap)/round_seconds); "
                "seed_fitness=progress*(1-damage)^health_exponent - damage_penalty*damage - "
                "wall_contact_penalty_per_s*wall_contact_seconds - "
                "low_progress_penalty_per_s*low_progress_seconds; overall=(1-worst_seed_weight)*mean + "
                "worst_seed_weight*minimum"
            ),
            "generations": generation_summaries,
        }
    _write_json(experiment_path, experiment)

    elite_cache: list[CandidateEvaluation] | None = None
    for generation_index in range(start_generation, config.generations):
        evaluations = evaluate_generation(generation_dir, config, elite_cache=elite_cache)
        ranked = sorted(evaluations, key=lambda item: (-item.fitness, item.index))
        summary = _generation_summary(generation_index, ranked)
        results_path = generation_dir / "results.json"
        if results_path.exists():
            raise FileExistsError(f"refusing to overwrite generation results: {results_path}")
        _write_json(
            results_path,
            {
                "schema_version": EVOLUTION_SCHEMA_VERSION,
                **summary,
                "ranking": [evaluation.to_dict() for evaluation in ranked],
            },
        )
        generation_summaries.append(summary)
        experiment["generations"] = generation_summaries
        _write_json(experiment_path, experiment)
        if generation_index + 1 < config.generations:
            generation_dir = create_next_generation(
                ranked,
                source_dir=generation_dir,
                output_dir=root / f"generation-{generation_index + 1:03d}",
                generation_index=generation_index + 1,
                population_size=population_size,
                config=config,
            )
            elite_cache = ranked

    experiment["status"] = "complete"
    _write_json(experiment_path, experiment)
    return experiment


def run_new_phase(
    source_generation_dir: Path,
    config: EvolutionConfig,
    *,
    population_size: int,
    progress_reporter: ProgressReporter | None = None,
    generation_evaluator: GenerationEvaluator | None = None,
) -> dict[str, object]:
    """Rescore an existing population and continue its lineage under a new objective."""
    _validate_config(config)
    if population_size <= config.elite_count:
        raise ValueError("population_size must be greater than elite_count")
    source_manifest = _load_json(source_generation_dir / "manifest.json")
    source_population_size = _required_int(source_manifest["population_size"], "source population_size")
    if config.elite_count > source_population_size:
        raise ValueError("elite_count cannot exceed the source population size")
    source_generation = _generation_index(source_generation_dir)
    root = source_generation_dir.parent
    phase_index = _next_phase_index(root)
    phase_path = root / f"phase-{phase_index:03d}.json"
    source_rescore_path = root / f"phase-{phase_index:03d}-source-rescore.json"
    if phase_path.exists() or source_rescore_path.exists():
        raise FileExistsError(f"refusing to overwrite phase {phase_index}")
    evaluator = evaluate_generation if generation_evaluator is None else generation_evaluator
    _seed_everything(config.training_seed)
    phase: dict[str, object] = {
        "schema_version": EVOLUTION_SCHEMA_VERSION,
        "phase": phase_index,
        "status": "running",
        "source_generation": source_generation,
        "source_population": str(source_generation_dir.resolve()),
        "source_population_size": source_population_size,
        "population_size": population_size,
        "config": {
            **asdict(config),
            "simulator_seeds": list(config.simulator_seeds),
            "fitness": asdict(config.fitness),
        },
        "policy": {**policy_metadata(), "observation": observation_metadata()},
        "generations": [],
    }
    _write_json(phase_path, phase)

    rescored = sorted(evaluator(source_generation_dir, config), key=lambda item: (-item.fitness, item.index))
    source_summary = _generation_summary(source_generation, rescored)
    _write_json(
        source_rescore_path,
        {
            "schema_version": EVOLUTION_SCHEMA_VERSION,
            "phase": phase_index,
            "purpose": "source_population_rescore",
            **source_summary,
            "ranking": [evaluation.to_dict() for evaluation in rescored],
        },
    )
    phase["source_rescore"] = source_summary
    if progress_reporter is not None:
        progress_reporter({"stage": "source_rescore", **source_summary})

    next_generation = source_generation + 1
    generation_dir = create_next_generation(
        rescored,
        source_dir=source_generation_dir,
        output_dir=root / f"generation-{next_generation:03d}",
        generation_index=next_generation,
        population_size=population_size,
        config=config,
    )
    parent_ranked = rescored
    summaries: list[dict[str, object]] = []
    final_generation = next_generation + config.generations - 1
    for generation_index in range(next_generation, final_generation + 1):
        evaluations = (
            evaluate_generation(generation_dir, config, elite_cache=parent_ranked)
            if generation_evaluator is None
            else evaluator(generation_dir, config)
        )
        ranked = sorted(evaluations, key=lambda item: (-item.fitness, item.index))
        summary = _generation_summary(generation_index, ranked)
        results_path = generation_dir / "results.json"
        if results_path.exists():
            raise FileExistsError(f"refusing to overwrite generation results: {results_path}")
        _write_json(
            results_path,
            {
                "schema_version": EVOLUTION_SCHEMA_VERSION,
                "phase": phase_index,
                **summary,
                "ranking": [evaluation.to_dict() for evaluation in ranked],
            },
        )
        summaries.append(summary)
        phase["generations"] = summaries
        _write_json(phase_path, phase)
        if progress_reporter is not None:
            progress_reporter({"stage": "generation", **summary})
        if generation_index < final_generation:
            generation_dir = create_next_generation(
                ranked,
                source_dir=generation_dir,
                output_dir=root / f"generation-{generation_index + 1:03d}",
                generation_index=generation_index + 1,
                population_size=population_size,
                config=config,
            )
            parent_ranked = ranked

    phase["status"] = "complete"
    phase["final_generation"] = final_generation
    _write_json(phase_path, phase)
    return phase


def evaluate_generation(
    generation_dir: Path,
    config: EvolutionConfig,
    *,
    elite_cache: list[CandidateEvaluation] | None = None,
) -> list[CandidateEvaluation]:
    manifest = _load_json(generation_dir / "manifest.json")
    candidates = cast(list[dict[str, object]], manifest["candidates"])
    cached_by_index = {} if elite_cache is None else {evaluation.index: evaluation for evaluation in elite_cache}
    evaluations: list[CandidateEvaluation] = []
    for entry in candidates:
        checkpoint_name = cast(str, entry["path"])
        policy, metadata = load_policy_checkpoint(generation_dir / checkpoint_name)
        parent_index = _optional_int(metadata.get("parent_index"))
        cached_parent = None if parent_index is None else cached_by_index.get(parent_index)
        if metadata.get("origin") == "elite" and cached_parent is not None and cached_parent.seed_results:
            evaluations.append(
                CandidateEvaluation(
                    index=_required_int(entry["index"], "candidate index"),
                    checkpoint=checkpoint_name,
                    origin="elite",
                    parent_index=parent_index,
                    fitness=cached_parent.fitness,
                    seed_results=cached_parent.seed_results,
                )
            )
            continue
        vector = flatten_parameters(policy)
        seed_results = tuple(
            evaluate_policy(
                vector,
                simulator_seed,
                round_seconds=config.round_seconds,
                fitness_config=config.fitness,
            )
            for simulator_seed in config.simulator_seeds
        )
        evaluations.append(
            CandidateEvaluation(
                index=_required_int(entry["index"], "candidate index"),
                checkpoint=checkpoint_name,
                origin=str(metadata.get("origin", entry.get("origin", "unknown"))),
                parent_index=parent_index,
                fitness=_aggregate_seed_fitness(seed_results, config.worst_seed_weight),
                seed_results=seed_results,
            )
        )
    return evaluations


def create_next_generation(
    ranked: list[CandidateEvaluation],
    *,
    source_dir: Path,
    output_dir: Path,
    generation_index: int,
    population_size: int,
    config: EvolutionConfig,
) -> Path:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite generation: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    elites = ranked[: config.elite_count]
    elite_vectors = [flatten_parameters(load_policy_checkpoint(source_dir / elite.checkpoint)[0]) for elite in elites]
    mutation_schedule = _mutation_schedule(population_size - config.elite_count, config)
    scope_masks = {
        profile.scope: parameter_mutation_mask(PolicyNetwork(), profile.scope) for profile in set(mutation_schedule)
    }
    entries: list[dict[str, object]] = []
    for index in range(population_size):
        parent_slot = index if index < config.elite_count else (index - config.elite_count) % config.elite_count
        parent = elites[parent_slot]
        vector = elite_vectors[parent_slot].clone()
        origin = "elite" if index < config.elite_count else "mutated"
        mutation_seed: int | None = None
        applied_mutation_std = 0.0
        applied_mutation_scope = "none"
        if origin == "mutated":
            profile = mutation_schedule[index - config.elite_count]
            applied_mutation_std = profile.std
            applied_mutation_scope = profile.scope
            mutation_seed = config.training_seed + generation_index * 100_000 + index
            generator = torch.Generator(device="cpu").manual_seed(mutation_seed)
            noise = torch.randn(vector.shape, generator=generator) * applied_mutation_std
            vector += noise * scope_masks[applied_mutation_scope]
        policy = PolicyNetwork()
        load_parameter_vector(policy, vector)
        filename = f"policy-{index:03d}-{origin}.pt"
        metadata: dict[str, object] = {
            **policy_metadata(),
            "observation": observation_metadata(),
            "population_index": index,
            "generation": generation_index,
            "origin": origin,
            "parent_index": parent.index,
            "parent_checkpoint": str((source_dir / parent.checkpoint).resolve()),
            "mutation_seed": mutation_seed,
            "mutation_std": applied_mutation_std,
            "mutation_scope": applied_mutation_scope,
        }
        checkpoint_path = output_dir / filename
        torch.save(checkpoint_payload(policy, metadata=metadata), checkpoint_path)
        entries.append(
            {
                "index": index,
                "origin": origin,
                "parent_index": parent.index,
                "mutation_std": applied_mutation_std,
                "mutation_scope": applied_mutation_scope,
                "path": filename,
                "sha256": _sha256(checkpoint_path),
            }
        )
    _write_json(
        output_dir / "manifest.json",
        {
            "schema_version": EVOLUTION_SCHEMA_VERSION,
            "generation": generation_index,
            "population_size": population_size,
            "elite_count": config.elite_count,
            "mutation_std": config.mutation_std,
            "mutation_stds": list(config.mutation_stds),
            "mutation_weights": list(config.mutation_weights),
            "mutation_scopes": list(config.mutation_scopes),
            "candidates": entries,
        },
    )
    return output_dir


def _generation_summary(generation: int, ranked: list[CandidateEvaluation]) -> dict[str, object]:
    scores = [evaluation.fitness for evaluation in ranked]
    return {
        "generation": generation,
        "best_fitness": max(scores),
        "mean_fitness": fmean(scores),
        "worst_fitness": min(scores),
        "best_candidate_index": ranked[0].index,
        "best_checkpoint": ranked[0].checkpoint,
    }


def _validate_config(config: EvolutionConfig) -> None:
    if config.generations < 1 or config.elite_count < 1:
        raise ValueError("generations and elite_count must be positive")
    if config.mutation_std <= 0.0 or config.round_seconds <= 0.0:
        raise ValueError("mutation_std and round_seconds must be positive")
    if bool(config.mutation_stds) != bool(config.mutation_weights):
        raise ValueError("mutation_stds and mutation_weights must be provided together")
    if config.mutation_stds:
        if len(config.mutation_stds) != len(config.mutation_weights):
            raise ValueError("mutation_stds and mutation_weights must have equal lengths")
        if any(value <= 0.0 for value in config.mutation_stds):
            raise ValueError("all mutation_stds must be positive")
        if any(value < 0.0 for value in config.mutation_weights) or sum(config.mutation_weights) <= 0.0:
            raise ValueError("mutation_weights must be non-negative with a positive sum")
        if config.mutation_scopes and len(config.mutation_scopes) != len(config.mutation_stds):
            raise ValueError("mutation_scopes and mutation_stds must have equal lengths")
        if any(scope not in PARAMETER_MUTATION_SCOPES for scope in config.mutation_scopes):
            valid = ", ".join(PARAMETER_MUTATION_SCOPES)
            raise ValueError(f"mutation_scopes must be selected from: {valid}")
    elif config.mutation_scopes:
        raise ValueError("mutation_scopes requires mutation_stds and mutation_weights")
    if not config.simulator_seeds:
        raise ValueError("at least one simulator seed is required")
    if not 0.0 <= config.worst_seed_weight <= 1.0:
        raise ValueError("worst_seed_weight must be in [0, 1]")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _mutation_schedule(child_count: int, config: EvolutionConfig) -> list[MutationProfile]:
    """Allocate mutation scales deterministically using largest remainders."""
    if not config.mutation_stds:
        return [MutationProfile(config.mutation_std, "all")] * child_count
    scopes = config.mutation_scopes or ("all",) * len(config.mutation_stds)
    total_weight = sum(config.mutation_weights)
    quotas = [child_count * weight / total_weight for weight in config.mutation_weights]
    counts = [int(quota) for quota in quotas]
    remaining = child_count - sum(counts)
    remainder_order = sorted(range(len(quotas)), key=lambda index: (-(quotas[index] - counts[index]), index))
    for index in remainder_order[:remaining]:
        counts[index] += 1
    return [
        MutationProfile(std, scope)
        for std, scope, count in zip(config.mutation_stds, scopes, counts, strict=True)
        for _ in range(count)
    ]


def _load_json(path: Path) -> dict[str, object]:
    loaded: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"expected JSON object: {path}")
    return cast(dict[str, object], loaded)


def _load_ranking(path: Path) -> list[CandidateEvaluation]:
    results = _load_json(path)
    ranking = cast(list[dict[str, object]], results["ranking"])
    return [
        CandidateEvaluation(
            index=_required_int(entry["index"], "candidate index"),
            checkpoint=cast(str, entry["checkpoint"]),
            origin=cast(str, entry["origin"]),
            parent_index=_optional_int(entry.get("parent_index")),
            fitness=_required_float(entry["fitness"], "fitness"),
            seed_results=(),
        )
        for entry in ranking
    ]


def _validate_resume_config(
    experiment: dict[str, object],
    config: EvolutionConfig,
    initial_population_dir: Path,
    population_size: int,
) -> None:
    if experiment.get("initial_population") != str(initial_population_dir.resolve()):
        raise ValueError("resume initial population does not match")
    if experiment.get("population_size") != population_size:
        raise ValueError("resume population size does not match")
    saved = cast(dict[str, object], experiment["config"])
    expected: dict[str, object] = {
        "elite_count": config.elite_count,
        "mutation_std": config.mutation_std,
        "mutation_stds": list(config.mutation_stds),
        "mutation_weights": list(config.mutation_weights),
        "mutation_scopes": list(config.mutation_scopes),
        "training_seed": config.training_seed,
        "simulator_seeds": list(config.simulator_seeds),
        "round_seconds": config.round_seconds,
        "worst_seed_weight": config.worst_seed_weight,
        "fitness": asdict(config.fitness),
    }
    for name, value in expected.items():
        saved_value = saved.get(name, [] if name in {"mutation_stds", "mutation_weights", "mutation_scopes"} else None)
        if saved_value != value:
            raise ValueError(f"resume configuration differs for {name}")


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _optional_int(value: object) -> int | None:
    return int(value) if isinstance(value, int) else None


def _required_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"expected integer {field_name}")
    return int(value)


def _required_float(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"expected numeric {field_name}")
    return float(value)


def _generation_index(generation_dir: Path) -> int:
    prefix = "generation-"
    if not generation_dir.name.startswith(prefix):
        raise ValueError(f"expected generation directory name: {generation_dir}")
    return _required_int(generation_dir.name.removeprefix(prefix), "generation index")


def _next_phase_index(root: Path) -> int:
    indices: list[int] = []
    for path in root.glob("phase-[0-9][0-9][0-9].json"):
        try:
            indices.append(int(path.stem.removeprefix("phase-")))
        except ValueError:
            continue
    return max(indices, default=0) + 1


def _aggregate_seed_fitness(results: tuple[FitnessResult, ...], worst_seed_weight: float) -> float:
    scores = [result.fitness for result in results]
    return (1.0 - worst_seed_weight) * fmean(scores) + worst_seed_weight * min(scores)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-population", type=Path, default=Path("artifacts/evolution/generation-000"))
    parser.add_argument("--new-phase-from", type=Path)
    parser.add_argument("--population-size", type=int)
    parser.add_argument("--generations", type=int, default=25)
    parser.add_argument("--elites", type=int, default=2)
    parser.add_argument("--mutation-std", type=float, default=0.05)
    parser.add_argument(
        "--mutation-stds",
        type=float,
        nargs="+",
        help="optional mixed mutation scales; use with --mutation-weights",
    )
    parser.add_argument(
        "--mutation-weights",
        type=float,
        nargs="+",
        help="relative offspring allocation for --mutation-stds",
    )
    parser.add_argument(
        "--mutation-scopes",
        choices=PARAMETER_MUTATION_SCOPES,
        nargs="+",
        help="parameter scope for each mixed mutation scale (default: all)",
    )
    parser.add_argument("--training-seed", type=int, default=110)
    parser.add_argument("--simulator-seeds", type=int, nargs="+", default=(110, 111, 112))
    parser.add_argument("--round-seconds", type=float, default=20.0)
    parser.add_argument("--worst-seed-weight", type=float, default=0.3)
    parser.add_argument("--distance-weight", type=float, default=1.0)
    parser.add_argument("--lap-completion-bonus", type=float, default=100.0)
    parser.add_argument("--fast-lap-bonus", type=float, default=100.0)
    parser.add_argument("--health-exponent", type=float, default=2.0)
    parser.add_argument("--damage-penalty", type=float, default=0.0)
    parser.add_argument("--wall-contact-penalty-per-s", type=float, default=30.0)
    parser.add_argument("--low-progress-penalty-per-s", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = EvolutionConfig(
        generations=args.generations,
        elite_count=args.elites,
        mutation_std=args.mutation_std,
        mutation_stds=tuple(args.mutation_stds or ()),
        mutation_weights=tuple(args.mutation_weights or ()),
        mutation_scopes=tuple(args.mutation_scopes or ()),
        training_seed=args.training_seed,
        simulator_seeds=tuple(args.simulator_seeds),
        round_seconds=args.round_seconds,
        worst_seed_weight=args.worst_seed_weight,
        fitness=FitnessConfig(
            distance_weight=args.distance_weight,
            lap_completion_bonus=args.lap_completion_bonus,
            fast_lap_bonus=args.fast_lap_bonus,
            health_exponent=args.health_exponent,
            damage_penalty=args.damage_penalty,
            wall_contact_penalty_per_s=args.wall_contact_penalty_per_s,
            low_progress_penalty_per_s=args.low_progress_penalty_per_s,
        ),
    )
    if args.new_phase_from is not None:
        if args.resume:
            parser.error("--new-phase-from cannot be combined with --resume")
        if args.population_size is None:
            parser.error("--new-phase-from requires --population-size")
        phase = run_new_phase(
            args.new_phase_from,
            config,
            population_size=args.population_size,
            progress_reporter=_print_progress,
        )
        print(json.dumps({"status": phase["status"], "final_generation": phase["final_generation"]}, indent=2))
    else:
        if args.population_size is not None:
            parser.error("--population-size is only used with --new-phase-from")
        experiment = run_evolution(args.initial_population, config, resume=args.resume)
        print(json.dumps(experiment["generations"], indent=2))


def _print_progress(summary: dict[str, object]) -> None:
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
