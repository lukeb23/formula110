"""Create equal BC-derived and random policy sets for generation zero."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from controllers.observation import observation_metadata
from controllers.policy_network import (
    PolicyNetwork,
    checkpoint_payload,
    flatten_parameters,
    load_parameter_vector,
    load_policy_checkpoint,
    policy_metadata,
)

# PyTorch random-state context managers contain intentionally dynamic annotations.
# pyright: reportUnknownMemberType=false

POPULATION_MANIFEST_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PopulationConfig:
    policies_per_origin: int = 5
    mutation_std: float = 0.05
    seed: int = 110


def generate_initial_population(
    bc_checkpoint: Path,
    output_dir: Path,
    config: PopulationConfig | None = None,
) -> dict[str, object]:
    """Write independently mutated BC policies and independently initialized random policies."""
    config = PopulationConfig() if config is None else config
    if config.policies_per_origin < 1:
        raise ValueError("policies_per_origin must be positive")
    if config.mutation_std <= 0.0:
        raise ValueError("mutation_std must be positive")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty population directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    _seed_everything(config.seed)

    bc_policy, bc_metadata = load_policy_checkpoint(bc_checkpoint)
    base_vector = flatten_parameters(bc_policy)
    entries: list[dict[str, object]] = []
    used_vectors: list[torch.Tensor] = []

    for index in range(config.policies_per_origin):
        candidate_seed = config.seed + index
        generator = torch.Generator(device="cpu").manual_seed(candidate_seed)
        vector = base_vector + torch.randn(base_vector.shape, generator=generator) * config.mutation_std
        policy = PolicyNetwork(bc_policy.hidden_size)
        load_parameter_vector(policy, vector)
        entries.append(
            _save_candidate(
                output_dir=output_dir,
                index=index,
                origin="bc_mutated",
                seed=candidate_seed,
                policy=policy,
                source_checkpoint=str(bc_checkpoint.resolve()),
                mutation_std=config.mutation_std,
            )
        )
        used_vectors.append(vector)

    for local_index in range(config.policies_per_origin):
        index = config.policies_per_origin + local_index
        candidate_seed = config.seed + config.policies_per_origin + local_index
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(candidate_seed)
            policy = PolicyNetwork(bc_policy.hidden_size)
        vector = flatten_parameters(policy)
        entries.append(
            _save_candidate(
                output_dir=output_dir,
                index=index,
                origin="random",
                seed=candidate_seed,
                policy=policy,
                source_checkpoint=None,
                mutation_std=None,
            )
        )
        used_vectors.append(vector)

    if len({vector.numpy().tobytes() for vector in used_vectors}) != len(used_vectors):
        raise RuntimeError("population generation produced duplicate policies")
    manifest: dict[str, object] = {
        "schema_version": POPULATION_MANIFEST_SCHEMA_VERSION,
        "population_size": config.policies_per_origin * 2,
        "counts": {"bc_mutated": config.policies_per_origin, "random": config.policies_per_origin},
        "seed": config.seed,
        "mutation_std": config.mutation_std,
        "bc_checkpoint": str(bc_checkpoint.resolve()),
        "bc_checkpoint_sha256": _sha256(bc_checkpoint),
        "bc_training_metadata": bc_metadata,
        "policy": {**policy_metadata(bc_policy), "observation": observation_metadata()},
        "candidates": entries,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _save_candidate(
    *,
    output_dir: Path,
    index: int,
    origin: str,
    seed: int,
    policy: PolicyNetwork,
    source_checkpoint: str | None,
    mutation_std: float | None,
) -> dict[str, object]:
    filename = f"policy-{index:03d}-{origin}.pt"
    path = output_dir / filename
    metadata: dict[str, object] = {
        **policy_metadata(policy),
        "observation": observation_metadata(),
        "population_index": index,
        "origin": origin,
        "seed": seed,
        "source_checkpoint": source_checkpoint,
        "mutation_std": mutation_std,
    }
    policy.eval()
    torch.save(checkpoint_payload(policy, metadata=metadata), path)
    return {
        "index": index,
        "origin": origin,
        "seed": seed,
        "path": filename,
        "sha256": _sha256(path),
    }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bc-checkpoint", type=Path, default=Path("artifacts/imitation_model.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/evolution/generation-000"))
    parser.add_argument("--policies-per-origin", type=int, default=5)
    parser.add_argument("--mutation-std", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=110)
    args = parser.parse_args()
    manifest = generate_initial_population(
        args.bc_checkpoint,
        args.output_dir,
        PopulationConfig(args.policies_per_origin, args.mutation_std, args.seed),
    )
    print(json.dumps({"population_size": manifest["population_size"], "counts": manifest["counts"]}, indent=2))


if __name__ == "__main__":
    main()
