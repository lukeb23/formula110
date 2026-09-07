from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch

from controllers.policy_network import (
    PolicyNetwork,
    checkpoint_payload,
    flatten_parameters,
    load_parameter_vector,
    load_policy_checkpoint,
    parameter_count,
)
from racing import CameraSensors, RobotSensors
from training.behavior_clone import BehaviorCloningConfig, train_behavior_clone
from training.dataset import DemonstrationDataset, load_demonstrations
from training.population import PopulationConfig, generate_initial_population
from training.watch_policy import CheckpointController

# PyTorch's seed helper is incompletely annotated upstream.
# pyright: reportUnknownMemberType=false


def test_policy_outputs_are_finite_and_bounded() -> None:
    policy = PolicyNetwork()

    with torch.inference_mode():
        actions = policy(torch.full((4, 12), 1e20))

    assert actions.shape == (4, 2)
    assert torch.isfinite(actions).all()
    assert torch.all(actions >= -1.0)
    assert torch.all(actions <= 1.0)


def test_parameter_vector_round_trip_preserves_values_and_shapes() -> None:
    source = PolicyNetwork()
    vector = flatten_parameters(source)
    restored = PolicyNetwork()

    load_parameter_vector(restored, vector)

    assert vector.numel() == parameter_count()
    source_shapes = [tuple(value.shape) for value in source.parameters()]
    restored_shapes = [tuple(value.shape) for value in restored.parameters()]
    assert source_shapes == restored_shapes
    assert torch.equal(vector, flatten_parameters(restored))


def test_behavior_clone_training_is_reproducible() -> None:
    observations = np.zeros((8, 12), dtype=np.float32)
    observations[:, 7] = np.linspace(-1.0, 1.0, 8, dtype=np.float32)
    actions = np.column_stack((observations[:, 7], np.full(8, 0.5, dtype=np.float32))).astype(np.float32)
    dataset = DemonstrationDataset(
        observations=observations,
        actions=actions,
        session_ids=("a", "a", "b", "b", "c", "c", "d", "d"),
        source_paths=(),
    )
    config = BehaviorCloningConfig(seed=31, epochs=2, batch_size=2, validation_fraction=0.25)

    first, first_metadata = train_behavior_clone(dataset, config)
    second, second_metadata = train_behavior_clone(dataset, config)

    assert torch.equal(flatten_parameters(first), flatten_parameters(second))
    assert first_metadata["training"] == second_metadata["training"]


def test_population_is_equal_independent_and_reproducible(tmp_path: Path) -> None:
    bc_path = tmp_path / "bc.pt"
    torch.manual_seed(7)
    bc_policy = PolicyNetwork()
    torch.save(checkpoint_payload(bc_policy, metadata={"training": "test"}), bc_path)
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    config = PopulationConfig(policies_per_origin=2, mutation_std=0.05, seed=23)

    first = generate_initial_population(bc_path, first_dir, config)
    second = generate_initial_population(bc_path, second_dir, config)

    assert first["counts"] == {"bc_mutated": 2, "random": 2}
    first_candidates = cast(list[dict[str, object]], first["candidates"])
    second_candidates = cast(list[dict[str, object]], second["candidates"])
    first_vectors = [
        flatten_parameters(load_policy_checkpoint(first_dir / cast(str, entry["path"]))[0])
        for entry in first_candidates
    ]
    second_vectors = [
        flatten_parameters(load_policy_checkpoint(second_dir / cast(str, entry["path"]))[0])
        for entry in second_candidates
    ]
    assert all(torch.equal(left, right) for left, right in zip(first_vectors, second_vectors, strict=True))
    assert all(
        not torch.equal(left, right) for index, left in enumerate(first_vectors) for right in first_vectors[index + 1 :]
    )
    assert all(not torch.equal(flatten_parameters(bc_policy), vector) for vector in first_vectors[:2])


def test_population_refuses_to_overwrite(tmp_path: Path) -> None:
    bc_path = tmp_path / "bc.pt"
    torch.save(checkpoint_payload(PolicyNetwork(), metadata={}), bc_path)
    output = tmp_path / "population"
    output.mkdir()
    (output / "keep.txt").write_text("user data", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        generate_initial_population(bc_path, output, PopulationConfig(policies_per_origin=1))


def test_checkpoint_controller_maps_steer_then_throttle(tmp_path: Path) -> None:
    checkpoint = tmp_path / "policy.pt"
    policy = PolicyNetwork()
    vector = torch.zeros(parameter_count())
    vector[-2:] = torch.tensor((0.25, 0.75))
    load_parameter_vector(policy, vector)
    torch.save(checkpoint_payload(policy, metadata={}), checkpoint)
    controller = CheckpointController(checkpoint)

    command = controller(RobotSensors(camera=CameraSensors()))

    assert command.steer == pytest.approx(float(torch.tanh(torch.tensor(0.25))))
    assert command.throttle == pytest.approx(float(torch.tanh(torch.tensor(0.75))))


def test_dataset_action_order_is_steer_then_throttle(tmp_path: Path) -> None:
    path = tmp_path / "trial.jsonl"
    path.write_text(
        json.dumps(
            {
                "record_type": "human_control_step",
                "session_id": "session",
                "sensors": {},
                "command": {"throttle": 0.75, "steer": 0.25},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = load_demonstrations([path])

    assert dataset.actions[0].tolist() == pytest.approx([0.25, 0.75])
