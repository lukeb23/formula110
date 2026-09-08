from pathlib import Path

import pytest
import torch

from controllers.observation import OBSERVATION_SIZE
from controllers.policy_network import (
    PolicyNetwork,
    checkpoint_payload,
    flatten_parameters,
    load_policy_checkpoint,
    parameter_count,
    parameter_mutation_mask,
    policy_from_vector,
    widen_policy,
)
from training.evolution import CandidateEvaluation, EvolutionConfig, create_next_generation

# pyright: reportUnknownMemberType=false


@pytest.mark.parametrize("width", [32, 128])
def test_checkpoint_and_vector_roundtrip_at_each_width(tmp_path: Path, width: int) -> None:
    policy = PolicyNetwork(width).eval()
    vector = flatten_parameters(policy)
    assert vector.numel() == parameter_count(width)
    restored = policy_from_vector(vector)
    assert restored.hidden_size == width
    assert torch.equal(vector, flatten_parameters(restored))
    path = tmp_path / "policy.pt"
    torch.save(checkpoint_payload(policy, metadata={}), path)
    loaded, metadata = load_policy_checkpoint(path)
    assert metadata["hidden_size"] == width
    assert torch.equal(vector, flatten_parameters(loaded))
    assert int(parameter_mutation_mask(policy, "all").sum()) == vector.numel()
    assert int(parameter_mutation_mask(policy, "steer_output").sum()) == width + 1


def test_widening_preserves_function_and_original_parameters() -> None:
    torch.manual_seed(19)
    source = PolicyNetwork().eval()
    original = flatten_parameters(source)
    wide = widen_policy(source, 128)
    observations = torch.randn(1000, OBSERVATION_SIZE)
    with torch.inference_mode():
        assert torch.allclose(source(observations), wide(observations), atol=1e-6)
    assert torch.equal(original, flatten_parameters(source))
    assert parameter_count(128) == 18434


def test_legacy_checkpoint_without_width_loads(tmp_path: Path) -> None:
    policy = PolicyNetwork()
    path = tmp_path / "legacy.pt"
    torch.save({"schema_version": 1, "model_state_dict": policy.state_dict(), "metadata": {}}, path)
    restored, _ = load_policy_checkpoint(path)
    assert restored.hidden_size == 32
    assert torch.equal(flatten_parameters(policy), flatten_parameters(restored))


def test_wide_evolution_keeps_elite_and_changes_offspring(tmp_path: Path) -> None:
    source = tmp_path / "parents"
    source.mkdir()
    policy = PolicyNetwork(128)
    torch.save(checkpoint_payload(policy, metadata={}), source / "parent.pt")
    directory = create_next_generation(
        [CandidateEvaluation(0, "parent.pt", "source", None, 1, ())],
        source_dir=source,
        output_dir=tmp_path / "generation-000",
        generation_index=0,
        population_size=3,
        config=EvolutionConfig(elite_count=1, mutation_std=0.0003),
    )
    elite, _ = load_policy_checkpoint(directory / "policy-000-elite.pt")
    child, metadata = load_policy_checkpoint(directory / "policy-001-mutated.pt")
    assert torch.equal(flatten_parameters(policy), flatten_parameters(elite))
    assert child.hidden_size == 128 and metadata["hidden_size"] == 128
    assert not torch.equal(flatten_parameters(policy), flatten_parameters(child))


def test_invalid_vector_and_shrinking_rejected() -> None:
    with pytest.raises(ValueError):
        policy_from_vector(torch.zeros(100))
    with pytest.raises(ValueError):
        widen_policy(PolicyNetwork(128), 32)
