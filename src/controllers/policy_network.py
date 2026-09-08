"""The single policy-network definition used by BC, evolution, and deployment."""

from __future__ import annotations

from collections.abc import Mapping
from math import isqrt
from pathlib import Path
from typing import TypeAlias, cast

import torch
from torch import Tensor, nn

from controllers.observation import OBSERVATION_SIZE

# PyTorch intentionally exposes several dynamically typed serialization APIs.
# pyright: reportUnknownMemberType=false

ACTION_FIELDS: tuple[str, ...] = ("steer", "throttle")
ACTION_SIZE = len(ACTION_FIELDS)
PARAMETER_MUTATION_SCOPES: tuple[str, ...] = ("all", "output", "steer_output", "throttle_output")
HIDDEN_SIZE = 32
POLICY_ARCHITECTURE = "12->32->ReLU->32->ReLU->2->tanh"
CHECKPOINT_SCHEMA_VERSION = 1
StateDict: TypeAlias = dict[str, Tensor]


class PolicyNetwork(nn.Module):
    """Small bounded-action MLP shared by every policy origin."""

    def __init__(self, hidden_size: int = HIDDEN_SIZE) -> None:
        super().__init__()
        if hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        self.hidden_size = hidden_size
        self.layers = nn.Sequential(
            nn.Linear(OBSERVATION_SIZE, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, ACTION_SIZE),
            nn.Tanh(),
        )

    def forward(self, observations: Tensor) -> Tensor:
        return self.layers(observations)


def parameter_count(hidden_size: int = HIDDEN_SIZE) -> int:
    return hidden_size**2 + (OBSERVATION_SIZE + ACTION_SIZE + 2) * hidden_size + ACTION_SIZE


def flatten_parameters(policy: PolicyNetwork) -> Tensor:
    """Flatten parameters in PyTorch registration order into an independent CPU vector."""
    return torch.cat([parameter.detach().cpu().reshape(-1) for parameter in policy.parameters()]).clone()


def load_parameter_vector(policy: PolicyNetwork, vector: Tensor) -> None:
    """Load a vector using the same deterministic parameter order as flatten_parameters."""
    flat = vector.detach().cpu().to(dtype=torch.float32).reshape(-1)
    expected = parameter_count(policy.hidden_size)
    if flat.numel() != expected:
        raise ValueError(f"expected {expected} parameters, got {flat.numel()}")
    offset = 0
    with torch.no_grad():
        for parameter in policy.parameters():
            count = parameter.numel()
            parameter.copy_(flat[offset : offset + count].reshape(parameter.shape))
            offset += count


def parameter_mutation_mask(policy: PolicyNetwork, scope: str) -> Tensor:
    """Return a flat mask for a named portion of the shared policy parameters."""
    if scope not in PARAMETER_MUTATION_SCOPES:
        valid = ", ".join(PARAMETER_MUTATION_SCOPES)
        raise ValueError(f"unknown mutation scope {scope!r}; expected one of: {valid}")
    if scope == "all":
        return torch.ones(parameter_count(policy.hidden_size), dtype=torch.bool)

    output_layer = next(layer for layer in reversed(policy.layers) if isinstance(layer, nn.Linear))
    action_index = None
    if scope.endswith("_output") and scope != "output":
        action_index = ACTION_FIELDS.index(scope.removesuffix("_output"))

    masks: list[Tensor] = []
    for parameter in policy.parameters():
        mask = torch.zeros(parameter.shape, dtype=torch.bool)
        if parameter is output_layer.weight:
            if action_index is None:
                mask.fill_(True)
            else:
                mask[action_index].fill_(True)
        elif parameter is output_layer.bias:
            if action_index is None:
                mask.fill_(True)
            else:
                mask[action_index] = True
        masks.append(mask.reshape(-1))
    return torch.cat(masks)


def policy_metadata(policy: PolicyNetwork | None = None) -> dict[str, object]:
    width = HIDDEN_SIZE if policy is None else policy.hidden_size
    return {
        "architecture": f"{OBSERVATION_SIZE}->{width}->ReLU->{width}->ReLU->{ACTION_SIZE}->tanh",
        "hidden_size": width,
        "action_fields": list(ACTION_FIELDS),
        "parameter_count": parameter_count(width),
    }


def checkpoint_payload(policy: PolicyNetwork, *, metadata: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_state_dict": {name: value.detach().cpu() for name, value in policy.state_dict().items()},
        "metadata": {**metadata, **policy_metadata(policy)},
    }


def load_policy_checkpoint(path: Path) -> tuple[PolicyNetwork, dict[str, object]]:
    loaded: object = torch.load(path, map_location="cpu", weights_only=True)
    raw = cast(dict[str, object], loaded) if isinstance(loaded, dict) else {}
    if raw.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"unsupported policy checkpoint: {path}")
    state = raw.get("model_state_dict")
    metadata = raw.get("metadata")
    if not isinstance(state, dict) or not isinstance(metadata, dict):
        raise ValueError(f"malformed policy checkpoint: {path}")
    typed_state = cast(StateDict, state)
    first_weight = typed_state.get("layers.0.weight")
    if not isinstance(first_weight, Tensor) or first_weight.ndim != 2:
        raise ValueError(f"malformed first layer: {path}")
    width = first_weight.shape[0]
    if metadata.get("hidden_size", width) != width:
        raise ValueError(f"checkpoint width metadata does not match weights: {path}")
    policy = PolicyNetwork(width)
    policy.load_state_dict(cast(StateDict, state), strict=True)
    policy.to("cpu")
    policy.eval()
    return policy, cast(dict[str, object], metadata)


def policy_from_vector(vector: Tensor) -> PolicyNetwork:
    """Infer the shared equal-hidden-layer architecture from its parameter count."""
    coefficient = OBSERVATION_SIZE + ACTION_SIZE + 2
    discriminant = coefficient**2 + 4 * (vector.numel() - ACTION_SIZE)
    if discriminant < 0:
        raise ValueError("invalid policy vector length")
    width = (isqrt(discriminant) - coefficient) // 2
    if width < 1 or parameter_count(width) != vector.numel():
        raise ValueError("invalid policy vector length")
    policy = PolicyNetwork(width)
    load_parameter_vector(policy, vector)
    return policy


def widen_policy(policy: PolicyNetwork, hidden_size: int) -> PolicyNetwork:
    """Replicate neurons and split outgoing weights to preserve the learned function."""
    if hidden_size < policy.hidden_size or hidden_size % policy.hidden_size:
        raise ValueError("new width must be a positive integer multiple of the source width")
    factor = hidden_size // policy.hidden_size
    widened = PolicyNetwork(hidden_size)
    old = [layer for layer in policy.layers if isinstance(layer, nn.Linear)]
    new = [layer for layer in widened.layers if isinstance(layer, nn.Linear)]
    with torch.no_grad():
        new[0].weight.copy_(old[0].weight.repeat(factor, 1))
        new[0].bias.copy_(old[0].bias.repeat(factor))
        new[1].weight.copy_(old[1].weight.repeat(factor, factor) / factor)
        new[1].bias.copy_(old[1].bias.repeat(factor))
        new[2].weight.copy_(old[2].weight.repeat(1, factor) / factor)
        new[2].bias.copy_(old[2].bias)
    return widened.eval()
