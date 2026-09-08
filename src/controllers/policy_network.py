"""The single policy-network definition used by BC, evolution, and deployment."""

from __future__ import annotations

from collections.abc import Mapping
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

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(OBSERVATION_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, ACTION_SIZE),
            nn.Tanh(),
        )

    def forward(self, observations: Tensor) -> Tensor:
        return self.layers(observations)


def parameter_count() -> int:
    return sum(parameter.numel() for parameter in PolicyNetwork().parameters())


def flatten_parameters(policy: PolicyNetwork) -> Tensor:
    """Flatten parameters in PyTorch registration order into an independent CPU vector."""
    return torch.cat([parameter.detach().cpu().reshape(-1) for parameter in policy.parameters()]).clone()


def load_parameter_vector(policy: PolicyNetwork, vector: Tensor) -> None:
    """Load a vector using the same deterministic parameter order as flatten_parameters."""
    flat = vector.detach().cpu().to(dtype=torch.float32).reshape(-1)
    if flat.numel() != parameter_count():
        raise ValueError(f"expected {parameter_count()} parameters, got {flat.numel()}")
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
        return torch.ones(parameter_count(), dtype=torch.bool)

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


def policy_metadata() -> dict[str, object]:
    return {
        "architecture": POLICY_ARCHITECTURE,
        "action_fields": list(ACTION_FIELDS),
        "parameter_count": parameter_count(),
    }


def checkpoint_payload(policy: PolicyNetwork, *, metadata: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_state_dict": {name: value.detach().cpu() for name, value in policy.state_dict().items()},
        "metadata": dict(metadata),
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
    policy = PolicyNetwork()
    policy.load_state_dict(cast(StateDict, state), strict=True)
    policy.to("cpu")
    policy.eval()
    return policy, cast(dict[str, object], metadata)
