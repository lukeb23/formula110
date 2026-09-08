"""Train the shared Formula 110 policy with action mean-squared error."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from controllers.observation import observation_metadata
from controllers.policy_network import HIDDEN_SIZE, PolicyNetwork, checkpoint_payload, policy_metadata
from training.dataset import DemonstrationDataset, discover_human_trials, load_demonstrations

# PyTorch's optimizer and NumPy bridge contain intentionally dynamic annotations.
# pyright: reportUnknownMemberType=false


@dataclass(frozen=True)
class BehaviorCloningConfig:
    seed: int = 110
    epochs: int = 50
    batch_size: int = 256
    learning_rate: float = 1e-3
    validation_fraction: float = 0.2
    loss: str = "mse"
    hidden_size: int = HIDDEN_SIZE


def train_behavior_clone(
    dataset: DemonstrationDataset,
    config: BehaviorCloningConfig,
) -> tuple[PolicyNetwork, dict[str, object]]:
    """Train one policy, splitting complete sessions between train and validation."""
    if config.epochs < 1 or config.batch_size < 1 or config.learning_rate <= 0.0:
        raise ValueError("epochs, batch size, and learning rate must be positive")
    if not 0.0 <= config.validation_fraction < 1.0:
        raise ValueError("validation fraction must be in [0, 1)")
    if config.loss not in {"mse", "smooth_l1"}:
        raise ValueError("loss must be mse or smooth_l1")
    _seed_everything(config.seed)
    train_indices, validation_indices = _session_split(dataset.session_ids, config.validation_fraction, config.seed)
    train_observations = torch.from_numpy(dataset.observations[train_indices])
    train_actions = torch.from_numpy(dataset.actions[train_indices])
    validation_observations = torch.from_numpy(dataset.observations[validation_indices])
    validation_actions = torch.from_numpy(dataset.actions[validation_indices])

    policy = PolicyNetwork(config.hidden_size).to("cpu")
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    loss_function = nn.MSELoss() if config.loss == "mse" else nn.SmoothL1Loss(beta=0.5)
    loader_generator = torch.Generator(device="cpu").manual_seed(config.seed)
    loader = DataLoader(
        TensorDataset(train_observations, train_actions),
        batch_size=config.batch_size,
        shuffle=True,
        generator=loader_generator,
    )
    epoch_losses: list[float] = []
    policy.train()
    for _epoch in range(config.epochs):
        total_squared_error = 0.0
        total_values = 0
        for observations, actions in loader:
            optimizer.zero_grad(set_to_none=True)
            predictions = policy(observations)
            loss = loss_function(predictions, actions)
            loss.backward()
            optimizer.step()
            total_squared_error += float(nn.functional.mse_loss(predictions.detach(), actions)) * actions.numel()
            total_values += actions.numel()
        epoch_losses.append(total_squared_error / total_values)
    policy.eval()
    train_mse = _mse(policy, train_observations, train_actions)
    validation_mse = _mse(policy, validation_observations, validation_actions)
    metadata: dict[str, object] = {
        **policy_metadata(policy),
        "observation": observation_metadata(),
        "training": {
            "algorithm": f"behavior_cloning_action_{config.loss}",
            "loss": config.loss,
            "smooth_l1_beta": 0.5 if config.loss == "smooth_l1" else None,
            "seed": config.seed,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "validation_fraction": config.validation_fraction,
            "training_rows": len(train_indices),
            "validation_rows": len(validation_indices),
            "train_mse": train_mse,
            "validation_mse": validation_mse,
            "final_epoch_mse": epoch_losses[-1],
            "source_paths": list(dataset.source_paths),
            "source_sha256": [_sha256(Path(path)) for path in dataset.source_paths],
        },
    }
    return policy, metadata


def save_behavior_clone(
    path: Path,
    policy: PolicyNetwork,
    metadata: dict[str, object],
    *,
    overwrite: bool = False,
) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite checkpoint: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload(policy, metadata=metadata), path)


def _session_split(
    session_ids: tuple[str, ...], fraction: float, seed: int
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    unique_sessions = sorted(set(session_ids))
    shuffled = unique_sessions.copy()
    random.Random(seed).shuffle(shuffled)
    validation_count = 0 if fraction == 0.0 else max(1, round(len(shuffled) * fraction))
    if validation_count >= len(shuffled):
        validation_count = max(0, len(shuffled) - 1)
    validation_sessions = set(shuffled[:validation_count])
    train = np.asarray(
        [index for index, session in enumerate(session_ids) if session not in validation_sessions], dtype=np.int64
    )
    validation = np.asarray(
        [index for index, session in enumerate(session_ids) if session in validation_sessions], dtype=np.int64
    )
    return train, validation


def _mse(policy: PolicyNetwork, observations: Tensor, actions: Tensor) -> float | None:
    if observations.shape[0] == 0:
        return None
    with torch.inference_mode():
        return float(nn.functional.mse_loss(policy(observations), actions))


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("artifacts/human-driving"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/imitation_model.pt"))
    parser.add_argument("--seed", type=int, default=110)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--loss", choices=("mse", "smooth_l1"), default="mse")
    parser.add_argument("--hidden-size", type=int, default=HIDDEN_SIZE)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    paths = discover_human_trials(args.data_dir)
    dataset = load_demonstrations(paths)
    config = BehaviorCloningConfig(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validation_fraction=args.validation_fraction,
        loss=args.loss,
        hidden_size=args.hidden_size,
    )
    policy, metadata = train_behavior_clone(dataset, config)
    save_behavior_clone(args.output, policy, metadata, overwrite=args.overwrite)
    print(json.dumps(metadata["training"], indent=2))


if __name__ == "__main__":
    main()
