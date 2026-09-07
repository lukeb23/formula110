"""Run one saved policy checkpoint in the graphical Formula 110 simulator."""

from __future__ import annotations

import argparse
from math import isfinite
from pathlib import Path

import torch

from controllers.observation import encode_observation
from controllers.policy_network import load_policy_checkpoint
from racing import CameraView, GameConfig, RacingAudioConfig, RobotCommand, RobotSensors, create_app

# PyTorch's NumPy bridge contains intentionally dynamic annotations.
# pyright: reportUnknownMemberType=false


class CheckpointController:
    """CPU controller adapter for inspecting a training checkpoint."""

    def __init__(self, checkpoint_path: Path) -> None:
        self.policy, self.metadata = load_policy_checkpoint(checkpoint_path)

    def __call__(self, sensors: RobotSensors) -> RobotCommand:
        observation = torch.from_numpy(encode_observation(sensors)).unsqueeze(0)
        with torch.inference_mode():
            steer, throttle = self.policy(observation)[0].tolist()
        return RobotCommand(
            throttle=_finite_clamped(float(throttle)),
            steer=_finite_clamped(float(steer)),
        )


def _finite_clamped(value: float) -> float:
    if not isfinite(value):
        return 0.0
    return max(-1.0, min(1.0, value))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=110)
    parser.add_argument("--camera", choices=[view.value for view in CameraView], default=CameraView.FOLLOW.value)
    parser.add_argument("--muted", action="store_true")
    args = parser.parse_args()
    controller = CheckpointController(args.checkpoint)
    create_app(
        GameConfig(
            title=f"Formula 110 Policy: {args.checkpoint.name}",
            student_controller=controller,
            random_seed=args.seed,
            camera_view=CameraView(args.camera),
            audio=RacingAudioConfig(muted=args.muted),
        )
    ).run()


if __name__ == "__main__":
    main()
