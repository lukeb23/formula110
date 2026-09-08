"""Deploy the verified evolved champion using the shared CPU policy."""

from math import isfinite
from pathlib import Path

import torch

from controllers.observation import encode_observation
from controllers.policy_network import load_policy_checkpoint
from racing import RobotCommand, RobotSensors

# pyright: reportUnknownMemberType=false

RACING_NAME = "Evolved Champion"
MODEL_PATH = Path(__file__).resolve().parent / "model" / "champion-1081.pt"


class Controller:
    def __init__(self) -> None:
        self.policy, _ = load_policy_checkpoint(MODEL_PATH)

    def __call__(self, sensors: RobotSensors) -> RobotCommand:
        observation = torch.from_numpy(encode_observation(sensors)).unsqueeze(0)
        with torch.inference_mode():
            steer, throttle = self.policy(observation)[0].tolist()
        return RobotCommand(steer=_bounded(steer), throttle=_bounded(throttle))


def _bounded(value: float) -> float:
    return max(-1.0, min(1.0, value)) if isfinite(value) else 0.0


def create_controller() -> Controller:
    return Controller()
