"""Deterministic policy fitness evaluation through Formula 110's public API."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite

import torch
from torch import Tensor

from controllers.observation import encode_observation
from controllers.policy_network import PolicyNetwork, policy_from_vector
from racing import HeadToHeadTeamRaceStats, RobotCommand, RobotSensors, run_headless_head_to_head

# PyTorch's NumPy bridge contains intentionally dynamic annotations.
# pyright: reportUnknownMemberType=false


@dataclass(frozen=True)
class FitnessConfig:
    """Weights for distance, fast completion, and vehicle-health objectives."""

    distance_weight: float = 1.0
    lap_completion_bonus: float = 100.0
    fast_lap_bonus: float = 100.0
    health_exponent: float = 2.0
    damage_penalty: float = 0.0
    wall_contact_penalty_per_s: float = 30.0
    low_progress_penalty_per_s: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.distance_weight,
            self.lap_completion_bonus,
            self.fast_lap_bonus,
            self.health_exponent,
            self.damage_penalty,
            self.wall_contact_penalty_per_s,
            self.low_progress_penalty_per_s,
        )
        if any(value < 0.0 for value in values):
            raise ValueError("fitness weights and health exponent cannot be negative")


DEFAULT_FITNESS_CONFIG = FitnessConfig()


@dataclass(frozen=True)
class FitnessResult:
    seed: int
    fitness: float
    scored_distance_m: float
    raw_distance_m: float
    lap_count: int
    best_lap_time_s: float | None
    max_speed_mps: float
    damage: float
    wall_contact_s: float
    car_contact_s: float
    low_progress_s: float
    marshal_count: int
    progress_before_health: float
    health_factor: float
    distance_component: float
    completion_component: float
    time_component: float
    damage_component: float
    wall_contact_component: float
    low_progress_component: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class PolicyController:
    """Adapt one in-memory policy to the public RobotController interface."""

    def __init__(self, policy: PolicyNetwork) -> None:
        self.policy = policy.to("cpu").eval()

    def __call__(self, sensors: RobotSensors) -> RobotCommand:
        observation = torch.from_numpy(encode_observation(sensors)).unsqueeze(0)
        with torch.inference_mode():
            steer, throttle = self.policy(observation)[0].tolist()
        return RobotCommand(steer=_bounded(steer), throttle=_bounded(throttle))


def evaluate_policy(
    weights: Tensor,
    seed: int,
    *,
    round_seconds: float = 20.0,
    fitness_config: FitnessConfig = DEFAULT_FITNESS_CONFIG,
) -> FitnessResult:
    """Evaluate one parameter vector and return its composite fitness metrics."""
    if round_seconds <= 0.0:
        raise ValueError("round_seconds must be positive")
    policy = policy_from_vector(weights)
    result = run_headless_head_to_head(
        challenger_controller=PolicyController(policy),
        incumbent_controller=_stationary_controller,
        challenger_name="candidate",
        incumbent_name="stationary benchmark",
        race_count=1,
        round_seconds=round_seconds,
        random_seed=seed,
    )
    return fitness_from_stats(
        result.races[0].challenger,
        seed=seed,
        round_seconds=round_seconds,
        config=fitness_config,
    )


def fitness_from_stats(
    stats: HeadToHeadTeamRaceStats,
    *,
    seed: int,
    round_seconds: float,
    config: FitnessConfig = DEFAULT_FITNESS_CONFIG,
) -> FitnessResult:
    """Calculate a transparent score from public race statistics."""
    scored_distance = stats.best_distance_m
    raw_distance = stats.best_raw_distance_m
    lap_count = stats.total_lap_count
    best_lap_time = stats.best_lap_time_seconds
    damage = stats.average_damage
    wall_contact = stats.total_wall_contact_seconds
    low_progress = stats.total_low_progress_seconds
    distance_component = config.distance_weight * scored_distance
    completion_component = config.lap_completion_bonus * lap_count
    time_component = 0.0
    if best_lap_time is not None:
        remaining_fraction = max(0.0, min(1.0, (round_seconds - best_lap_time) / round_seconds))
        time_component = config.fast_lap_bonus * remaining_fraction
    progress_before_health = distance_component + completion_component + time_component
    health_factor = max(0.0, 1.0 - damage) ** config.health_exponent
    damage_component = -config.damage_penalty * damage
    wall_contact_component = -config.wall_contact_penalty_per_s * wall_contact
    low_progress_component = -config.low_progress_penalty_per_s * low_progress
    fitness = (
        progress_before_health * health_factor
        + damage_component
        + wall_contact_component
        + low_progress_component
    )
    return FitnessResult(
        seed=seed,
        fitness=fitness,
        scored_distance_m=scored_distance,
        raw_distance_m=raw_distance,
        lap_count=lap_count,
        best_lap_time_s=best_lap_time,
        max_speed_mps=stats.max_speed_mps,
        damage=damage,
        wall_contact_s=wall_contact,
        car_contact_s=stats.total_car_contact_seconds,
        low_progress_s=low_progress,
        marshal_count=stats.total_marshal_count,
        progress_before_health=progress_before_health,
        health_factor=health_factor,
        distance_component=distance_component,
        completion_component=completion_component,
        time_component=time_component,
        damage_component=damage_component,
        wall_contact_component=wall_contact_component,
        low_progress_component=low_progress_component,
    )


def _stationary_controller(sensors: RobotSensors) -> RobotCommand:
    del sensors
    return RobotCommand()


def _bounded(value: float) -> float:
    if not isfinite(value):
        return 0.0
    return max(-1.0, min(1.0, float(value)))
