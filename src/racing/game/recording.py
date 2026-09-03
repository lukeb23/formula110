"""JSON Lines recording for human and controller trajectories."""

from __future__ import annotations

import json
from atexit import register, unregister
from math import isfinite
from pathlib import Path
from typing import Self, TextIO
from uuid import uuid4

from racing.student.api import LidarSensors, RobotCommand, RobotSensors

HUMAN_GAMEPLAY_SCHEMA_VERSION = 2
CONTROLLER_GAMEPLAY_SCHEMA_VERSION = 3


def robot_command_to_dict(command: RobotCommand) -> dict[str, float]:
    """Return the normalized controller output fields as JSON-compatible values."""
    return {
        "throttle": float(command.throttle),
        "steer": float(command.steer),
    }


def robot_sensors_to_dict(sensors: RobotSensors) -> dict[str, object]:
    """Return the complete public sensor snapshot as JSON-compatible values.

    Infinite LiDAR values mean no object was detected and are represented as
    JSON ``null`` so every emitted line remains standards-compliant JSON.
    """
    return {
        "dt_s": sensors.dt_s,
        "tick": sensors.tick,
        "imu": {
            "heading_degrees": sensors.imu.heading_degrees,
            "yaw_rate_degrees_per_s": sensors.imu.yaw_rate_degrees_per_s,
            "pitch_degrees": sensors.imu.pitch_degrees,
            "roll_degrees": sensors.imu.roll_degrees,
            "forward_acceleration_mps2": sensors.imu.forward_acceleration_mps2,
            "lateral_acceleration_mps2": sensors.imu.lateral_acceleration_mps2,
        },
        "odometry": {
            "speed_mps": sensors.odometry.speed_mps,
            "distance_m": sensors.odometry.distance_m,
        },
        "lidar": _lidar_sensors_to_dict(sensors.lidar),
        "wall_lidar": _lidar_sensors_to_dict(sensors.wall_lidar),
        "camera": {
            "visible": sensors.camera.visible,
            "center_offset_m": sensors.camera.center_offset_m,
            "heading_error_degrees": sensors.camera.heading_error_degrees,
            "lookahead_offsets_m": list(sensors.camera.lookahead_offsets_m),
            "lookahead_distances_m": list(sensors.camera.lookahead_distances_m),
            "competitors": [
                {
                    "distance_m": competitor.distance_m,
                    "angle_degrees": competitor.angle_degrees,
                    "relative_heading_degrees": competitor.relative_heading_degrees,
                    "speed_mps": competitor.speed_mps,
                    "closing_speed_mps": competitor.closing_speed_mps,
                }
                for competitor in sensors.camera.competitors
            ],
        },
        "contact": {
            "wall": sensors.contact.wall,
            "robot": sensors.contact.robot,
            "any_contact": sensors.contact.any_contact,
            "damage": sensors.contact.damage,
        },
    }


def human_gameplay_record(
    *,
    session_id: str,
    simulation_time_s: float,
    sensors: RobotSensors,
    command: RobotCommand,
) -> dict[str, object]:
    """Build one versioned observation/action record for a human control tick."""
    return {
        "schema_version": HUMAN_GAMEPLAY_SCHEMA_VERSION,
        "record_type": "human_control_step",
        "session_id": session_id,
        "simulation_time_s": simulation_time_s,
        "sensors": robot_sensors_to_dict(sensors),
        "command": robot_command_to_dict(command),
    }


def controller_gameplay_record(
    *,
    session_id: str,
    simulation_time_s: float,
    sensors: RobotSensors,
    command: RobotCommand,
    controller_module: str,
    control_function: str,
) -> dict[str, object]:
    """Build one versioned observation/action record from a student controller."""
    return {
        "schema_version": CONTROLLER_GAMEPLAY_SCHEMA_VERSION,
        "record_type": "controller_control_step",
        "session_id": session_id,
        "simulation_time_s": simulation_time_s,
        "control_source": {
            "type": "student_controller",
            "module": controller_module,
            "function": control_function,
        },
        "sensors": robot_sensors_to_dict(sensors),
        "command": robot_command_to_dict(command),
    }


class _GameplayRecorder:
    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.session_id = uuid4().hex
        self._stream: TextIO | None = self.path.open("a", encoding="utf-8", buffering=1)
        register(self.close)

    def close(self) -> None:
        """Flush and close the recording file."""
        if self._stream is None:
            return
        self._stream.close()
        self._stream = None
        unregister(self.close)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exception: object) -> None:
        self.close()


class HumanGameplayRecorder(_GameplayRecorder):
    """Append human observation/action pairs to a line-buffered JSONL file."""

    def record(self, *, simulation_time_s: float, sensors: RobotSensors, command: RobotCommand) -> None:
        """Append and flush one control-tick record."""
        if self._stream is None:
            raise ValueError("human gameplay recorder is closed")
        record = human_gameplay_record(
            session_id=self.session_id,
            simulation_time_s=simulation_time_s,
            sensors=sensors,
            command=command,
        )
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        self._stream.write(line + "\n")
        self._stream.flush()


class ControllerGameplayRecorder(_GameplayRecorder):
    """Append controller observation/action pairs with source provenance."""

    def __init__(self, path: Path, *, controller_module: str, control_function: str = "control") -> None:
        super().__init__(path)
        self.controller_module = controller_module
        self.control_function = control_function

    def record(self, *, simulation_time_s: float, sensors: RobotSensors, command: RobotCommand) -> None:
        """Append and flush one controller tick."""
        if self._stream is None:
            raise ValueError("controller gameplay recorder is closed")
        record = controller_gameplay_record(
            session_id=self.session_id,
            simulation_time_s=simulation_time_s,
            sensors=sensors,
            command=command,
            controller_module=self.controller_module,
            control_function=self.control_function,
        )
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        self._stream.write(line + "\n")
        self._stream.flush()


def _lidar_sensors_to_dict(lidar: LidarSensors) -> dict[str, object]:
    return {
        "angles_degrees": list(lidar.angles_degrees),
        "distances_m": [_finite_float_or_none(distance_m) for distance_m in lidar.distances_m],
        "max_distance_m": _finite_float_or_none(lidar.max_distance_m),
    }


def _finite_float_or_none(value: float) -> float | None:
    return float(value) if isfinite(value) else None
