"""Authoritative observation encoding shared by training and controllers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from racing import RobotSensors

OBSERVATION_FIELDS: tuple[str, ...] = (
    "speed_mps",
    "wall_lidar_left",
    "wall_lidar_front_left",
    "wall_lidar_front",
    "wall_lidar_front_right",
    "wall_lidar_right",
    "camera_center_offset_m",
    "camera_heading_error_degrees",
    "camera_lookahead_offset_0",
    "camera_lookahead_offset_1",
    "camera_lookahead_offset_2",
    "imu_yaw_rate_degrees_per_s",
)
OBSERVATION_SIZE = len(OBSERVATION_FIELDS)

SPEED_ABS_MAX_MPS = 50.0
LIDAR_CAP_M = 100.0
CAMERA_OFFSET_ABS_MAX_M = 20.0
HEADING_ERROR_ABS_MAX_DEGREES = 180.0
YAW_RATE_ABS_MAX_DEGREES_PER_S = 360.0

Observation = NDArray[np.float32]
SensorRecord: TypeAlias = Mapping[str, object]


def encode_observation(sensors: RobotSensors | SensorRecord) -> Observation:
    """Encode a live snapshot or its JSON representation as 12 finite float32 values."""
    if isinstance(sensors, RobotSensors):
        speed = sensors.odometry.speed_mps
        lidar = (
            sensors.wall_lidar.left_m,
            sensors.wall_lidar.front_left_m,
            sensors.wall_lidar.front_m,
            sensors.wall_lidar.front_right_m,
            sensors.wall_lidar.right_m,
        )
        camera_visible = sensors.camera.visible
        center_offset = sensors.camera.center_offset_m
        heading_error = sensors.camera.heading_error_degrees
        lookahead = sensors.camera.lookahead_offsets_m
        yaw_rate = sensors.imu.yaw_rate_degrees_per_s
    else:
        odometry = _mapping(sensors.get("odometry"))
        wall_lidar = _mapping(sensors.get("wall_lidar"))
        camera = _mapping(sensors.get("camera"))
        imu = _mapping(sensors.get("imu"))
        speed = _number(odometry.get("speed_mps"))
        lidar = tuple(
            _recorded_lidar_distance(wall_lidar, target_angle) for target_angle in (-90.0, -20.0, 0.0, 20.0, 90.0)
        )
        camera_visible = bool(camera.get("visible", False))
        center_offset = _number(camera.get("center_offset_m"))
        heading_error = _number(camera.get("heading_error_degrees"))
        lookahead = _sequence(camera.get("lookahead_offsets_m"))
        yaw_rate = _number(imu.get("yaw_rate_degrees_per_s"))

    if not camera_visible:
        center_offset = 0.0
        heading_error = 0.0
        lookahead = ()

    values = (
        _signed_scale(speed, SPEED_ABS_MAX_MPS),
        *(_lidar_scale(value) for value in lidar),
        _signed_scale(center_offset, CAMERA_OFFSET_ABS_MAX_M),
        _signed_scale(heading_error, HEADING_ERROR_ABS_MAX_DEGREES),
        *(_signed_scale(_number_at(lookahead, index), CAMERA_OFFSET_ABS_MAX_M) for index in range(3)),
        _signed_scale(yaw_rate, YAW_RATE_ABS_MAX_DEGREES_PER_S),
    )
    encoded = np.asarray(values, dtype=np.float32)
    if encoded.shape != (OBSERVATION_SIZE,) or not np.isfinite(encoded).all():
        raise ValueError("observation encoder produced an invalid vector")
    return encoded


def observation_metadata() -> dict[str, object]:
    """Return the stable representation metadata stored with model artifacts."""
    return {
        "fields": list(OBSERVATION_FIELDS),
        "size": OBSERVATION_SIZE,
        "speed_abs_max_mps": SPEED_ABS_MAX_MPS,
        "lidar_cap_m": LIDAR_CAP_M,
        "camera_offset_abs_max_m": CAMERA_OFFSET_ABS_MAX_M,
        "heading_error_abs_max_degrees": HEADING_ERROR_ABS_MAX_DEGREES,
        "yaw_rate_abs_max_degrees_per_s": YAW_RATE_ABS_MAX_DEGREES_PER_S,
    }


def _mapping(value: object) -> SensorRecord:
    return cast(SensorRecord, value) if isinstance(value, Mapping) else {}


def _sequence(value: object) -> Sequence[object]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return cast(Sequence[object], value)
    return ()


def _number(value: object, *, default: float = 0.0) -> float:
    if value is None or isinstance(value, bool):
        return default
    try:
        result = float(cast(float | int | str, value))
    except (TypeError, ValueError):
        return default
    return result if isfinite(result) else default


def _number_at(values: Sequence[object], index: int, *, default: float = 0.0) -> float:
    return _number(values[index], default=default) if index < len(values) else default


def _recorded_lidar_distance(lidar: SensorRecord, target_angle: float) -> float:
    angles = _sequence(lidar.get("angles_degrees"))
    distances = _sequence(lidar.get("distances_m"))
    if not angles:
        angles = (-90.0, -45.0, -20.0, 0.0, 20.0, 45.0, 90.0)
    usable_count = min(len(angles), len(distances))
    if usable_count == 0:
        return LIDAR_CAP_M
    index = min(range(usable_count), key=lambda item: abs(_number(angles[item]) - target_angle))
    return _number_at(distances, index, default=LIDAR_CAP_M)


def _signed_scale(value: object, maximum: float) -> float:
    finite = _number(value)
    return max(-maximum, min(maximum, finite)) / maximum


def _lidar_scale(value: object) -> float:
    # JSON no-hit values are null; live no-hit values are positive infinity.
    finite = _number(value, default=LIDAR_CAP_M)
    return max(0.0, min(LIDAR_CAP_M, finite)) / LIDAR_CAP_M
