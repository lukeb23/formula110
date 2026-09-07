from __future__ import annotations

import numpy as np

from controllers.observation import OBSERVATION_SIZE, encode_observation
from racing import CameraSensors, ImuSensors, LidarSensors, OdometrySensors, RobotSensors


def test_observation_has_authoritative_order_shape_and_dtype() -> None:
    sensors = RobotSensors(
        odometry=OdometrySensors(speed_mps=25.0),
        wall_lidar=LidarSensors(distances_m=(10.0, 99.0, 20.0, 30.0, 40.0, 98.0, 50.0)),
        camera=CameraSensors(
            center_offset_m=10.0,
            heading_error_degrees=90.0,
            lookahead_offsets_m=(-20.0, 0.0, 20.0),
        ),
        imu=ImuSensors(yaw_rate_degrees_per_s=-180.0),
    )

    observation = encode_observation(sensors)

    assert observation.dtype == np.float32
    assert observation.shape == (OBSERVATION_SIZE,)
    np.testing.assert_allclose(
        observation,
        (0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.5, -1.0, 0.0, 1.0, -0.5),
    )


def test_observation_replaces_lidar_infinity_and_clips_extremes() -> None:
    sensors = RobotSensors(
        odometry=OdometrySensors(speed_mps=float("inf")),
        wall_lidar=LidarSensors(distances_m=(float("inf"),) * 7),
        camera=CameraSensors(center_offset_m=1_000.0, heading_error_degrees=-1_000.0),
        imu=ImuSensors(yaw_rate_degrees_per_s=float("nan")),
    )

    observation = encode_observation(sensors)

    assert np.isfinite(observation).all()
    assert observation[0] == 0.0
    np.testing.assert_array_equal(observation[1:6], np.ones(5, dtype=np.float32))
    assert observation[6] == 1.0
    assert observation[7] == -1.0
    assert observation[11] == 0.0


def test_recorded_null_lidar_uses_cap() -> None:
    observation = encode_observation(
        {
            "odometry": {"speed_mps": 0.0},
            "wall_lidar": {"distances_m": [None] * 7},
            "camera": {"visible": False},
            "imu": {},
        }
    )

    np.testing.assert_array_equal(observation[1:6], np.ones(5, dtype=np.float32))
    np.testing.assert_array_equal(observation[6:11], np.zeros(5, dtype=np.float32))
