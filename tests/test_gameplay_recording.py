from __future__ import annotations

import json
from pathlib import Path

from racing.game.recording import ControllerGameplayRecorder, HumanGameplayRecorder, robot_sensors_to_dict
from racing.student.api import (
    CameraCompetitorReading,
    CameraSensors,
    ContactSensors,
    LidarSensors,
    OdometrySensors,
    RobotCommand,
    RobotSensors,
)


def demonstration_sensors() -> RobotSensors:
    return RobotSensors(
        dt_s=1 / 60,
        tick=12,
        odometry=OdometrySensors(speed_mps=4.5, distance_m=8.0),
        lidar=LidarSensors(
            angles_degrees=(0.0, 45.0),
            distances_m=(float("inf"), 3.5),
            max_distance_m=float("inf"),
        ),
        wall_lidar=LidarSensors(
            angles_degrees=(0.0,),
            distances_m=(7.0,),
            max_distance_m=float("inf"),
        ),
        camera=CameraSensors(
            center_offset_m=0.25,
            heading_error_degrees=-4.0,
            competitors=(CameraCompetitorReading(distance_m=5.0, angle_degrees=12.0),),
        ),
        contact=ContactSensors(wall=0.1, damage=0.2),
    )


def test_robot_sensors_to_dict_uses_null_for_infinite_lidar_values() -> None:
    record = robot_sensors_to_dict(demonstration_sensors())
    lidar = record["lidar"]

    assert isinstance(lidar, dict)
    assert lidar["distances_m"] == [None, 3.5]
    assert lidar["max_distance_m"] is None
    json.dumps(record, allow_nan=False)


def test_human_gameplay_recorder_writes_one_complete_json_object_per_line(tmp_path: Path) -> None:
    output = tmp_path / "demonstrations" / "human.jsonl"
    command = RobotCommand(throttle=0.75, steer=-0.25)

    with HumanGameplayRecorder(output) as recorder:
        recorder.record(simulation_time_s=0.2, sensors=demonstration_sensors(), command=command)
        session_id = recorder.session_id

    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["schema_version"] == 2
    assert record["record_type"] == "human_control_step"
    assert record["session_id"] == session_id
    assert record["simulation_time_s"] == 0.2
    assert record["sensors"]["tick"] == 12
    assert record["sensors"]["camera"]["competitors"][0]["distance_m"] == 5.0
    assert record["command"] == {"throttle": 0.75, "steer": -0.25}


def test_human_gameplay_recorder_appends_new_sessions(tmp_path: Path) -> None:
    output = tmp_path / "human.jsonl"
    sensors = demonstration_sensors()
    command = RobotCommand()

    with HumanGameplayRecorder(output) as first:
        first.record(simulation_time_s=0.1, sensors=sensors, command=command)
        first_session_id = first.session_id
    with HumanGameplayRecorder(output) as second:
        second.record(simulation_time_s=0.1, sensors=sensors, command=command)
        second_session_id = second.session_id

    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 2
    assert first_session_id != second_session_id
    assert [record["session_id"] for record in records] == [first_session_id, second_session_id]


def test_controller_gameplay_recorder_writes_controller_provenance(tmp_path: Path) -> None:
    output = tmp_path / "controller.jsonl"

    with ControllerGameplayRecorder(output, controller_module="path/to/controller.py") as recorder:
        recorder.record(
            simulation_time_s=0.2,
            sensors=demonstration_sensors(),
            command=RobotCommand(throttle=0.8, steer=0.1),
        )

    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["schema_version"] == 3
    assert record["record_type"] == "controller_control_step"
    assert record["control_source"] == {
        "type": "student_controller",
        "module": "path/to/controller.py",
        "function": "control",
    }
    assert record["command"] == {"throttle": 0.8, "steer": 0.1}
