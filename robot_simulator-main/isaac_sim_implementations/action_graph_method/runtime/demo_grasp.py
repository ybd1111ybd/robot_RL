"""Demo grasp overlay controller for the Action Graph mainline."""

from __future__ import annotations

from enum import Enum
import time
from typing import Any, Optional, Tuple

import numpy as np

from runtime.mainline_shared import (
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    LEFT_GRIPPER_JOINTS,
    RIGHT_GRIPPER_JOINTS,
    _normalize_quaternion,
    _quat_conjugate,
    _quat_rotate_vector,
)


class DemoGraspState(Enum):
    WAITING = "waiting"
    APPROACH = "approach"
    DESCEND = "descend"
    GRASP = "grasp"
    LIFT = "lift"
    MOVE = "move"
    PLACE = "place"
    OPEN = "open"
    RETURN = "return"


class MainlineGraspDemo:
    def __init__(
        self,
        world: Any,
        articulation: Any,
        ik_bridge: Any,
        *,
        arm: str,
        auto_start: bool,
        assist_attach: bool,
        trigger_key: str,
        block_base: Optional[np.ndarray],
        place_base: Optional[np.ndarray],
        track_orientation: bool,
        headless: bool,
    ) -> None:
        self._world = world
        self._articulation = articulation
        self._ik_bridge = ik_bridge
        self._arm = arm
        self._auto_start = bool(auto_start or headless)
        self._assist_attach = bool(assist_attach)
        self._trigger_key = (trigger_key or "Q").strip().upper() or "Q"
        self._track_orientation = bool(track_orientation)
        self._headless = bool(headless)

        self._state = DemoGraspState.WAITING
        self._state_started_at = time.time()
        self._cycle_count = 0
        self._pending_start = False
        self._scene_ready = False
        self._keyboard_sub = None
        self._warmup_stable_count = 0
        self._last_observed_ee_pos_base: Optional[np.ndarray] = None
        self._cycle_block_base: Optional[np.ndarray] = None

        self._home_ee_pos_base: Optional[np.ndarray] = None
        self._home_ee_quat_base: Optional[np.ndarray] = None
        self._block_home_base = (
            None if block_base is None else np.asarray(block_base, dtype=float)
        )
        self._place_base = None if place_base is None else np.asarray(place_base, dtype=float)

        self._block = None
        self._place_marker = None
        self._attached = False
        self._attach_offset_local = np.zeros(3, dtype=float)
        self._gripper_joint_names = (
            LEFT_GRIPPER_JOINTS if self._arm == "left" else RIGHT_GRIPPER_JOINTS
        )
        self._gripper_joint_indices = np.array([], dtype=np.int32)
        self._gripper_open_targets = np.array(
            [GRIPPER_OPEN["narrow"], GRIPPER_OPEN["wide"]],
            dtype=np.float32,
        )
        self._gripper_closed_targets = np.array(
            [GRIPPER_CLOSED["narrow"], GRIPPER_CLOSED["wide"]],
            dtype=np.float32,
        )

        self._block_size = 0.04
        self._approach_height = 0.10
        self._grasp_height_offset = 0.012
        self._lift_height = 0.14
        self._reach_threshold = 0.018
        self._approach_reach_threshold = 0.03
        self._descend_reach_threshold = 0.025
        self._attach_threshold = 0.045
        self._close_hold_sec = 0.75
        self._open_hold_sec = 0.80

        print(
            "[demo-grasp] Enabled:"
            f" arm={self._arm}"
            f" auto_start={self._auto_start}"
            f" assist_attach={self._assist_attach}"
            f" target_frame=base_link"
        )
        if not self._headless:
            print(
                "[demo-grasp] Click the Isaac viewport, then press"
                f" {self._trigger_key} / Enter / Space to start one grasp cycle."
            )

    def _transition(self, state: DemoGraspState, message: str) -> None:
        self._state = state
        self._state_started_at = time.time()
        print(f"[demo-grasp] {message}")

    def _setup_keyboard(self) -> None:
        if self._headless or self._keyboard_sub is not None:
            return
        try:
            import carb
            import omni.appwindow
        except Exception as exc:
            print(f"[demo-grasp] Keyboard input unavailable: {exc}")
            return

        appwindow = omni.appwindow.get_default_app_window()
        if appwindow is None:
            print("[demo-grasp] Keyboard input unavailable: no app window.")
            return
        keyboard = appwindow.get_keyboard()
        if keyboard is None:
            print("[demo-grasp] Keyboard input unavailable: no keyboard handle.")
            return
        input_interface = carb.input.acquire_input_interface()

        trigger_names = {self._trigger_key, "ENTER", "SPACE"}

        def on_keyboard_event(event):
            if event.type not in (
                carb.input.KeyboardEventType.KEY_PRESS,
                carb.input.KeyboardEventType.KEY_REPEAT,
            ):
                return True
            key_name = getattr(event.input, "name", str(event.input)).upper()
            if key_name in trigger_names:
                self._pending_start = True
                print(f"[demo-grasp] Trigger received: key={key_name}")
            return True

        self._keyboard_sub = input_interface.subscribe_to_keyboard_events(
            keyboard, on_keyboard_event
        )

    def _cache_gripper_indices(self) -> None:
        dof_names = list(getattr(self._articulation, "dof_names", []) or [])
        if not dof_names:
            return
        name_to_index = {name: idx for idx, name in enumerate(dof_names)}
        indices = []
        for joint_name in self._gripper_joint_names:
            idx = name_to_index.get(joint_name)
            if idx is not None:
                indices.append(int(idx))
        self._gripper_joint_indices = np.asarray(indices, dtype=np.int32)
        if self._gripper_joint_indices.size != len(self._gripper_joint_names):
            print(
                "[demo-grasp] Warning: gripper joint mapping incomplete:"
                f" names={self._gripper_joint_names}"
                f" indices={self._gripper_joint_indices.tolist()}"
            )

    def _get_joint_positions_vector(self) -> Optional[np.ndarray]:
        try:
            positions = self._articulation.get_joint_positions()
        except Exception:
            return None
        if positions is None:
            return None
        arr = np.asarray(positions, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 1 or arr.size == 0:
            return None
        return arr.copy()

    def _set_gripper(self, opened: bool) -> None:
        if self._gripper_joint_indices.size == 0:
            return
        desired = self._gripper_open_targets if opened else self._gripper_closed_targets
        try:
            self._articulation.set_joint_position_targets(
                np.asarray(desired, dtype=np.float32).reshape(1, -1),
                joint_indices=self._gripper_joint_indices[: len(desired)],
            )
        except Exception as exc:
            print(f"[demo-grasp] Failed to set gripper targets: {exc}")

    def _base_to_world(self, position_base: np.ndarray) -> Optional[np.ndarray]:
        base_pose = self._ik_bridge.get_robot_base_pose()
        if base_pose is None:
            return None
        base_pos, base_quat = base_pose
        return np.asarray(base_pos, dtype=float) + _quat_rotate_vector(
            np.asarray(base_quat, dtype=float), np.asarray(position_base, dtype=float)
        )

    def _world_to_base(self, position_world: np.ndarray) -> Optional[np.ndarray]:
        base_pose = self._ik_bridge.get_robot_base_pose()
        if base_pose is None:
            return None
        base_pos, base_quat = base_pose
        return _quat_rotate_vector(
            _quat_conjugate(np.asarray(base_quat, dtype=float)),
            np.asarray(position_world, dtype=float) - np.asarray(base_pos, dtype=float),
        )

    def _get_current_ee_pose_base(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        return self._ik_bridge.get_current_ee_pose(self._arm, frame_id="base_link")

    def _get_current_ee_pose_world(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        return self._ik_bridge.get_current_ee_pose(self._arm, frame_id="world")

    def _get_block_pose_world(self) -> Optional[np.ndarray]:
        if self._block is None:
            return None
        try:
            position, _ = self._block.get_world_pose()
        except Exception:
            return None
        if position is None:
            return None
        return np.asarray(position, dtype=float)

    def _get_block_pose_base(self) -> Optional[np.ndarray]:
        block_world = self._get_block_pose_world()
        if block_world is None:
            return None
        return self._world_to_base(block_world)

    def _set_block_pose_base(self, position_base: np.ndarray) -> None:
        if self._block is None:
            return
        position_world = self._base_to_world(position_base)
        if position_world is None:
            return
        try:
            self._block.set_world_pose(
                position=np.asarray(position_world, dtype=float),
                orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            )
            if hasattr(self._block, "set_linear_velocity"):
                self._block.set_linear_velocity(np.zeros(3, dtype=float))
            if hasattr(self._block, "set_angular_velocity"):
                self._block.set_angular_velocity(np.zeros(3, dtype=float))
        except Exception as exc:
            print(f"[demo-grasp] Failed to set block pose: {exc}")

    def _update_attached_block(self) -> None:
        if not self._attached or self._block is None:
            return
        ee_pose = self._get_current_ee_pose_world()
        if ee_pose is None:
            return
        ee_pos_world, ee_quat_world = ee_pose
        block_pos_world = np.asarray(ee_pos_world, dtype=float) + _quat_rotate_vector(
            np.asarray(ee_quat_world, dtype=float), self._attach_offset_local
        )
        try:
            self._block.set_world_pose(
                position=block_pos_world,
                orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            )
            if hasattr(self._block, "set_linear_velocity"):
                self._block.set_linear_velocity(np.zeros(3, dtype=float))
            if hasattr(self._block, "set_angular_velocity"):
                self._block.set_angular_velocity(np.zeros(3, dtype=float))
        except Exception as exc:
            print(f"[demo-grasp] Failed to update attached block pose: {exc}")

    def _try_attach_block(self) -> bool:
        if not self._assist_attach or self._attached:
            return self._attached
        ee_pose = self._get_current_ee_pose_world()
        block_pos_world = self._get_block_pose_world()
        if ee_pose is None or block_pos_world is None:
            return False
        ee_pos_world, ee_quat_world = ee_pose
        delta = np.asarray(block_pos_world, dtype=float) - np.asarray(ee_pos_world, dtype=float)
        distance = float(np.linalg.norm(delta))
        if distance > self._attach_threshold:
            print(
                "[demo-grasp] Attach assist skipped:"
                f" distance={distance:.4f} threshold={self._attach_threshold:.4f}"
            )
            return False
        self._attach_offset_local = _quat_rotate_vector(
            _quat_conjugate(np.asarray(ee_quat_world, dtype=float)), delta
        )
        self._attached = True
        print(
            "[demo-grasp] Assist attach engaged:"
            f" distance={distance:.4f}"
            f" offset_local={self._attach_offset_local.tolist()}"
        )
        return True

    def _release_block_at_place(self) -> None:
        if self._attached:
            self._attached = False
            self._set_block_pose_base(self._place_base)
            print("[demo-grasp] Block released at place marker.")

    def _maybe_init_scene(self) -> bool:
        if self._scene_ready:
            return True
        base_pose = self._ik_bridge.get_robot_base_pose()
        ee_pose_base = self._get_current_ee_pose_base()
        if base_pose is None or ee_pose_base is None:
            return False
        observed_pos = np.asarray(ee_pose_base[0], dtype=float)
        if self._last_observed_ee_pos_base is None:
            self._warmup_stable_count = 0
        elif self._distance(self._last_observed_ee_pos_base, observed_pos) <= 0.002:
            self._warmup_stable_count += 1
        else:
            self._warmup_stable_count = 0
        self._last_observed_ee_pos_base = observed_pos.copy()
        if self._warmup_stable_count < 10:
            return False

        from isaacsim.core.api.objects import DynamicCuboid, VisualCuboid

        self._home_ee_pos_base = observed_pos
        self._home_ee_quat_base = _normalize_quaternion(np.asarray(ee_pose_base[1], dtype=float))
        arm_sign = 1.0 if self._arm == "left" else -1.0
        if self._block_home_base is None:
            self._block_home_base = self._home_ee_pos_base + np.array(
                [0.06, 0.0, -0.18], dtype=float
            )
        if self._place_base is None:
            self._place_base = self._home_ee_pos_base + np.array(
                [0.16, -0.12 * arm_sign, -0.18], dtype=float
            )
        self._cache_gripper_indices()
        self._setup_keyboard()

        block_world = self._base_to_world(self._block_home_base)
        place_world = self._base_to_world(self._place_base)
        if block_world is None or place_world is None:
            return False

        self._block = self._world.scene.add(
            DynamicCuboid(
                prim_path=f"/World/demo_grasp_{self._arm}_block",
                name=f"demo_grasp_{self._arm}_block",
                position=np.asarray(block_world, dtype=float),
                size=self._block_size,
                color=np.array([0.95, 0.15, 0.15], dtype=float),
                mass=0.02,
            )
        )
        self._place_marker = self._world.scene.add(
            VisualCuboid(
                prim_path=f"/World/demo_grasp_{self._arm}_place",
                name=f"demo_grasp_{self._arm}_place",
                position=np.asarray(place_world, dtype=float),
                size=self._block_size * 1.1,
                color=np.array([0.15, 0.85, 0.20], dtype=float),
            )
        )
        self._scene_ready = True
        self._set_gripper(opened=True)
        ee_pose_world = self._get_current_ee_pose_world()
        base_pose = self._ik_bridge.get_robot_base_pose() if self._ik_bridge is not None else None
        solver_base_pose = (
            self._ik_bridge._get_solver_base_pose(self._arm)  # type: ignore[attr-defined]
            if self._ik_bridge is not None
            else None
        )
        base_pose_text = (
            "None"
            if base_pose is None
            else np.array2string(np.asarray(base_pose[0], dtype=float), precision=4)
        )
        solver_base_text = (
            "None"
            if solver_base_pose is None
            else np.array2string(np.asarray(solver_base_pose[0], dtype=float), precision=4)
        )
        ee_world_text = (
            "None"
            if ee_pose_world is None
            else np.array2string(np.asarray(ee_pose_world[0], dtype=float), precision=4)
        )
        print(
            "[demo-grasp] Scene ready:"
            f" block_base={self._block_home_base.tolist()}"
            f" place_base={self._place_base.tolist()}"
            f" track_orientation={self._track_orientation}"
            f" home_ee_base={self._home_ee_pos_base.tolist()}"
            f" ee_world={ee_world_text}"
            f" base_world={base_pose_text}"
            f" solver_base_world={solver_base_text}"
        )
        return True

    def _publish_target(self, position_base: np.ndarray) -> None:
        if self._home_ee_quat_base is None:
            return
        orientation = self._home_ee_quat_base if self._track_orientation else None
        self._ik_bridge.set_target_pose(
            self._arm,
            np.asarray(position_base, dtype=float),
            orientation=orientation,
            frame_id="base_link",
        )

    @staticmethod
    def _distance(lhs: np.ndarray, rhs: np.ndarray) -> float:
        return float(np.linalg.norm(np.asarray(lhs, dtype=float) - np.asarray(rhs, dtype=float)))

    def _start_cycle(self, reason: str) -> None:
        self._pending_start = False
        self._attached = False
        block_base = self._get_block_pose_base()
        if block_base is None and self._block_home_base is not None:
            block_base = self._block_home_base.copy()
        self._cycle_block_base = (
            None if block_base is None else np.asarray(block_base, dtype=float).copy()
        )
        self._set_gripper(opened=True)
        self._transition(
            DemoGraspState.APPROACH,
            f"WAITING -> APPROACH ({reason})",
        )

    def update(self) -> None:
        if self._ik_bridge is None:
            return
        if not self._maybe_init_scene():
            return

        self._update_attached_block()

        if self._state == DemoGraspState.WAITING:
            self._set_gripper(opened=True)
            self._cycle_block_base = None
            if self._auto_start and self._cycle_count == 0:
                self._start_cycle("auto-start")
            elif self._pending_start:
                self._start_cycle(f"key={self._trigger_key}")
            return

        if self._home_ee_pos_base is None:
            return
        if self._cycle_block_base is None:
            block_base = self._get_block_pose_base()
            if block_base is None:
                block_base = self._block_home_base.copy()
            self._cycle_block_base = np.asarray(block_base, dtype=float).copy()
        block_base = self._cycle_block_base.copy()

        approach_target = block_base.copy()
        approach_target[2] += self._approach_height
        descend_target = block_base.copy()
        descend_target[2] += self._grasp_height_offset
        lift_target = block_base.copy()
        lift_target[2] += self._lift_height
        move_target = self._place_base.copy()
        move_target[2] += self._lift_height
        place_target = self._place_base.copy()
        place_target[2] += self._grasp_height_offset
        ee_pose_base = self._get_current_ee_pose_base()
        ee_pos_base = None if ee_pose_base is None else np.asarray(ee_pose_base[0], dtype=float)

        if self._state == DemoGraspState.APPROACH:
            self._set_gripper(opened=True)
            self._publish_target(approach_target)
            if ee_pos_base is not None and self._distance(
                ee_pos_base, approach_target
            ) <= self._approach_reach_threshold:
                self._transition(DemoGraspState.DESCEND, "APPROACH -> DESCEND")
            return

        if self._state == DemoGraspState.DESCEND:
            self._set_gripper(opened=True)
            self._publish_target(descend_target)
            if ee_pos_base is not None and self._distance(
                ee_pos_base, descend_target
            ) <= self._descend_reach_threshold:
                self._transition(DemoGraspState.GRASP, "DESCEND -> GRASP")
            return

        if self._state == DemoGraspState.GRASP:
            self._set_gripper(opened=False)
            self._publish_target(descend_target)
            if time.time() - self._state_started_at >= self._close_hold_sec:
                self._try_attach_block()
                self._transition(DemoGraspState.LIFT, "GRASP -> LIFT")
            return

        if self._state == DemoGraspState.LIFT:
            self._set_gripper(opened=False)
            self._publish_target(lift_target)
            if ee_pos_base is not None and self._distance(
                ee_pos_base, lift_target
            ) <= self._reach_threshold:
                self._transition(DemoGraspState.MOVE, "LIFT -> MOVE")
            return

        if self._state == DemoGraspState.MOVE:
            self._set_gripper(opened=False)
            self._publish_target(move_target)
            if ee_pos_base is not None and self._distance(
                ee_pos_base, move_target
            ) <= self._reach_threshold:
                self._transition(DemoGraspState.PLACE, "MOVE -> PLACE")
            return

        if self._state == DemoGraspState.PLACE:
            self._set_gripper(opened=False)
            self._publish_target(place_target)
            if ee_pos_base is not None and self._distance(
                ee_pos_base, place_target
            ) <= self._reach_threshold:
                self._transition(DemoGraspState.OPEN, "PLACE -> OPEN")
            return

        if self._state == DemoGraspState.OPEN:
            self._set_gripper(opened=True)
            self._publish_target(place_target)
            self._release_block_at_place()
            if time.time() - self._state_started_at >= self._open_hold_sec:
                self._transition(DemoGraspState.RETURN, "OPEN -> RETURN")
            return

        if self._state == DemoGraspState.RETURN:
            self._set_gripper(opened=True)
            self._publish_target(self._home_ee_pos_base)
            if ee_pos_base is not None and self._distance(
                ee_pos_base, self._home_ee_pos_base
            ) <= self._reach_threshold:
                self._set_block_pose_base(self._block_home_base)
                self._cycle_block_base = None
                self._cycle_count += 1
                self._transition(
                    DemoGraspState.WAITING,
                    f"RETURN -> WAITING (completed_cycles={self._cycle_count})",
                )
