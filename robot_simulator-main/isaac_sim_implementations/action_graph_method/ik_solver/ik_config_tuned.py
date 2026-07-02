"""
Tuned IK Configuration for Stable Performance

This configuration addresses oscillation and slow convergence issues:
- Increased damping to reduce oscillation
- Reduced position gain to prevent overshoot
- Increased max_dq for faster convergence
- Relaxed tolerance to avoid continuous micro-adjustments
"""

from dataclasses import dataclass
from typing import List


@dataclass
class IKConfigTuned:
    """Tuned IK configuration for stable, fast convergence."""

    # Robot configuration
    left_arm_joints: List[str] = None
    right_arm_joints: List[str] = None

    # IK algorithm parameters - TUNED FOR STABILITY
    ik_rate_hz: float = 50.0

    # Damping: Increased from 0.03 to 0.15 to reduce oscillation
    # Higher damping = smoother motion, less overshoot
    ik_damping: float = 0.15

    # Position gain: Reduced from 1.2 to 0.8 to prevent overshoot
    # Lower gain = more conservative approach to target
    ik_pos_gain: float = 0.8

    # Orientation gain: Keep moderate
    ik_ori_gain: float = 0.6

    # Max joint change per step: Increased from 0.08 to 0.15 for faster convergence
    # Allows larger steps when far from target
    ik_max_dq: float = 0.15

    # Max joint velocity: Keep reasonable limit
    ik_max_joint_vel: float = 1.5

    # Position tolerance: Relaxed from 0.01m (1cm) to 0.02m (2cm)
    # Prevents continuous micro-adjustments that cause oscillation
    ik_pos_tol: float = 0.02

    # Orientation tolerance: Keep moderate
    ik_ori_tol: float = 0.05

    # Timeout: Increased to allow more time for convergence
    ik_timeout_sec: float = 1.0

    # Hold last target: Keep enabled
    ik_hold_last_target: bool = True

    # Orientation: Keep disabled for now
    ik_enable_orientation: bool = False

    # Performance logging
    ik_perf_log_interval_sec: float = 5.0
    ik_perf_warn_threshold_ms: float = 5.0

    # Command topic remapping
    left_cmd_topic: str = "/arm_left/joint_commands"
    right_cmd_topic: str = "/arm_right/joint_commands"

    def __post_init__(self):
        """Initialize default joint groups if not provided."""
        if self.left_arm_joints is None:
            self.left_arm_joints = [
                "left_arm_joint1",
                "left_arm_joint2",
                "left_arm_joint3",
                "left_arm_joint4",
                "left_arm_joint5",
                "left_arm_joint6",
                "left_arm_joint7",
            ]

        if self.right_arm_joints is None:
            self.right_arm_joints = [
                "right_arm_joint1",
                "right_arm_joint2",
                "right_arm_joint3",
                "right_arm_joint4",
                "right_arm_joint5",
                "right_arm_joint6",
                "right_arm_joint7",
            ]

    @property
    def ik_dt(self) -> float:
        """Time step for IK updates."""
        return 1.0 / self.ik_rate_hz


# Comparison table for reference:
"""
Parameter Comparison:
┌─────────────────────┬──────────┬──────────┬────────────────────────────┐
│ Parameter           │ Original │ Tuned    │ Effect                     │
├─────────────────────┼──────────┼──────────┼────────────────────────────┤
│ ik_damping          │ 0.03     │ 0.15     │ 5x more damping → less     │
│                     │          │          │ oscillation                │
├─────────────────────┼──────────┼──────────┼────────────────────────────┤
│ ik_pos_gain         │ 1.2      │ 0.8      │ 33% less aggressive →      │
│                     │          │          │ less overshoot             │
├─────────────────────┼──────────┼──────────┼────────────────────────────┤
│ ik_max_dq           │ 0.08     │ 0.15     │ 87% larger steps →         │
│                     │          │          │ faster convergence         │
├─────────────────────┼──────────┼──────────┼────────────────────────────┤
│ ik_pos_tol          │ 0.01m    │ 0.02m    │ 2x more tolerant →         │
│                     │          │          │ stops adjusting sooner     │
├─────────────────────┼──────────┼──────────┼────────────────────────────┤
│ ik_timeout_sec      │ 0.5      │ 1.0      │ More time to converge      │
└─────────────────────┴──────────┴──────────┴────────────────────────────┘

Expected behavior with tuned parameters:
✓ Faster approach to target (larger max_dq)
✓ Smoother motion (higher damping)
✓ Less overshoot (lower gain)
✓ Stops oscillating near target (relaxed tolerance)
"""
