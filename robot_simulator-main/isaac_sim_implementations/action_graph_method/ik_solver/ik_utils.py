"""
IK Utility Functions

Helper functions for IK solver, including forward kinematics.
"""

import numpy as np
from typing import Optional, Tuple

# Try to import Isaac Sim rotation utils
try:
    import omni.isaac.core.utils.numpy.rotations as rotation_utils
    HAS_ISAAC_SIM = True
except ImportError:
    HAS_ISAAC_SIM = False
    rotation_utils = None


def get_end_effector_pose_from_articulation(
    articulation_view,
    ee_link_name: str,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Get end-effector pose from articulation view.

    Args:
        articulation_view: Isaac Sim ArticulationView
        ee_link_name: Name of the end-effector link

    Returns:
        Tuple of (position, quaternion) or (None, None) if not found
    """
    try:
        # Find body index by name
        body_names = articulation_view.body_names
        ee_body_index = None
        for i, name in enumerate(body_names):
            if name.lower().endswith(ee_link_name.lower()):
                ee_body_index = i
                break

        if ee_body_index is None:
            print(f"[IK Utils] EE link {ee_link_name} not found in body_names: {body_names}")
            return None, None

        # Get body poses (world frame)
        # Returns: positions [num_robots, num_bodies, 3], orientations [num_robots, num_bodies, 4]
        body_positions, body_orientations = articulation_view.get_world_poses(indices=[ee_body_index])

        if body_positions is not None and len(body_positions) > 0:
            # Assume single robot (index 0)
            pos = body_positions[0]  # [x, y, z]
            quat = body_orientations[0]  # [w, x, y, z]
            return pos, quat

    except Exception as e:
        print(f"[IK Utils] Error getting EE pose: {e}")

    return None, None


def compute_forward_kinematics_simple(
    joint_positions: np.ndarray,
    ee_link_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simplified FK computation (placeholder).

    NOTE: This function is deprecated. Use get_end_effector_pose_from_articulation()
    with ArticulationView instead for real FK from Isaac Sim.

    Args:
        joint_positions: Joint positions for the arm
        ee_link_name: End-effector link name

    Returns:
        Tuple of (position, quaternion)
    """
    # Placeholder implementation - not used in production
    print("[IK Utils] Warning: compute_forward_kinematics_simple is deprecated, use ArticulationView instead")
    pos = np.array([0.5, 0.3, 0.8])
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    return pos, quat


def extract_ee_jacobian(
    full_jacobian: np.ndarray,
    ee_link_index: int,
) -> np.ndarray:
    """
    Extract end-effector Jacobian from full robot Jacobian.

    Args:
        full_jacobian: Full Jacobian matrix [6, num_dofs]
        ee_link_index: Index of the end-effector link

    Returns:
        End-effector Jacobian [6, num_arm_dofs]
    """
    # For now, return the full Jacobian
    # TODO: Implement proper extraction based on link index
    return full_jacobian
