"""Entry point for ros2 run standalone_ik_solver standalone_ik_solver."""

from __future__ import annotations

import rclpy

from standalone_ik_solver.config import SolverConfig
from standalone_ik_solver.node import StandaloneIKSolverNode


def main() -> None:
    rclpy.init()
    node = StandaloneIKSolverNode(SolverConfig())
    node.start()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.0)
            node.update()
            node._sleep()
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
