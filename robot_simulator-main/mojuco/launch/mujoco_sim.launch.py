#!/usr/bin/env python3
"""
Launch file for MuJoCo Simulator Node

Usage:
    ros2 launch mujoco_simulator mujoco_sim.launch.py
    ros2 launch mujoco_simulator mujoco_sim.launch.py enable_viewer:=false
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Get package directories
    pkg_share = get_package_share_directory('mujoco_simulator')

    # Declare launch arguments
    enable_viewer_arg = DeclareLaunchArgument(
        'enable_viewer',
        default_value='true',
        description='Enable MuJoCo viewer window'
    )

    sim_rate_arg = DeclareLaunchArgument(
        'sim_rate',
        default_value='500.0',
        description='Simulation step rate in Hz'
    )

    pub_rate_arg = DeclareLaunchArgument(
        'pub_rate',
        default_value='50.0',
        description='State publishing rate in Hz'
    )

    viewer_rate_arg = DeclareLaunchArgument(
        'viewer_rate',
        default_value='60.0',
        description='MuJoCo viewer sync rate in Hz'
    )

    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='/mnt/e/jz_robot/jz_descripetion/robot_urdf/urdf/robot_model.mjcf.xml',
        description='Path to MuJoCo model XML file'
    )

    base_body_name_arg = DeclareLaunchArgument(
        'base_body_name',
        default_value='link0',
        description='Body name corresponding to ROS base_link frame for target marker transform'
    )

    keyframe_name_arg = DeclareLaunchArgument(
        'keyframe_name',
        default_value='teleop_home',
        description='Name of keyframe to apply at startup'
    )

    keyframe_zero_velocity_arg = DeclareLaunchArgument(
        'keyframe_zero_velocity',
        default_value='true',
        description='Zero keyframe joint velocities at startup to prevent jitter'
    )

    gravity_compensation_arg = DeclareLaunchArgument(
        'gravity_compensation',
        default_value='true',
        description='Enable gravity compensation'
    )

    gravity_compensation_scale_arg = DeclareLaunchArgument(
        'gravity_compensation_scale',
        default_value='1.0',
        description='Gravity compensation scale'
    )

    gravity_compensation_body_only_arg = DeclareLaunchArgument(
        'gravity_compensation_body_only',
        default_value='true',
        description='Apply gravity compensation only to body joints'
    )

    gravity_compensation_body_scale_arg = DeclareLaunchArgument(
        'gravity_compensation_body_scale',
        default_value='1.0',
        description='Extra gravity compensation scale for body joints'
    )

    gravity_compensation_body1_scale_arg = DeclareLaunchArgument(
        'gravity_compensation_body1_scale',
        default_value='1.2',
        description='Extra gravity compensation scale for body1'
    )

    gravity_compensation_body2_scale_arg = DeclareLaunchArgument(
        'gravity_compensation_body2_scale',
        default_value='1.0',
        description='Extra gravity compensation scale for body2'
    )

    gravity_compensation_body3_scale_arg = DeclareLaunchArgument(
        'gravity_compensation_body3_scale',
        default_value='1.0',
        description='Extra gravity compensation scale for body3'
    )

    gravity_compensation_filter_alpha_arg = DeclareLaunchArgument(
        'gravity_compensation_filter_alpha',
        default_value='1.0',
        description='Low-pass filter alpha for gravity compensation (0..1)'
    )

    gravity_compensation_velocity_threshold_arg = DeclareLaunchArgument(
        'gravity_compensation_velocity_threshold',
        default_value='0.0',
        description='Velocity threshold for gravity compensation attenuation (rad/s)'
    )

    gravity_compensation_start_delay_arg = DeclareLaunchArgument(
        'gravity_compensation_start_delay',
        default_value='0.5',
        description='Delay before gravity compensation ramp (s)'
    )

    gravity_compensation_ramp_time_arg = DeclareLaunchArgument(
        'gravity_compensation_ramp_time',
        default_value='2.0',
        description='Ramp time for gravity compensation (s)'
    )

    body_position_deadband_arg = DeclareLaunchArgument(
        'body_position_deadband',
        default_value='0.002',
        description='Body joint deadband in radians to reduce small jitter'
    )

    enable_contact_arg = DeclareLaunchArgument(
        'enable_contact',
        default_value='true',
        description='Enable contact simulation in MuJoCo (set false to avoid startup self-collision drift)'
    )

    # MuJoCo simulator node
    mujoco_sim_node = Node(
        package='mujoco_simulator',
        executable='mujoco_sim_node',
        name='mujoco_sim_node',
        output='screen',
        parameters=[{
            'enable_viewer': LaunchConfiguration('enable_viewer'),
            'sim_rate': LaunchConfiguration('sim_rate'),
            'pub_rate': LaunchConfiguration('pub_rate'),
            'viewer_rate': LaunchConfiguration('viewer_rate'),
            'model_path': LaunchConfiguration('model_path'),
            'base_body_name': LaunchConfiguration('base_body_name'),
            'keyframe_name': LaunchConfiguration('keyframe_name'),
            'keyframe_zero_velocity': LaunchConfiguration('keyframe_zero_velocity'),
            'gravity_compensation': LaunchConfiguration('gravity_compensation'),
            'gravity_compensation_scale': LaunchConfiguration('gravity_compensation_scale'),
            'gravity_compensation_body_only': LaunchConfiguration('gravity_compensation_body_only'),
            'gravity_compensation_body_scale': LaunchConfiguration('gravity_compensation_body_scale'),
            'gravity_compensation_body1_scale': LaunchConfiguration('gravity_compensation_body1_scale'),
            'gravity_compensation_body2_scale': LaunchConfiguration('gravity_compensation_body2_scale'),
            'gravity_compensation_body3_scale': LaunchConfiguration('gravity_compensation_body3_scale'),
            'gravity_compensation_filter_alpha': LaunchConfiguration('gravity_compensation_filter_alpha'),
            'gravity_compensation_velocity_threshold': LaunchConfiguration('gravity_compensation_velocity_threshold'),
            'gravity_compensation_start_delay': LaunchConfiguration('gravity_compensation_start_delay'),
            'gravity_compensation_ramp_time': LaunchConfiguration('gravity_compensation_ramp_time'),
            'body_position_deadband': LaunchConfiguration('body_position_deadband'),
            'enable_contact': LaunchConfiguration('enable_contact'),
            'position_control_kp': 100.0,
            'position_control_kd': 10.0,
            'body3_kp': 500.0,
            'body3_ki': 20.0,
            'body3_kd': 50.0,
        }],
    )

    return LaunchDescription([
        enable_viewer_arg,
        sim_rate_arg,
        pub_rate_arg,
        viewer_rate_arg,
        model_path_arg,
        base_body_name_arg,
        keyframe_name_arg,
        keyframe_zero_velocity_arg,
        gravity_compensation_arg,
        gravity_compensation_scale_arg,
        gravity_compensation_body_only_arg,
        gravity_compensation_body_scale_arg,
        gravity_compensation_body1_scale_arg,
        gravity_compensation_body2_scale_arg,
        gravity_compensation_body3_scale_arg,
        gravity_compensation_filter_alpha_arg,
        gravity_compensation_velocity_threshold_arg,
        gravity_compensation_start_delay_arg,
        gravity_compensation_ramp_time_arg,
        body_position_deadband_arg,
        enable_contact_arg,
        mujoco_sim_node,
    ])
