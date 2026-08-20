import os

from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node


def generate_launch_description():

    workspace_root = os.path.expanduser('~/ai_fight_arena')

    world_path = os.path.join(
        workspace_root,
        'worlds',
        'arena.sdf'
    )

    python_path = os.path.join(
        workspace_root,
        'training',
        'venv',
        'bin',
        'python'
    )

    alpha_script = os.path.join(
        workspace_root,
        'training',
        'rl_fighter_alpha.py'
    )

    gazebo = ExecuteProcess(
        cmd=[
            'gz',
            'sim',
            '-r',
            world_path
        ],
        output='screen'
    )

    fighter_alpha = Node(
        package='arena_control',
        executable='fighter_alpha',
        name='fighter_alpha',
        output='screen'
    )

    fighter_bravo = Node(
        package='arena_control',
        executable='fighter_bravo',
        name='fighter_bravo',
        output='screen'
    )

    velocity_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/fighter_alpha/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist',
            '/fighter_bravo/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist'
        ],
        output='screen'
    )

    pose_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/world/fight_arena/pose/info'
            '@tf2_msgs/msg/TFMessage'
            '[gz.msgs.Pose_V'
        ],
        output='screen'
    )

    bravo_ai = Node(
        package='arena_control',
        executable='bravo_ai',
        name='bravo_ai',
        output='screen'
    )

    alpha_ppo = ExecuteProcess(
        cmd=[
            python_path,
            alpha_script
        ],
        output='screen'
    )

    return LaunchDescription([
        gazebo,

        TimerAction(
            period=3.0,
            actions=[
                fighter_alpha,
                fighter_bravo,
                velocity_bridge,
                pose_bridge
            ]
        ),

        TimerAction(
            period=5.0,
            actions=[
                bravo_ai,
                alpha_ppo
            ]
        ),
    ])
