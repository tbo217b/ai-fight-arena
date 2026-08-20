from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node


def generate_launch_description():

    # --------------------------------------------------
    # Gazebo
    # -r = start simulation running instead of paused
    # --------------------------------------------------
    gazebo = ExecuteProcess(
        cmd=[
            'gz',
            'sim',
            '-r',
            '/home/thomas/ai_fight_arena/worlds/arena.sdf'
        ],
        output='screen'
    )

    # --------------------------------------------------
    # Low-level fighter controllers
    # --------------------------------------------------
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

    # --------------------------------------------------
    # ROS <-> Gazebo velocity bridge
    # --------------------------------------------------
    velocity_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/fighter_alpha/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist',
            '/fighter_bravo/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist'
        ],
        output='screen'
    )

    # --------------------------------------------------
    # Gazebo pose feedback -> ROS
    # --------------------------------------------------
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

    # --------------------------------------------------
    # Rule-based Bravo AI
    # --------------------------------------------------
    bravo_ai = Node(
        package='arena_control',
        executable='bravo_ai',
        name='bravo_ai',
        output='screen'
    )

    # --------------------------------------------------
    # PPO Alpha
    # Uses the separate RL virtual environment
    # --------------------------------------------------
    alpha_ppo = ExecuteProcess(
        cmd=[
            '/home/thomas/ai_fight_arena/training/venv/bin/python',
            '/home/thomas/ai_fight_arena/training/rl_fighter_alpha.py'
        ],
        output='screen'
    )

    return LaunchDescription([

        # Start Gazebo immediately.
        gazebo,

        # Give the fresh world 3 seconds to initialize.
        TimerAction(
            period=3.0,
            actions=[
                fighter_alpha,
                fighter_bravo,
                velocity_bridge,
                pose_bridge
            ]
        ),

        # Give bridges / controllers another couple seconds
        # before starting the actual AI fight.
        TimerAction(
            period=5.0,
            actions=[
                bravo_ai,
                alpha_ppo
            ]
        ),
    ])
