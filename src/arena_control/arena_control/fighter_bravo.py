import rclpy
from rclpy.node import Node

from std_msgs.msg import String, Int32
from geometry_msgs.msg import Point, Twist


class FighterBravo(Node):

    def __init__(self):
        super().__init__('fighter_bravo')

        self.health = 100
        self.x = 2.0
        self.y = 0.0

        self.status_pub = self.create_publisher(
            String,
            '/fighter_bravo/status',
            10
        )

        self.health_pub = self.create_publisher(
            Int32,
            '/fighter_bravo/health',
            10
        )

        self.position_pub = self.create_publisher(
            Point,
            '/fighter_bravo/position',
            10
        )

        self.velocity_pub = self.create_publisher(
            Twist,
            '/fighter_bravo/cmd_vel',
            10
        )

        self.command_sub = self.create_subscription(
            String,
            '/fighter_bravo/command',
            self.command_callback,
            10
        )

        self.state_timer = self.create_timer(
            1.0,
            self.publish_state
        )

        self.stop_timer = None

        self.get_logger().info(
            'Fighter Bravo online and connected to Gazebo controls.'
        )

    def command_callback(self, msg):

        command = msg.data.strip().upper()

        velocity = Twist()

        if command == 'FORWARD':
            velocity.linear.x = 0.5

        elif command == 'BACKWARD':
            velocity.linear.x = -0.5

        elif command == 'LEFT':
            velocity.angular.z = 1.0

        elif command == 'RIGHT':
            velocity.angular.z = -1.0

        elif command == 'STOP':
            self.stop_fighter()
            return

        else:
            self.get_logger().warning(
                f'Unknown command: {command}'
            )
            return

        self.velocity_pub.publish(velocity)

        self.get_logger().info(
            f'Executing command: {command}'
        )

        if self.stop_timer is not None:
            self.stop_timer.cancel()

        self.stop_timer = self.create_timer(
            0.5,
            self.stop_after_command
        )

    def stop_after_command(self):

        self.stop_fighter()

        if self.stop_timer is not None:
            self.stop_timer.cancel()
            self.stop_timer = None

    def stop_fighter(self):

        stop = Twist()

        self.velocity_pub.publish(stop)

        self.get_logger().info('Bravo stopped.')

    def publish_state(self):

        status = String()
        status.data = 'READY'

        health = Int32()
        health.data = self.health

        position = Point()
        position.x = self.x
        position.y = self.y
        position.z = 0.0

        self.status_pub.publish(status)
        self.health_pub.publish(health)
        self.position_pub.publish(position)


def main(args=None):

    rclpy.init(args=args)

    node = FighterBravo()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        node.stop_fighter()

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
