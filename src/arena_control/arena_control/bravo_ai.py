import math

import rclpy
from rclpy.node import Node

from std_msgs.msg import String, Int32
from tf2_msgs.msg import TFMessage


class BravoAI(Node):

    def __init__(self):
        super().__init__('bravo_ai')

        self.alpha_position = None
        self.bravo_position = None

        self.bravo_health = 100
        self.fight_over = False

        # V2.2 combat state
        self.attack_telegraphed = False

        self.command_pub = self.create_publisher(
            String,
            '/fighter_bravo/command',
            10
        )

        self.attack_pub = self.create_publisher(
            String,
            '/fighter_bravo/attack',
            10
        )

        self.telegraph_pub = self.create_publisher(
            String,
            '/fighter_bravo/telegraph',
            10
        )

        self.pose_sub = self.create_subscription(
            TFMessage,
            '/world/fight_arena/pose/info',
            self.pose_callback,
            10
        )

        self.health_sub = self.create_subscription(
            Int32,
            '/fighter_bravo/health',
            self.health_callback,
            10
        )

        self.result_sub = self.create_subscription(
            String,
            '/fight/result',
            self.result_callback,
            10
        )

        self.timer = self.create_timer(
            1.0,
            self.make_decision
        )

        self.get_logger().info(
            'Rule-Based Bravo V2.2 AI online.'
        )

    def health_callback(self, msg):

        self.bravo_health = msg.data

        if self.bravo_health <= 0:
            self.stop_after_ko()

    def result_callback(self, msg):

        result = msg.data.strip().upper()

        if result:
            self.fight_over = True
            self.clear_telegraph()
            self.publish_command('STOP')

            self.get_logger().info(
                f'Fight result received: {result}'
            )

    def stop_after_ko(self):

        if self.fight_over:
            return

        self.fight_over = True

        self.clear_telegraph()
        self.publish_command('STOP')

        self.get_logger().info(
            'Bravo has been KO\'d. AI stopped.'
        )

    def pose_callback(self, msg):

        if self.fight_over:
            return

        candidates = []

        for transform in msg.transforms:

            position = transform.transform.translation

            if 0.25 < position.z < 0.45:

                candidates.append(
                    (
                        position.x,
                        position.y,
                        position.z
                    )
                )

        if len(candidates) < 2:
            return

        unique = []

        for candidate in candidates:

            duplicate = False

            for existing in unique:

                dx = candidate[0] - existing[0]
                dy = candidate[1] - existing[1]
                dz = candidate[2] - existing[2]

                distance = math.sqrt(
                    dx * dx +
                    dy * dy +
                    dz * dz
                )

                if distance < 0.05:
                    duplicate = True
                    break

            if not duplicate:
                unique.append(candidate)

        if len(unique) < 2:
            return

        unique.sort(key=lambda p: p[0])

        self.alpha_position = unique[0]
        self.bravo_position = unique[-1]

    def get_distance(self):

        if (
            self.alpha_position is None or
            self.bravo_position is None
        ):
            return None

        dx = (
            self.bravo_position[0] -
            self.alpha_position[0]
        )

        dy = (
            self.bravo_position[1] -
            self.alpha_position[1]
        )

        return math.sqrt(
            dx * dx +
            dy * dy
        )

    def publish_telegraph(self):

        msg = String()
        msg.data = 'ATTACK'

        self.telegraph_pub.publish(msg)

        self.get_logger().info(
            'BRAVO TELEGRAPHS ATTACK!'
        )

    def clear_telegraph(self):

        if not self.attack_telegraphed:
            return

        msg = String()
        msg.data = 'CLEAR'

        self.telegraph_pub.publish(msg)

        self.attack_telegraphed = False

    def make_decision(self):

        if self.fight_over:
            return

        if self.bravo_health <= 0:
            self.stop_after_ko()
            return

        distance = self.get_distance()

        if distance is None:

            self.get_logger().info(
                'Bravo waiting for Gazebo position feedback...'
            )
            return

        self.get_logger().info(
            f'Bravo sees distance={distance:.3f} m'
        )

        # --------------------------------------
        # OUTSIDE ATTACK RANGE
        # --------------------------------------
        if distance > 1.0:

            # Cancel a telegraph if Alpha escaped.
            if self.attack_telegraphed:
                self.clear_telegraph()

                self.get_logger().info(
                    'Bravo attack cancelled: Alpha moved out of range.'
                )

            self.publish_command('FORWARD')

            self.get_logger().info(
                'Bravo decision: APPROACH'
            )

            return

        # --------------------------------------
        # CLOSE RANGE
        # --------------------------------------

        self.publish_command('STOP')

        # First close-range cycle:
        # warn Alpha about the incoming attack.
        if not self.attack_telegraphed:

            self.attack_telegraphed = True

            self.publish_telegraph()

            self.get_logger().info(
                'Bravo decision: TELEGRAPH'
            )

            return

        # Second close-range cycle:
        # execute the previously telegraphed attack.
        self.clear_telegraph()

        attack = String()
        attack.data = 'ATTACK'

        self.attack_pub.publish(attack)

        self.get_logger().info(
            'BRAVO ATTACKS!'
        )

    def publish_command(self, command):

        msg = String()
        msg.data = command

        self.command_pub.publish(msg)


def main(args=None):

    rclpy.init(args=args)

    node = BravoAI()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:

        if rclpy.ok():

            try:
                node.clear_telegraph()

                stop = String()
                stop.data = 'STOP'

                node.command_pub.publish(stop)

            except Exception:
                pass

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
