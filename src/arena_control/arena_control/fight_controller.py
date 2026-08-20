import rclpy
from rclpy.node import Node

from std_msgs.msg import String, Int32


class FightController(Node):

    def __init__(self):
        super().__init__('fight_controller')

        self.alpha_health = 100
        self.bravo_health = 100

        self.alpha_command_pub = self.create_publisher(
            String,
            '/fighter_alpha/command',
            10
        )

        self.bravo_command_pub = self.create_publisher(
            String,
            '/fighter_bravo/command',
            10
        )

        self.alpha_health_pub = self.create_publisher(
            Int32,
            '/fight/alpha_health',
            10
        )

        self.bravo_health_pub = self.create_publisher(
            Int32,
            '/fight/bravo_health',
            10
        )

        self.event_pub = self.create_publisher(
            String,
            '/fight/event',
            10
        )

        self.winner_pub = self.create_publisher(
            String,
            '/fight/winner',
            10
        )

        self.state = 'APPROACH'
        self.tick = 0
        self.finished = False

        self.timer = self.create_timer(
            1.0,
            self.update_fight
        )

        self.publish_event('FIGHT START')

        self.get_logger().info(
            'Fight Controller online. Fight started.'
        )

    def publish_event(self, text):

        msg = String()
        msg.data = text

        self.event_pub.publish(msg)
        self.get_logger().info(text)

    def send_command(self, publisher, command):

        msg = String()
        msg.data = command

        publisher.publish(msg)

    def publish_health(self):

        alpha = Int32()
        alpha.data = self.alpha_health

        bravo = Int32()
        bravo.data = self.bravo_health

        self.alpha_health_pub.publish(alpha)
        self.bravo_health_pub.publish(bravo)

    def update_fight(self):

        if self.finished:
            return

        self.tick += 1

        if self.state == 'APPROACH':

            self.send_command(
                self.alpha_command_pub,
                'FORWARD'
            )

            self.send_command(
                self.bravo_command_pub,
                'FORWARD'
            )

            self.publish_event(
                f'Fighters approaching - step {self.tick}'
            )

            if self.tick >= 4:

                self.state = 'COMBAT'

                self.send_command(
                    self.alpha_command_pub,
                    'STOP'
                )

                self.send_command(
                    self.bravo_command_pub,
                    'STOP'
                )

                self.publish_event(
                    'Fighters are in attack range'
                )

        elif self.state == 'COMBAT':

            # Alternate attacks so the fight is easy to observe.
            if self.tick % 2 == 0:

                damage = 20

                self.bravo_health -= damage

                self.publish_event(
                    f'ALPHA hits BRAVO for {damage} damage'
                )

            else:

                damage = 15

                self.alpha_health -= damage

                self.publish_event(
                    f'BRAVO hits ALPHA for {damage} damage'
                )

            self.publish_health()

            self.get_logger().info(
                f'Health | Alpha: {self.alpha_health} | Bravo: {self.bravo_health}'
            )

            if self.alpha_health <= 0:

                self.finish_fight('BRAVO')

            elif self.bravo_health <= 0:

                self.finish_fight('ALPHA')

    def finish_fight(self, winner):

        self.finished = True

        self.send_command(
            self.alpha_command_pub,
            'STOP'
        )

        self.send_command(
            self.bravo_command_pub,
            'STOP'
        )

        winner_msg = String()
        winner_msg.data = winner

        self.winner_pub.publish(winner_msg)

        self.publish_event(
            f'KO - {winner} WINS'
        )

        self.get_logger().info(
            f'Winner: {winner}'
        )


def main(args=None):

    rclpy.init(args=args)

    node = FightController()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
