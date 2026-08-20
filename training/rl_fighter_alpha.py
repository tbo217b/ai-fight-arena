import math
import os
import time

import numpy as np
import rclpy

from rclpy.node import Node
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage
from stable_baselines3 import PPO


WAIT = 0
APPROACH = 1
RETREAT = 2
ATTACK = 3
BLOCK = 4
DODGE = 5


ACTION_NAMES = {
    WAIT: "WAIT",
    APPROACH: "APPROACH",
    RETREAT: "RETREAT",
    ATTACK: "ATTACK",
    BLOCK: "BLOCK",
    DODGE: "DODGE",
}


class RLFighterAlpha(Node):

    def __init__(self):
        super().__init__("rl_fighter_alpha")

        # ==========================================
        # LOAD PPO V2.2
        # ==========================================

        model_path = os.path.join(
            os.path.dirname(__file__),
            "alpha_policy_v22.zip"
        )

        self.get_logger().info(
            f"Loading PPO V2.2 policy from {model_path}"
        )

        self.model = PPO.load(model_path)

        # ==========================================
        # COMBAT STATE
        # ==========================================

        self.alpha_health = 100.0
        self.bravo_health = 100.0

        self.alpha_stamina = 100.0
        self.bravo_stamina = 100.0

        self.alpha_cooldown = 0
        self.bravo_cooldown = 0

        self.alpha_position = None
        self.bravo_position = None

        self.fight_over = False

        # ==========================================
        # V2.2 OBSERVATION STATE
        # ==========================================

        self.bravo_attack_telegraph = 0

        self.previous_alpha_action = WAIT
        self.previous_bravo_action = WAIT

        # Training randomized this between 0.35 and 0.85.
        # The live Bravo is fairly aggressive, so 0.70 is
        # a reasonable live representation.
        self.bravo_aggression = 0.70

        # ==========================================
        # TIMED DEFENSIVE STATES
        # ==========================================

        self.block_until = 0.0
        self.dodge_until = 0.0

        self.block_duration = 1.20
        self.dodge_duration = 1.20

        # ==========================================
        # ROS PUBLISHERS
        # ==========================================

        self.command_pub = self.create_publisher(
            String,
            "/fighter_alpha/command",
            10
        )

        self.result_pub = self.create_publisher(
            String,
            "/fight/result",
            10
        )

        # ==========================================
        # ROS SUBSCRIBERS
        # ==========================================

        self.pose_sub = self.create_subscription(
            TFMessage,
            "/world/fight_arena/pose/info",
            self.pose_callback,
            10
        )

        self.bravo_attack_sub = self.create_subscription(
            String,
            "/fighter_bravo/attack",
            self.bravo_attack_callback,
            10
        )

        self.bravo_telegraph_sub = self.create_subscription(
            String,
            "/fighter_bravo/telegraph",
            self.bravo_telegraph_callback,
            10
        )

        # ==========================================
        # PPO DECISION LOOP
        # ==========================================

        self.decision_timer = self.create_timer(
            1.0,
            self.make_decision
        )

        self.get_logger().info(
            "Closed-loop PPO V2.2 Fighter Alpha online."
        )

        self.get_logger().info(
            "Waiting for real Gazebo fighter positions..."
        )

    # ==============================================
    # DEFENSIVE STATE
    # ==============================================

    def alpha_is_blocking(self):
        return time.monotonic() < self.block_until

    def alpha_is_dodging(self):
        return time.monotonic() < self.dodge_until

    # ==============================================
    # GAZEBO POSE
    # ==============================================

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

    def get_real_distance(self):
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

    # ==============================================
    # BRAVO TELEGRAPH
    # ==============================================

    def bravo_telegraph_callback(self, msg):
        if self.fight_over:
            return

        state = msg.data.strip().upper()

        if state in ("ATTACK", "ATTACKING", "1", "ON"):
            self.bravo_attack_telegraph = 1
            self.previous_bravo_action = WAIT

            self.get_logger().info(
                "BRAVO TELEGRAPHS AN INCOMING ATTACK!"
            )

        elif state in ("CLEAR", "0", "OFF"):
            self.bravo_attack_telegraph = 0

    # ==============================================
    # BRAVO ATTACK
    # ==============================================

    def bravo_attack_callback(self, msg):
        if self.fight_over:
            return

        if msg.data.strip().upper() != "ATTACK":
            return

        self.previous_bravo_action = ATTACK

        # Attack has now been committed.
        self.bravo_attack_telegraph = 0

        distance = self.get_real_distance()

        if distance is None or distance > 1.0:
            return

        if self.alpha_health <= 0:
            return

        if self.bravo_cooldown > 0:
            return

        if self.bravo_stamina < 20:
            return

        self.bravo_stamina -= 20
        self.bravo_cooldown = 2

        if self.alpha_is_dodging():
            self.get_logger().info(
                "ALPHA DODGED BRAVO'S ATTACK!"
            )

            self.get_logger().info(
                f"Alpha health remains "
                f"{self.alpha_health:.1f}"
            )

            return

        damage = 15.0

        if self.alpha_is_blocking():
            damage *= 0.45

            self.get_logger().info(
                "ALPHA BLOCKS PART OF THE ATTACK!"
            )

        self.alpha_health = max(
            0.0,
            self.alpha_health - damage
        )

        self.get_logger().info(
            f"BRAVO hits ALPHA for "
            f"{damage:.1f} damage"
        )

        self.get_logger().info(
            f"Alpha health: "
            f"{self.alpha_health:.1f}"
        )

        if self.alpha_health <= 0:
            self.finish_fight(
                "BRAVO_WINS",
                "KO - BRAVO WINS!"
            )

    # ==============================================
    # PPO V2.2
    # ==============================================

    def make_decision(self):
        if self.fight_over:
            return

        distance = self.get_real_distance()

        if distance is None:
            self.get_logger().info(
                "Still waiting for Gazebo pose feedback..."
            )
            return

        # Match training cooldown progression.
        self.alpha_cooldown = max(
            0,
            self.alpha_cooldown - 1
        )

        self.bravo_cooldown = max(
            0,
            self.bravo_cooldown - 1
        )

        # Match training stamina regeneration.
        self.alpha_stamina = min(
            100.0,
            self.alpha_stamina + 5.0
        )

        self.bravo_stamina = min(
            100.0,
            self.bravo_stamina + 5.0
        )

        # ==========================================
        # EXACT 11-VALUE V2.2 OBSERVATION
        # ==========================================

        observation = np.array(
            [
                self.alpha_health,
                self.bravo_health,
                distance,
                self.alpha_stamina,
                self.bravo_stamina,
                float(self.alpha_cooldown),
                float(self.bravo_cooldown),
                float(self.bravo_attack_telegraph),
                float(self.previous_alpha_action),
                float(self.previous_bravo_action),
                float(self.bravo_aggression),
            ],
            dtype=np.float32
        )

        action, _ = self.model.predict(
            observation,
            deterministic=True
        )

        action = int(action)

        self.get_logger().info(
            f"REAL distance={distance:.3f} m | "
            f"Alpha HP={self.alpha_health:.1f} "
            f"STA={self.alpha_stamina:.0f} | "
            f"Bravo HP={self.bravo_health:.1f} "
            f"STA={self.bravo_stamina:.0f} | "
            f"TEL={self.bravo_attack_telegraph} | "
            f"PPO={ACTION_NAMES[action]}"
        )

        self.previous_alpha_action = action

        if action == WAIT:
            self.publish_command("STOP")

        elif action == APPROACH:
            self.publish_command("FORWARD")

        elif action == RETREAT:
            self.publish_command("BACKWARD")

        elif action == BLOCK:
            self.perform_block()

        elif action == DODGE:
            self.perform_dodge()

        elif action == ATTACK:
            self.perform_alpha_attack(
                distance
            )

    # ==============================================
    # BLOCK
    # ==============================================

    def perform_block(self):
        if self.alpha_stamina < 10:
            self.publish_command("STOP")

            self.get_logger().info(
                "Alpha does not have enough "
                "stamina to block."
            )

            return

        self.alpha_stamina -= 10

        self.block_until = (
            time.monotonic() +
            self.block_duration
        )

        self.publish_command("STOP")

        self.get_logger().info(
            f"ALPHA BLOCKS! Guard active for "
            f"{self.block_duration:.1f}s"
        )

    # ==============================================
    # DODGE
    # ==============================================

    def perform_dodge(self):
        if self.alpha_stamina < 15:
            self.publish_command("STOP")

            self.get_logger().info(
                "Alpha does not have enough "
                "stamina to dodge."
            )

            return

        self.alpha_stamina -= 15

        self.dodge_until = (
            time.monotonic() +
            self.dodge_duration
        )

        self.publish_command("BACKWARD")

        self.get_logger().info(
            f"ALPHA DODGES! Evasion active for "
            f"{self.dodge_duration:.1f}s"
        )

    # ==============================================
    # ALPHA ATTACK
    # ==============================================

    def perform_alpha_attack(self, distance):
        if self.alpha_cooldown > 0:
            self.publish_command("STOP")

            self.get_logger().info(
                "Alpha attack still on cooldown."
            )

            return

        if self.alpha_stamina < 20:
            self.publish_command("STOP")

            self.get_logger().info(
                "Alpha does not have enough "
                "stamina to attack."
            )

            return

        if distance > 1.0:
            self.publish_command("FORWARD")

            self.get_logger().info(
                "PPO selected ATTACK out of range; "
                "continuing physical approach."
            )

            return

        self.alpha_stamina -= 20
        self.alpha_cooldown = 2

        self.publish_command("STOP")

        self.get_logger().info(
            "RL ALPHA ATTACKS!"
        )

        damage = 20.0

        self.bravo_health = max(
            0.0,
            self.bravo_health - damage
        )

        self.get_logger().info(
            f"Bravo health: "
            f"{self.bravo_health:.1f}"
        )

        if self.bravo_health <= 0:
            self.finish_fight(
                "ALPHA_WINS",
                "KO - PPO V2.2 ALPHA WINS!"
            )

    # ==============================================
    # FIGHT RESULT
    # ==============================================

    def finish_fight(self, result_text, log_text):
        if self.fight_over:
            return

        self.fight_over = True

        self.publish_command("STOP")

        self.get_logger().info(log_text)

        result = String()
        result.data = result_text

        self.result_pub.publish(result)

    # ==============================================
    # MOVEMENT COMMAND
    # ==============================================

    def publish_command(self, command):
        msg = String()
        msg.data = command

        self.command_pub.publish(msg)

        self.get_logger().info(
            f"Published command: {command}"
        )


def main():
    rclpy.init()

    node = RLFighterAlpha()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        if rclpy.ok():
            try:
                stop = String()
                stop.data = "STOP"

                node.command_pub.publish(stop)

                time.sleep(0.05)

            except Exception:
                pass

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
