import gymnasium as gym
from gymnasium import spaces
import numpy as np


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


class FightEnvV23(gym.Env):
    """
    PPO V2.3 environment.

    Designed to better match the Isaac Sim execution semantics:
      * Bravo telegraphs one decision before attacking.
      * The following observation identifies Bravo's committed ATTACK.
      * Alpha guard / dodge state persists across decisions.
      * Block costs 10 stamina.
      * Attack costs 20 stamina.
      * Bravo attack does 15 damage.
      * Blocking reduces Bravo damage to 45%.
      * Alpha attacks do 20 damage.
      * Finishing low-HP Bravo is strongly rewarded.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()

        self.action_space = spaces.Discrete(6)

        self.observation_space = spaces.Box(
            low=np.array(
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                dtype=np.float32,
            ),
            high=np.array(
                [100, 100, 10, 100, 100, 3, 3, 1, 5, 5, 1],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        self.max_steps = 180
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.alpha_health = 100.0
        self.bravo_health = 100.0

        self.alpha_stamina = 100.0
        self.bravo_stamina = 100.0

        # Center training around the Isaac deployment start distance,
        # but retain some variation for robustness.
        self.distance = float(
            self.np_random.uniform(3.5, 4.5)
        )

        self.alpha_cooldown = 0
        self.bravo_cooldown = 0

        self.alpha_block_steps = 0
        self.alpha_dodge_steps = 0

        self.bravo_attack_pending = False
        self.bravo_attack_telegraph = 0

        self.previous_alpha_action = WAIT
        self.previous_bravo_action = APPROACH

        self.bravo_aggression = float(
            self.np_random.uniform(0.60, 0.85)
        )

        self.steps = 0

        # Prepare the state Alpha sees for its first decision.
        self._prepare_next_bravo_state()

        return self._get_observation(), {}

    def _get_observation(self):
        return np.array(
            [
                self.alpha_health,
                self.bravo_health,
                self.distance,
                self.alpha_stamina,
                self.bravo_stamina,
                float(self.alpha_cooldown),
                float(self.bravo_cooldown),
                float(self.bravo_attack_telegraph),
                float(self.previous_alpha_action),
                float(self.previous_bravo_action),
                self.bravo_aggression,
            ],
            dtype=np.float32,
        )

    def _prepare_next_bravo_state(self):
        """
        Create the Bravo state Alpha will see on the NEXT decision.

        Close range sequence:
            TEL=1, Bravo=WAIT
                   ↓
            TEL=0, Bravo=ATTACK
                   ↓
            new telegraph
        """

        if self.distance > 1.0:
            self.bravo_attack_telegraph = 0
            self.bravo_attack_pending = False
            self.previous_bravo_action = APPROACH
            return

        if not self.bravo_attack_pending:
            self.bravo_attack_telegraph = 1
            self.bravo_attack_pending = True
            self.previous_bravo_action = WAIT
        else:
            self.bravo_attack_telegraph = 0
            self.bravo_attack_pending = False
            self.previous_bravo_action = ATTACK

    def step(self, action):
        action = int(action)
        self.steps += 1

        reward = -0.02

        # Defensive state inherited from the prior decision.
        alpha_blocking = self.alpha_block_steps > 0
        alpha_dodging = self.alpha_dodge_steps > 0

        self.alpha_block_steps = max(
            0, self.alpha_block_steps - 1
        )
        self.alpha_dodge_steps = max(
            0, self.alpha_dodge_steps - 1
        )

        self.alpha_cooldown = max(
            0, self.alpha_cooldown - 1
        )
        self.bravo_cooldown = max(
            0, self.bravo_cooldown - 1
        )

        self.alpha_stamina = min(
            100.0, self.alpha_stamina + 5.0
        )
        self.bravo_stamina = min(
            100.0, self.bravo_stamina + 5.0
        )

        current_bravo_action = self.previous_bravo_action
        current_telegraph = self.bravo_attack_telegraph

        # ==================================================
        # ALPHA ACTION
        # ==================================================

        if action == WAIT:
            reward -= 0.04

            if (
                self.bravo_health <= 20
                and self.distance <= 1.0
                and self.alpha_stamina >= 20
                and self.alpha_cooldown == 0
            ):
                reward -= 0.80

        elif action == APPROACH:
            if self.distance > 0.8:
                self.distance = max(
                    0.0, self.distance - 0.35
                )
                reward += 0.08
            else:
                reward -= 0.15

            # Isaac V2.2 failure mode: approaching a nearly-dead
            # opponent instead of finishing it.
            if (
                self.bravo_health <= 20
                and self.distance <= 1.0
                and self.alpha_stamina >= 20
                and self.alpha_cooldown == 0
            ):
                reward -= 1.00

        elif action == RETREAT:
            self.distance = min(
                10.0, self.distance + 0.25
            )

            if (
                self.alpha_health < 40
                or self.alpha_stamina < 20
                or current_bravo_action == ATTACK
            ):
                reward += 0.15
            else:
                reward -= 0.10

        elif action == ATTACK:
            reward += self._alpha_attack()

        elif action == BLOCK:
            if self.alpha_stamina >= 10:
                self.alpha_stamina -= 10
                self.alpha_block_steps = 2

                # Blocking the ACTUAL committed attack is useful.
                if current_bravo_action == ATTACK:
                    reward += 0.70

                # Blocking only the telegraph wastes stamina.
                elif current_telegraph == 1:
                    reward -= 0.18

                else:
                    reward -= 0.15
            else:
                reward -= 0.35

        elif action == DODGE:
            if self.alpha_stamina >= 15:
                self.alpha_stamina -= 15
                self.alpha_dodge_steps = 2

                self.distance = min(
                    10.0, self.distance + 0.45
                )

                if current_bravo_action == ATTACK:
                    reward += 0.90
                elif current_telegraph == 1:
                    reward += 0.05
                else:
                    reward -= 0.15
            else:
                reward -= 0.35

        # Re-check after Alpha's current action.
        alpha_blocking = self.alpha_block_steps > 0
        alpha_dodging = self.alpha_dodge_steps > 0

        # ==================================================
        # BRAVO ACTION
        # ==================================================

        if current_bravo_action == APPROACH:
            self.distance = max(
                0.0, self.distance - 0.30
            )

        elif current_bravo_action == ATTACK:
            if (
                self.distance <= 1.0
                and self.bravo_cooldown == 0
                and self.bravo_stamina >= 20
            ):
                self.bravo_stamina -= 20
                self.bravo_cooldown = 2

                if alpha_dodging:
                    reward += 1.50

                else:
                    damage = 15.0

                    if alpha_blocking:
                        damage *= 0.45
                        reward += 0.90

                    self.alpha_health = max(
                        0.0,
                        self.alpha_health - damage,
                    )

                    reward -= damage * 0.12

        # ==================================================
        # END-OF-STEP
        # ==================================================

        self.previous_alpha_action = action

        terminated = False
        truncated = False

        if self.bravo_health <= 0:
            terminated = True
            reward += 35.0

            # Reward efficient victories.
            reward += self.alpha_health * 0.03

        elif self.alpha_health <= 0:
            terminated = True
            reward -= 35.0

        if self.steps >= self.max_steps:
            truncated = True
            reward -= 5.0

        if not terminated and not truncated:
            self._prepare_next_bravo_state()

        observation = self._get_observation()

        info = {
            "alpha_health": self.alpha_health,
            "bravo_health": self.bravo_health,
            "distance": self.distance,
            "alpha_stamina": self.alpha_stamina,
            "bravo_stamina": self.bravo_stamina,
            "alpha_action": ACTION_NAMES[action],
            "bravo_action": ACTION_NAMES[current_bravo_action],
            "bravo_telegraph": current_telegraph,
            "bravo_aggression": self.bravo_aggression,
        }

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )

    def _alpha_attack(self):
        if self.alpha_cooldown > 0:
            return -0.25

        if self.alpha_stamina < 20:
            return -0.35

        if self.distance > 1.0:
            return -0.40

        self.alpha_stamina -= 20
        self.alpha_cooldown = 2

        damage = 20.0

        before = self.bravo_health

        self.bravo_health = max(
            0.0,
            self.bravo_health - damage,
        )

        reward = damage * 0.14

        # Strong finishing incentive.
        if before <= 20:
            reward += 4.0

        elif before <= 40:
            reward += 1.0

        return reward

    def render(self):
        print(
            f"AHP={self.alpha_health:.1f} "
            f"ASTA={self.alpha_stamina:.0f} | "
            f"BHP={self.bravo_health:.1f} "
            f"BSTA={self.bravo_stamina:.0f} | "
            f"D={self.distance:.2f} | "
            f"TEL={self.bravo_attack_telegraph} | "
            f"BRAVO={ACTION_NAMES[self.previous_bravo_action]}"
        )
