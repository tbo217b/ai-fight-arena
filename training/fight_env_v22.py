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


class FightEnvV22(gym.Env):

    metadata = {
        "render_modes": ["human"]
    }

    def __init__(self):
        super().__init__()

        self.action_space = spaces.Discrete(6)

        # Observation:
        # 0 Alpha HP
        # 1 Bravo HP
        # 2 Distance
        # 3 Alpha stamina
        # 4 Bravo stamina
        # 5 Alpha cooldown
        # 6 Bravo cooldown
        # 7 Bravo attack telegraph
        # 8 Previous Alpha action
        # 9 Previous Bravo action
        # 10 Bravo aggression
        self.observation_space = spaces.Box(
            low=np.array(
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                dtype=np.float32
            ),
            high=np.array(
                [100, 100, 10, 100, 100, 3, 3, 1, 5, 5, 1],
                dtype=np.float32
            ),
            dtype=np.float32
        )

        self.max_steps = 180

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.alpha_health = 100.0
        self.bravo_health = 100.0

        self.alpha_stamina = 100.0
        self.bravo_stamina = 100.0

        self.distance = float(
            self.np_random.uniform(3.0, 5.0)
        )

        self.alpha_cooldown = 0
        self.bravo_cooldown = 0

        self.alpha_blocking = False
        self.alpha_dodging = False
        self.bravo_blocking = False

        self.bravo_attack_telegraph = 0

        self.previous_alpha_action = WAIT
        self.previous_bravo_action = WAIT

        # Random opponent personality each episode.
        self.bravo_aggression = float(
            self.np_random.uniform(0.35, 0.85)
        )

        self.steps = 0

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
            dtype=np.float32
        )

    def step(self, action):
        action = int(action)

        self.steps += 1

        reward = -0.02

        self.alpha_blocking = False
        self.alpha_dodging = False
        self.bravo_blocking = False

        self.alpha_cooldown = max(
            0,
            self.alpha_cooldown - 1
        )

        self.bravo_cooldown = max(
            0,
            self.bravo_cooldown - 1
        )

        self.alpha_stamina = min(
            100.0,
            self.alpha_stamina + 5.0
        )

        self.bravo_stamina = min(
            100.0,
            self.bravo_stamina + 5.0
        )

        # =================================
        # ALPHA ACTION
        # =================================

        if action == WAIT:
            reward -= 0.05

        elif action == APPROACH:
            if self.distance > 0.8:
                self.distance = max(
                    0.0,
                    self.distance - 0.35
                )

                reward += 0.08
            else:
                reward -= 0.12

        elif action == RETREAT:
            self.distance = min(
                10.0,
                self.distance + 0.45
            )

            if (
                self.alpha_health < 50 or
                self.alpha_stamina < 30 or
                self.bravo_attack_telegraph == 1
            ):
                reward += 0.25
            else:
                reward -= 0.08

        elif action == ATTACK:
            reward += self._alpha_attack()

        elif action == BLOCK:
            if self.alpha_stamina >= 10:
                self.alpha_stamina -= 10
                self.alpha_blocking = True

                if self.bravo_attack_telegraph == 1:
                    reward += 0.30
                else:
                    reward -= 0.08
            else:
                reward -= 0.15

        elif action == DODGE:
            if self.alpha_stamina >= 15:
                self.alpha_stamina -= 15
                self.alpha_dodging = True

                self.distance = min(
                    10.0,
                    self.distance + 0.50
                )

                if self.bravo_attack_telegraph == 1:
                    reward += 0.55
                else:
                    reward -= 0.10
            else:
                reward -= 0.20

        # =================================
        # BRAVO DECISION
        # =================================

        bravo_action = self._choose_bravo_action()

        reward += self._perform_bravo_action(
            bravo_action
        )

        self.previous_alpha_action = action
        self.previous_bravo_action = bravo_action

        terminated = False
        truncated = False

        if self.bravo_health <= 0:
            terminated = True
            reward += 30.0

        elif self.alpha_health <= 0:
            terminated = True
            reward -= 30.0

        if self.steps >= self.max_steps:
            truncated = True
            reward -= 3.0

        observation = self._get_observation()

        info = {
            "alpha_health": self.alpha_health,
            "bravo_health": self.bravo_health,
            "distance": self.distance,
            "alpha_stamina": self.alpha_stamina,
            "bravo_stamina": self.bravo_stamina,
            "alpha_action": ACTION_NAMES[action],
            "bravo_action": ACTION_NAMES[bravo_action],
            "bravo_telegraph": self.bravo_attack_telegraph,
            "bravo_aggression": self.bravo_aggression,
        }

        return (
            observation,
            reward,
            terminated,
            truncated,
            info
        )

    def _alpha_attack(self):
        if self.alpha_cooldown > 0:
            return -0.20

        if self.alpha_stamina < 20:
            return -0.30

        self.alpha_stamina -= 20
        self.alpha_cooldown = 2

        if self.distance > 1.0:
            return -0.35

        damage = 20.0

        if self.bravo_blocking:
            damage *= 0.40

        self.bravo_health = max(
            0.0,
            self.bravo_health - damage
        )

        reward = damage * 0.12

        return reward

    def _choose_bravo_action(self):
        r = self.np_random.random()

        # Telegraph is cleared unless Bravo starts
        # another attack wind-up.
        if self.bravo_attack_telegraph == 1:

            # Bravo now commits to the attack.
            self.bravo_attack_telegraph = 0

            if (
                self.distance <= 1.0 and
                self.bravo_cooldown == 0 and
                self.bravo_stamina >= 20
            ):
                return ATTACK

        # Far range
        if self.distance > 1.5:

            if r < 0.75:
                return APPROACH

            if r < 0.85:
                return WAIT

            return RETREAT

        # Mid range
        if self.distance > 1.0:

            if r < 0.50:
                return APPROACH

            if r < 0.68:
                return BLOCK

            if r < 0.82:
                return DODGE

            return WAIT

        # Close range
        if (
            self.bravo_cooldown == 0 and
            self.bravo_stamina >= 20 and
            r < self.bravo_aggression
        ):
            # Telegraph now, attack next step.
            self.bravo_attack_telegraph = 1
            return WAIT

        if r < 0.65:
            return BLOCK

        if r < 0.80:
            return DODGE

        if r < 0.90:
            return RETREAT

        return WAIT

    def _perform_bravo_action(self, action):
        reward = 0.0

        if action == APPROACH:
            self.distance = max(
                0.0,
                self.distance - 0.30
            )

        elif action == RETREAT:
            self.distance = min(
                10.0,
                self.distance + 0.35
            )

        elif action == BLOCK:
            if self.bravo_stamina >= 10:
                self.bravo_stamina -= 10
                self.bravo_blocking = True

        elif action == DODGE:
            if self.bravo_stamina >= 15:
                self.bravo_stamina -= 15

                self.distance = min(
                    10.0,
                    self.distance + 0.45
                )

        elif action == ATTACK:
            if self.bravo_cooldown > 0:
                return reward

            if self.bravo_stamina < 20:
                return reward

            if self.distance > 1.0:
                return reward

            self.bravo_stamina -= 20
            self.bravo_cooldown = 2

            if self.alpha_dodging:
                reward += 2.50
                return reward

            damage = 15.0

            if self.alpha_blocking:
                damage *= 0.45
                reward += 0.75

            self.alpha_health = max(
                0.0,
                self.alpha_health - damage
            )

            reward -= damage * 0.12

        return reward

    def render(self):
        print(
            f"AHP={self.alpha_health:.0f} "
            f"ASTA={self.alpha_stamina:.0f} | "
            f"BHP={self.bravo_health:.0f} "
            f"BSTA={self.bravo_stamina:.0f} | "
            f"D={self.distance:.2f} | "
            f"TEL={self.bravo_attack_telegraph}"
        )


if __name__ == "__main__":

    env = FightEnvV22()

    observation, _ = env.reset()

    print("Starting observation:")
    print(observation)

    terminated = False
    truncated = False

    while not terminated and not truncated:

        action = env.action_space.sample()

        (
            observation,
            reward,
            terminated,
            truncated,
            info
        ) = env.step(action)

        print(
            f"Alpha={info['alpha_action']:8s} | "
            f"Bravo={info['bravo_action']:8s} | "
            f"Telegraph={info['bravo_telegraph']} | "
            f"Distance={info['distance']:.2f} | "
            f"AHP={info['alpha_health']:.0f} | "
            f"BHP={info['bravo_health']:.0f} | "
            f"Reward={reward:.2f}"
        )
