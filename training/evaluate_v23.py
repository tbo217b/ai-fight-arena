from collections import Counter

from stable_baselines3 import PPO

from fight_env_v23 import (
    FightEnvV23,
    ACTION_NAMES,
    ATTACK,
    BLOCK,
    DODGE,
)


FIGHTS = 200

env = FightEnvV23()

model = PPO.load(
    "alpha_policy_v23.zip",
    device="cpu",
)

alpha_wins = 0
bravo_wins = 0
draws = 0

alpha_hp_sum = 0.0
reward_sum = 0.0
step_sum = 0

actions = Counter()

telegraph_events = 0
telegraph_actions = Counter()

attack_phase_events = 0
attack_phase_actions = Counter()


for fight in range(1, FIGHTS + 1):

    obs, _ = env.reset()

    terminated = False
    truncated = False

    total_reward = 0.0
    steps = 0

    while not terminated and not truncated:

        telegraph_before = int(obs[7])
        bravo_action_before = int(obs[9])

        action, _ = model.predict(
            obs,
            deterministic=True,
        )

        action = int(action)

        actions[action] += 1

        if telegraph_before == 1:
            telegraph_events += 1
            telegraph_actions[action] += 1

        if bravo_action_before == ATTACK:
            attack_phase_events += 1
            attack_phase_actions[action] += 1

        (
            obs,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        total_reward += reward
        steps += 1

    if info["bravo_health"] <= 0:
        alpha_wins += 1

    elif info["alpha_health"] <= 0:
        bravo_wins += 1

    else:
        draws += 1

    alpha_hp_sum += info["alpha_health"]
    reward_sum += total_reward
    step_sum += steps


print()
print("=" * 62)
print("V2.3 ISAAC-TRANSFER TRAINING ENV EVALUATION")
print("=" * 62)

print(f"Fights:                 {FIGHTS}")
print(f"Alpha wins:             {alpha_wins}")
print(f"Bravo wins:             {bravo_wins}")
print(f"Draws:                  {draws}")
print(f"Alpha win rate:         {100.0 * alpha_wins / FIGHTS:.1f}%")
print(f"Average Alpha HP:       {alpha_hp_sum / FIGHTS:.1f}")
print(f"Average reward:         {reward_sum / FIGHTS:.2f}")
print(f"Average fight length:   {step_sum / FIGHTS:.1f} decisions")

print()
print("=" * 62)
print("OVERALL ALPHA ACTION USAGE")
print("=" * 62)

total_actions = sum(actions.values())

for action in range(6):
    count = actions[action]
    pct = 100.0 * count / total_actions if total_actions else 0.0

    print(
        f"{ACTION_NAMES[action]:10s}: "
        f"{count:6d} ({pct:6.2f}%)"
    )

print()
print("=" * 62)
print("ALPHA RESPONSE TO BRAVO TELEGRAPH")
print("=" * 62)

print(f"Telegraph events:       {telegraph_events}")

for action in range(6):
    count = telegraph_actions[action]
    pct = (
        100.0 * count / telegraph_events
        if telegraph_events
        else 0.0
    )

    print(
        f"{ACTION_NAMES[action]:10s}: "
        f"{count:6d} ({pct:6.2f}%)"
    )

print()
print("=" * 62)
print("ALPHA RESPONSE TO COMMITTED BRAVO ATTACK")
print("=" * 62)

print(f"Attack-phase events:    {attack_phase_events}")

for action in range(6):
    count = attack_phase_actions[action]
    pct = (
        100.0 * count / attack_phase_events
        if attack_phase_events
        else 0.0
    )

    print(
        f"{ACTION_NAMES[action]:10s}: "
        f"{count:6d} ({pct:6.2f}%)"
    )

defensive = (
    attack_phase_actions[BLOCK]
    + attack_phase_actions[DODGE]
)

defensive_rate = (
    100.0 * defensive / attack_phase_events
    if attack_phase_events
    else 0.0
)

print()
print(
    f"Attack-phase defensive response rate: "
    f"{defensive_rate:.1f}%"
)
