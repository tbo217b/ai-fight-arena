from collections import Counter

from stable_baselines3 import PPO

from fight_env_v22 import FightEnvV22, ACTION_NAMES


NUM_FIGHTS = 100

model = PPO.load("alpha_policy_v22")
env = FightEnvV22()

alpha_wins = 0
bravo_wins = 0
draws = 0

total_reward = 0.0
total_steps = 0
total_alpha_hp = 0.0

action_counts = Counter()
telegraph_action_counts = Counter()

telegraph_events = 0


for fight in range(1, NUM_FIGHTS + 1):

    obs, _ = env.reset()

    terminated = False
    truncated = False

    fight_reward = 0.0
    steps = 0

    while not (terminated or truncated):

        # Telegraph is observation index 7.
        telegraph_active = obs[7] >= 0.5

        action, _ = model.predict(
            obs,
            deterministic=True
        )

        action = int(action)

        action_name = ACTION_NAMES[action]

        action_counts[action_name] += 1

        if telegraph_active:
            telegraph_events += 1
            telegraph_action_counts[action_name] += 1

        obs, reward, terminated, truncated, info = env.step(
            action
        )

        fight_reward += reward
        steps += 1

    total_reward += fight_reward
    total_steps += steps
    total_alpha_hp += env.alpha_health

    if env.bravo_health <= 0 and env.alpha_health > 0:
        alpha_wins += 1

    elif env.alpha_health <= 0 and env.bravo_health > 0:
        bravo_wins += 1

    else:
        draws += 1


env.close()


print()
print("=" * 60)
print("V2.2 PPO EVALUATION")
print("=" * 60)

print(f"Fights:                 {NUM_FIGHTS}")
print(f"Alpha wins:             {alpha_wins}")
print(f"Bravo wins:             {bravo_wins}")
print(f"Draws:                  {draws}")

print(
    f"Alpha win rate:         "
    f"{100.0 * alpha_wins / NUM_FIGHTS:.1f}%"
)

print(
    f"Average Alpha HP:       "
    f"{total_alpha_hp / NUM_FIGHTS:.1f}"
)

print(
    f"Average reward:         "
    f"{total_reward / NUM_FIGHTS:.2f}"
)

print(
    f"Average fight length:   "
    f"{total_steps / NUM_FIGHTS:.1f} steps"
)


print()
print("=" * 60)
print("OVERALL ALPHA ACTION USAGE")
print("=" * 60)

total_actions = sum(action_counts.values())

for action_name in ACTION_NAMES.values():

    count = action_counts[action_name]

    percent = (
        100.0 * count / total_actions
        if total_actions > 0
        else 0.0
    )

    print(
        f"{action_name:10s}: "
        f"{count:5d} "
        f"({percent:6.2f}%)"
    )


print()
print("=" * 60)
print("ALPHA RESPONSE WHEN BRAVO TELEGRAPHS ATTACK")
print("=" * 60)

print(
    f"Telegraph events:       {telegraph_events}"
)

for action_name in ACTION_NAMES.values():

    count = telegraph_action_counts[action_name]

    percent = (
        100.0 * count / telegraph_events
        if telegraph_events > 0
        else 0.0
    )

    print(
        f"{action_name:10s}: "
        f"{count:5d} "
        f"({percent:6.2f}%)"
    )


defensive_responses = (
    telegraph_action_counts["BLOCK"] +
    telegraph_action_counts["DODGE"] +
    telegraph_action_counts["RETREAT"]
)

defensive_rate = (
    100.0 * defensive_responses / telegraph_events
    if telegraph_events > 0
    else 0.0
)

print()
print(
    f"Defensive response rate: {defensive_rate:.1f}%"
)
