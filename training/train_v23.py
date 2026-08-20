from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from fight_env_v23 import FightEnvV23


env = Monitor(FightEnvV23())

print("=" * 60)
print("PPO V2.3 ISAAC-TRANSFER FINE-TUNE")
print("=" * 60)

# Warm-start from the known V2.2 policy rather than learning
# everything again from random initialization.
model = PPO.load(
    "alpha_policy_v22.zip",
    env=env,
    device="cpu",
)

print("Loaded alpha_policy_v22.zip")
print("Fine-tuning for 400,000 additional timesteps...")

model.learn(
    total_timesteps=400_000,
    reset_num_timesteps=False,
    progress_bar=False,
)

model.save("alpha_policy_v23")

print()
print("Training complete.")
print("Saved alpha_policy_v23.zip")
