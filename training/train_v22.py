from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from fight_env_v22 import FightEnvV22


def main():
    env = Monitor(
        FightEnvV22()
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.02,
    )

    print("Training Alpha V2.2...")

    model.learn(
        total_timesteps=300_000
    )

    model.save(
        "alpha_policy_v22"
    )

    print(
        "Training complete. "
        "Saved alpha_policy_v22.zip"
    )


if __name__ == "__main__":
    main()
