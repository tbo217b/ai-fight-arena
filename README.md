# AI Fight Arena

AI Fight Arena is a reinforcement-learning robotics project that trains and deploys an autonomous simulated fighter across **ROS 2, Gazebo, and NVIDIA Isaac Sim**.

The project demonstrates an end-to-end reinforcement-learning workflow:

**ROS 2 / Gazebo prototype → PPO training → policy evaluation → NVIDIA Isaac Sim deployment → transfer analysis → policy adaptation**

The goal is to explore how an autonomous agent trained in one simulated environment behaves when transferred into a different simulator with different execution dynamics.

---

## Project Highlights

- Built a two-agent simulated combat environment using ROS 2 and Gazebo.
- Developed a reinforcement-learning environment with six discrete actions.
- Trained the Alpha fighter using Proximal Policy Optimization (PPO).
- Implemented opponent attack telegraphing, blocking, dodging, stamina, health, and movement mechanics.
- Deployed the trained PPO policy into NVIDIA Isaac Sim.
- Evaluated sim-to-sim transfer performance.
- Identified a major transfer-performance failure in PPO V2.2.
- Adapted the training environment and produced PPO V2.3.
- Improved Isaac Sim evaluation performance from **15% to 100% win rate** under the current evaluation configuration.

---

## System Architecture

```text
                  AI Fight Arena
                        |
        +---------------+---------------+
        |                               |
     ROS 2 / Gazebo                 PPO Training
        |                               |
  Fighter simulation              Stable-Baselines3
        |                               |
        +---------------+---------------+
                        |
                 Trained Policy
                        |
                 NVIDIA Isaac Sim
                        |
                Transfer Evaluation
                        |
             V2.2: 15% win rate
                        |
                 Policy Adaptation
                        |
             V2.3: 100% win rate
```

---

## Reinforcement Learning

Alpha uses a PPO policy with an 11-dimensional observation space and six discrete actions:

| Action | Behavior |
|---|---|
| WAIT | Hold position |
| APPROACH | Move toward opponent |
| RETREAT | Increase distance |
| ATTACK | Perform an attack |
| BLOCK | Reduce incoming damage |
| DODGE | Evade an incoming attack |

The environment includes:

- Health
- Stamina
- Fighter distance
- Attack range
- Opponent attack telegraphs
- Committed attack phases
- Blocking
- Dodging
- Movement
- Damage
- Knockouts

---

## PPO V2.2

The initial PPO V2.2 model was trained in the custom fight environment.

The trained model is stored at:

```text
training/alpha_policy_v22.zip
```

When transferred into Isaac Sim, the policy loaded successfully with:

```text
Observation space: (11,)
Action space: Discrete(6)
```

However, deployment exposed a significant **sim-to-sim transfer problem**.

### V2.2 Isaac Sim Evaluation

| Metric | Result |
|---|---:|
| Fights | 20 |
| Alpha wins | 3 |
| Bravo wins | 17 |
| Alpha win rate | **15.0%** |
| Average Alpha HP | 2.9 |
| Average Bravo HP | 17.0 |
| Average fight length | 45.6 decisions |
| Alpha attacks | 83 |
| Alpha blocks | 403 |
| Alpha dodges | 0 |

Although Alpha recognized many attack situations, its behavior transferred poorly. It heavily favored blocking and frequently failed to finish Bravo when Bravo reached low health.

This became the primary transfer problem addressed in V2.3.

---

## PPO V2.3 Transfer Adaptation

PPO V2.3 was developed after analyzing the V2.2 Isaac Sim failure.

The training behavior was modified to better distinguish between:

1. **Opponent telegraph phase**
2. **Opponent committed attack phase**

This encouraged the policy to learn different offensive and defensive responses depending on the stage of Bravo's attack.

The resulting policy is stored at:

```text
training/alpha_policy_v23.zip
```

### V2.3 Training Evaluation

V2.3 was evaluated across 200 fights:

| Metric | Result |
|---|---:|
| Fights | 200 |
| Alpha wins | 200 |
| Bravo wins | 0 |
| Alpha win rate | **100.0%** |
| Average Alpha HP | 79.8 |
| Average reward | 60.11 |
| Average fight length | 15.6 decisions |

The policy also demonstrated a **100% defensive response rate during committed Bravo attacks** in this evaluation.

---

## NVIDIA Isaac Sim Deployment

The trained policy was then deployed into **NVIDIA Isaac Sim 6.0.1** on a cloud GPU environment.

The Isaac Sim environment ran on an **NVIDIA L40S GPU**.

The deployment includes:

- Isaac Sim physics simulation
- Alpha PPO policy inference
- Rule-based Bravo opponent
- Health and stamina systems
- Attack telegraphing
- Blocking
- Dodging
- Movement
- Knockout detection
- Automated multi-fight evaluation

The main Isaac Sim demo is:

```text
isaac/isaac_fight_arena_v23.py
```

Evaluation scripts are located under:

```text
isaac/
```

---

## V2.3 Isaac Sim Results

The adapted PPO V2.3 policy was evaluated in 20 Isaac Sim fights.

| Metric | Result |
|---|---:|
| Fights | 20 |
| Alpha wins | 20 |
| Bravo wins | 0 |
| Alpha win rate | **100.0%** |
| Average Alpha HP | 79.8 |
| Average Bravo HP | 0.0 |
| Average fight length | 15.0 decisions |
| Alpha attacks | 100 |
| Alpha blocks | 60 |
| Alpha dodges | 20 |
| Bravo telegraphs | 100 |

The complete results are stored in:

```text
results/isaac_v23_results.txt
```

---

## Transfer Improvement

The most important result of the project was the improvement between V2.2 and V2.3 after deployment exposed a transfer failure.

| Policy | Isaac Sim Wins | Win Rate |
|---|---:|---:|
| PPO V2.2 | 3 / 20 | **15%** |
| PPO V2.3 | 20 / 20 | **100%** |

This represents an **85 percentage-point improvement** under the same current opponent/evaluation configuration.

The result demonstrates an iterative robotics/ML workflow:

```text
Train
  ↓
Deploy
  ↓
Measure transfer failure
  ↓
Analyze agent behavior
  ↓
Modify training environment
  ↓
Retrain / adapt
  ↓
Redeploy
  ↓
Re-evaluate
```

The 100% result applies specifically to the current Bravo opponent and evaluation configuration and should not be interpreted as performance against arbitrary opponents or environments.

---

## Repository Structure

```text
ai-fight-arena/
├── isaac/
│   ├── isaac_evaluate_v22.py
│   ├── isaac_evaluate_v23.py
│   ├── isaac_fight_arena.py
│   └── isaac_fight_arena_v23.py
│
├── results/
│   └── isaac_v23_results.txt
│
├── training/
│   ├── alpha_policy_v22.zip
│   ├── alpha_policy_v23.zip
│   ├── evaluate_v22.py
│   ├── evaluate_v23.py
│   ├── fight_env_v22.py
│   ├── fight_env_v23.py
│   ├── rl_fighter_alpha.py
│   ├── train_v22.py
│   └── train_v23.py
│
├── src/
├── worlds/
└── README.md
```

---

## Technologies

- Python
- ROS 2
- Gazebo
- NVIDIA Isaac Sim
- NVIDIA Isaac Lab
- Stable-Baselines3
- PPO reinforcement learning
- Gymnasium
- NumPy
- PyTorch
- CUDA
- Git / GitHub

---

## Key Takeaway

AI Fight Arena is not only a reinforcement-learning training demonstration.

The project shows the complete process of taking a trained policy, deploying it into another robotics simulator, discovering that the original policy does not transfer successfully, measuring the failure, adapting the learning environment, and validating the improved policy through repeated simulation.

The transition from **15% V2.2 Isaac Sim win rate to 100% V2.3 win rate** is the project's primary sim-to-sim transfer result.