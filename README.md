# AI Fight Arena

AI Fight Arena is a reinforcement-learning robotics project that combines **ROS 2, Gazebo, Gymnasium, and Stable-Baselines3 PPO** to train and deploy an autonomous simulated fighter.

The project separates reinforcement-learning training from robotic deployment:

- Alpha is trained in a custom Gymnasium combat environment using PPO.
- The trained neural-network policy is loaded into a ROS 2 controller.
- Gazebo provides live physical position feedback.
- ROS 2 bridges simulator state and robot movement commands.
- Bravo operates as an autonomous rule-based opponent.
- Alpha reacts to opponent state, including telegraphed attacks, using its learned policy.

The result is a closed-loop system where a trained RL policy makes decisions from live simulator state rather than following a fixed movement script.

---

## Demo Architecture

```text
                         +----------------------+
                         |       Gazebo         |
                         | Physics + Robot Pose |
                         +----------+-----------+
                                    |
                              pose feedback
                                    |
                                    v
                         +----------------------+
                         |    ros_gz_bridge     |
                         +----------+-----------+
                                    |
                                    v
       +------------------ PPO Alpha Controller ------------------+
       |                                                         |
       | Observation                                             |
       |  - health                                               |
       |  - opponent health                                      |
       |  - distance                                             |
       |  - stamina                                              |
       |  - cooldowns                                            |
       |  - Bravo attack telegraph                               |
       |  - previous actions                                     |
       |                                                         |
       +-------------------------+-------------------------------+
                                 |
                           PPO inference
                                 |
                                 v
                 +-------------------------------+
                 | WAIT / APPROACH / RETREAT     |
                 | ATTACK / BLOCK / DODGE        |
                 +---------------+---------------+
                                 |
                           ROS 2 commands
                                 |
                                 v
                         +---------------+
                         |    Gazebo     |
                         | Robot Motion  |
                         +---------------+

Bravo AI ---- attack telegraph / attack events ----> Alpha
```

---

## What the Project Demonstrates

- Reinforcement learning with **Proximal Policy Optimization (PPO)**
- Custom **Gymnasium** environment design
- Reward shaping and discrete-action RL
- Stable-Baselines3 model training and evaluation
- ROS 2 nodes, publishers, subscribers, and launch files
- Gazebo physics simulation
- `ros_gz_bridge` communication between ROS 2 and Gazebo
- Closed-loop control from live simulator feedback
- Autonomous multi-agent interaction
- Learned defensive behavior
- Health, stamina, attack range, cooldown, blocking, dodging, and KO systems
- Deployment of a trained ML policy into a robotic simulation

---

## PPO V2.2

The current policy is **Alpha PPO V2.2**.

Training configuration:

| Parameter | Value |
|---|---:|
| Algorithm | PPO |
| Policy | MLP |
| Training timesteps | 300,000 |
| Learning rate | 3e-4 |
| PPO rollout steps | 2,048 |
| Batch size | 64 |
| Gamma | 0.99 |
| GAE lambda | 0.95 |
| Entropy coefficient | 0.02 |

The trained model is stored at:

```text
training/alpha_policy_v22.zip
```

The training script is:

```text
training/train_v22.py
```

---

## V2.2 Observation Space

Alpha receives an 11-value state vector:

```text
0   Alpha health
1   Bravo health
2   Distance between fighters
3   Alpha stamina
4   Bravo stamina
5   Alpha attack cooldown
6   Bravo attack cooldown
7   Bravo attack telegraph
8   Previous Alpha action
9   Previous Bravo action
10  Bravo aggression parameter
```

The attack telegraph is particularly important because it gives the policy information about an imminent opponent attack and allows PPO to learn a context-dependent defensive response.

---

## Action Space

Alpha has six discrete actions:

```text
0  WAIT
1  APPROACH
2  RETREAT
3  ATTACK
4  BLOCK
5  DODGE
```

These actions are selected by the PPO model during inference.

---

## Evaluation Results

V2.2 was evaluated over **100 episodes** against the current rule-based Bravo opponent.

| Metric | Result |
|---|---:|
| Fights | 100 |
| Alpha wins | 100 |
| Bravo wins | 0 |
| Draws | 0 |
| Alpha win rate | 100.0% |
| Average Alpha HP remaining | 76.6 |
| Average reward | 42.77 |
| Average fight length | 19.5 steps |
| Bravo attack telegraphs observed | 346 |
| Defensive response rate | 100.0% |

### Overall Alpha Action Usage

| Action | Usage |
|---|---:|
| APPROACH | 51.03% |
| ATTACK | 31.19% |
| BLOCK | 17.78% |
| WAIT | 0.00% |
| RETREAT | 0.00% |
| DODGE | 0.00% |

### Response to Telegraphed Attacks

During evaluation, Bravo telegraphed an attack **346 times**.

Alpha responded with:

```text
BLOCK: 346 / 346
```

That means the learned policy did not simply block continuously. BLOCK represented only 17.78% of Alpha's overall actions, but it was selected for every observed attack telegraph during this evaluation.

This demonstrates a **state-dependent learned defensive strategy**.

> The 100% win rate applies to the current V2.2 evaluation environment and rule-based opponent. It should not be interpreted as performance against arbitrary or unseen opponents.

---

## Live Gazebo Deployment

The trained V2.2 policy was subsequently integrated into the live ROS 2/Gazebo simulation.

A representative fight demonstrated the following sequence:

```text
Alpha and Bravo physically approach each other
                ↓
Bravo enters attack range
                ↓
Bravo publishes an attack telegraph
                ↓
Alpha receives TEL=1
                ↓
PPO selects BLOCK
                ↓
Alpha activates its guard
                ↓
Bravo executes the attack
                ↓
Incoming damage is reduced
                ↓
Alpha returns to ATTACK
                ↓
Fight ends in a KO
```

In the live Gazebo test, an unblocked Bravo attack caused:

```text
15.0 damage
```

A successfully blocked attack caused approximately:

```text
6.8 damage
```

This verifies that the learned decision was connected to the live combat state rather than existing only inside the training environment.

---

## Repository Structure

```text
ai-fight-arena/
│
├── src/
│   └── arena_control/
│       │
│       ├── arena_control/
│       │   ├── fighter_alpha.py
│       │   ├── fighter_bravo.py
│       │   ├── bravo_ai.py
│       │   ├── fight_controller.py
│       │   └── __init__.py
│       │
│       ├── launch/
│       │   └── fight_demo.launch.py
│       │
│       ├── package.xml
│       ├── setup.py
│       └── setup.cfg
│
├── training/
│   ├── fight_env_v22.py
│   ├── train_v22.py
│   ├── evaluate_v22.py
│   ├── rl_fighter_alpha.py
│   └── alpha_policy_v22.zip
│
├── worlds/
│   └── arena.sdf
│
├── .gitignore
└── README.md
```

---

## Training the Policy

Create or activate a Python environment containing the required ML dependencies, including:

- Python
- NumPy
- Gymnasium
- Stable-Baselines3
- PyTorch

Then:

```bash
cd training
python train_v22.py
```

The trained model will be saved as:

```text
alpha_policy_v22.zip
```

Training is **not required** to run the already-saved V2.2 model.

---

## Evaluating the Policy

From the training directory:

```bash
python evaluate_v22.py
```

The evaluator measures:

- win rate
- remaining Alpha health
- average episode reward
- episode length
- action distribution
- opponent telegraph events
- Alpha's response to telegraphed attacks
- defensive response rate

---

## Building the ROS 2 Workspace

From the workspace root:

```bash
colcon build \
  --base-paths src \
  --symlink-install
```

Then source ROS 2 and the workspace:

```bash
source /opt/ros/lyrical/setup.bash
source install/setup.bash
```

---

## Running the Gazebo Fight

Launch the complete simulation with:

```bash
ros2 launch arena_control fight_demo.launch.py
```

The launch system starts:

- Gazebo
- Fighter Alpha's low-level controller
- Fighter Bravo's low-level controller
- ROS/Gazebo velocity bridges
- Gazebo pose feedback bridge
- Bravo's autonomous controller
- Alpha's trained PPO controller

---

## Technology Stack

**Robotics / Simulation**

- ROS 2
- Gazebo
- ros_gz_bridge

**Machine Learning**

- Python
- Gymnasium
- Stable-Baselines3
- PPO
- PyTorch
- NumPy

**Development Environment**

- Ubuntu / WSL2
- colcon
- Git
- GitHub

---

## Current Milestone

The current version demonstrates an end-to-end RL workflow:

```text
custom environment
        ↓
PPO training
        ↓
policy evaluation
        ↓
saved neural-network policy
        ↓
ROS 2 integration
        ↓
Gazebo deployment
        ↓
live state feedback
        ↓
learned autonomous decisions
```

The Gazebo/PPO V2.2 milestone is considered the first completed portfolio version of the project.

---

## Planned Development

### Multi-Agent Self-Play

The next major RL extension is to replace the fixed rule-based opponent with a second learning policy.

The intended system is:

```text
Alpha PPO
    ↕
competitive self-play
    ↕
Bravo PPO
```

Rather than permanently making one fighter stronger, Alpha and Bravo will learn against changing opponent policies. Either fighter may dominate for a period until the opposing policy adapts.

Historical opponent policies may also be retained in a policy pool to reduce overfitting to only the newest opponent.

### NVIDIA Isaac Sim

The project is also planned for deployment in **NVIDIA Isaac Sim**.

The goal is to reuse the existing ROS 2 / learned-policy architecture with a second robotics simulator and explore:

- Isaac Sim robot control
- ROS 2 integration
- policy inference using Isaac simulation state
- richer simulated sensors
- potential depth-camera or LiDAR observations
- comparison between Gazebo and Isaac Sim workflows

---

## Project Status

**PPO V2.2 training:** Complete  
**100-episode evaluation:** Complete  
**ROS 2 integration:** Complete  
**Gazebo deployment:** Complete  
**Closed-loop PPO control:** Complete  
**Telegraph-aware blocking:** Complete  
**GitHub portfolio release:** Complete  
**Isaac Sim port:** Planned  
**PPO vs. PPO self-play:** Planned

---

## Author

**Thomas Byrne**

Computer Science / Cybersecurity background with interests in machine learning, autonomous systems, robotics simulation, AI engineering, and reinforcement learning.

