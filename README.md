
# AI Fight Arena

A reinforcement-learning robotics simulation that trains a PPO agent to make autonomous combat decisions and deploys the learned policy into a live ROS 2 + Gazebo control loop.

Alpha is trained in a custom Gymnasium environment using Stable-Baselines3 PPO. The trained policy is then loaded by a ROS 2 node that receives live Gazebo pose feedback and publishes movement and combat commands.

Bravo currently uses a rule-based controller with an explicit attack telegraph, allowing Alpha to react to incoming attacks using its learned policy.

## Project Highlights

- Reinforcement learning with Proximal Policy Optimization (PPO)
- Custom Gymnasium environment and reward shaping
- Stable-Baselines3 training and evaluation
- ROS 2 publishers, subscribers, nodes, and launch files
- Gazebo physics simulation
- ROS 2 / Gazebo integration through `ros_gz_bridge`
- Closed-loop learned control using real simulator state
- Multi-agent interaction between PPO-controlled Alpha and rule-based Bravo
- Health, stamina, cooldowns, attack range, blocking, dodging, and KO logic
- Opponent attack telegraphing and state-dependent defensive decisions

## V2.2 Evaluation Results

The current PPO V2.2 policy was trained for:

**300,000 timesteps**

It was then evaluated over 100 fights.

| Metric | Result |
|---|---:|
| Alpha wins | 100 / 100 |
| Alpha win rate | 100.0% |
| Average Alpha HP remaining | 76.6 |
| Average reward | 42.77 |
| Average fight length | 19.5 steps |
| Telegraph events | 346 |
| Defensive response rate | 100.0% |

### Alpha Action Usage

| Action | Usage |
|---|---:|
| APPROACH | 51.03% |
| ATTACK | 31.19% |
| BLOCK | 17.78% |
| WAIT | 0.00% |
| RETREAT | 0.00% |
| DODGE | 0.00% |

When Bravo telegraphed an incoming attack, Alpha selected:

**BLOCK on 346 / 346 telegraph events**

This demonstrates state-dependent defensive behavior rather than continuous block usage.

These results describe performance against the current V2.2 opponent/environment distribution and should not be interpreted as performance against arbitrary opponents.

## Live Gazebo Deployment

The trained V2.2 policy has also been deployed into the live ROS 2 / Gazebo simulation.

The closed-loop system works like this:

```text
             Gazebo
               |
               | pose feedback
               v
        ros_gz_bridge
               |
               v
     PPO Alpha Controller
               ^
               |
       Bravo Telegraph
               |
               v
      PPO Action Selection
               |
       +-------+-------+
       |       |       |
   APPROACH  ATTACK  BLOCK
   RETREAT   DODGE    WAIT
       |       |       |
       +-------+-------+
               |
               v
       ROS 2 Controller
               |
               v
        Gazebo Motion
