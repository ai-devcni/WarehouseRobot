# WarehouseRobot

WarehouseRobot is a small reinforcement-learning project built to explore how a simple Deep Q-Network (DQN) agent learns to navigate a warehouse-like floor. The robot starts in a fixed corner of the room, scans its surroundings with a handful of forward-facing range sensors, and tries to reach a target dock while avoiding obstacles and walls.

The project is intentionally compact. It is designed to be easy to read and quick to run, while still capturing the basic loop of training an agent in a simulated environment.

## What the project does

- Builds a simple 2D Pygame environment.
- Places a robot and a goal in a room with static barriers.
- Gives the robot five lidar-style sensor readings across different angles.
- Lets the agent choose from three actions:
  - turn left
  - move forward
  - turn right
- Uses a replay memory and DQN update rule to learn a policy over time.
- Saves checkpoints during training so the model can be resumed or reused later.

## Why this project exists

The aim is not to produce a production-grade robotics stack. It is a practical learning project: a compact example of reinforcement learning in a small, controlled environment.

The scene is deliberately simple so the focus stays on:

- the environment design,
- the observation model,
- the training loop,
- and the behaviour of the learning agent.

## Project structure

- `agent.py` – contains the neural network and the DQN training logic.
- `environment.py` – defines the warehouse room, obstacles, sensor rays, and environment step logic.
- `main.py` – runs the training loop and handles command-line options.
- `requirements.txt` – Python dependencies for the project.

## Requirements

This project depends on:

- Python 3.10+
- PyGame
- NumPy
- PyTorch

Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Running the project

Train the agent in the default mode with a visible Pygame window:

```bash
python main.py --episodes 500
```

Run training without opening the visual display:

```bash
python main.py --episodes 500 --no-render
```

Use a custom checkpoint location:

```bash
python main.py --episodes 500 --checkpoint my_robot.pt
```

## How the environment is represented

The robot does not see the full map. Instead, it receives a compact state vector made up of:

- five ray-distance values from the front of the robot,
- the normalised distance to the goal,
- and the relative angle to the goal.

That means the policy is based on partial information only, which makes the task more realistic and more interesting for learning.

## Reward shaping

The environment gives a small negative reward for each step so the robot is encouraged to reach the goal quickly. It also rewards progress toward the target and penalises collisions with walls or obstacles.

The result is a simple but workable reinforcement-learning setup where the agent learns patterns such as:

- turning away from a wall,
- steering around obstacles,
- and approaching the target efficiently.

## Checkpoints

The project saves a model checkpoint every 50 episodes by default. The file is written to the location passed in via `--checkpoint` and contains the trained network weights and the current epsilon value.

## Notes

This is a learning project rather than a full warehouse automation system. The room, sensors, and movement model are intentionally small and deterministic so the behaviour is easy to follow and debug.

It is best viewed as a practical experiment in:

- agent-environment interaction,
- value-based reinforcement learning,
- and simple robotics simulation.
