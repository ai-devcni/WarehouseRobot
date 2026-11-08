"""Robot Training Loop."""

import argparse

from agent import DQNAgent
from environment import RobotEnv


class TrainingConfig:
    def __init__(self, episodes=500, render=True, checkpoint="robot_dqn.pt"):
        self.episodes = episodes
        self.render = render
        self.checkpoint = checkpoint


def train(config: TrainingConfig):
    env = RobotEnv(render=config.render)
    agent = DQNAgent(state_size=7, action_size=3, lr=5e-4, epsilon=1.0)

    try:
        for episode in range(1, config.episodes + 1):
            state = env.reset()
            total_reward = 0.0
            done = False

            while not done:
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                agent.learn()
                state = next_state
                total_reward += reward
                env.render()

                if env.episode_steps >= 1000:
                    done = True

            print(f"Episode {episode:4d} | reward {total_reward:8.1f} | epsilon {agent.epsilon:.3f}")

            if episode % 50 == 0:
                agent.save(config.checkpoint)
    finally:
        env.close()

    agent.save(config.checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a tiny DQN robot to navigate a warehouse-like room.")
    parser.add_argument("--episodes", type=int, default=500, help="how many training episodes to run")
    parser.add_argument("--no-render", action="store_true", help="disable the Pygame window during training")
    parser.add_argument("--checkpoint", default="robot_dqn.pt", help="where to save the trained model")
    args = parser.parse_args()

    config = TrainingConfig(episodes=args.episodes, render=not args.no_render, checkpoint=args.checkpoint)
    train(config)
