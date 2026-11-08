"""A compact DQN agent for the warehouse robot demo."""

from collections import deque
import random

import numpy as np
import torch
from torch import nn


class DQN(nn.Module):
    """Main value network that estimates the best action from the current state."""

    def __init__(self, state_size=7, action_size=3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
        )

    def forward(self, state):
        return self.network(state)


class DQNAgent:
    """A basic learning agent with replay memory, decaying exploration, and a target network."""

    def __init__(
        self,
        state_size=7,
        action_size=3,
        lr=5e-4,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        memory_size=50_000,
        batch_size=64,
        target_update=200,
        device=None,
    ):
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.memory = deque(maxlen=memory_size)
        self.training_steps = 0

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.q_network = DQN(state_size, action_size).to(self.device)
        self.target_network = DQN(state_size, action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def choose_action(self, state):
        """Epsilon-greedy policy: explore early, act greedily once the policy improves."""
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        with torch.no_grad():
            tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            action = self.q_network(tensor).argmax(dim=1).item()
        return int(action)

    def act(self, state):
        return self.choose_action(state)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def learn(self):
        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(np.asarray(next_states), dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        predicted = self.q_network(states).gather(1, actions).squeeze(1)

        with torch.no_grad():
            target_q_values = self.target_network(next_states).max(dim=1).values
            targets = rewards + self.gamma * target_q_values * (1.0 - dones)

        loss = self.loss_fn(predicted, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()

        self.training_steps += 1
        if self.training_steps % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return float(loss.item())

    def save(self, path):
        checkpoint = {
            "model": self.q_network.state_dict(),
            "target_model": self.target_network.state_dict(),
            "epsilon": self.epsilon,
        }
        torch.save(checkpoint, path)

    def load(self, path):
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        except TypeError:
            checkpoint = torch.load(path, map_location=self.device)

        self.q_network.load_state_dict(checkpoint["model"])
        if "target_model" in checkpoint:
            self.target_network.load_state_dict(checkpoint["target_model"])
        else:
            self.target_network.load_state_dict(checkpoint["model"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon_min)
