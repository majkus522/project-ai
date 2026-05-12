import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from game import Game
import numpy as np
import time
import sys

#Parameters
GAMMA = 0.99
BATCH_SIZE = 100
EPISODES = 1000
epsilon = 1.0

#Neural network
class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16, int(sys.argv[1])),
            nn.ReLU(),
            nn.Linear(int(sys.argv[1]), 4)
        )

    def forward(self, x):
        return self.net(x)

#Init
env = Game()
policy_net = DQN()
optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
memory = deque(maxlen=EPISODES)
best_reward = float("-inf")
best_game = []
start = 0

#Load checkpoint
"""
if os.path.isfile("checkpoint.pth"):
    checkpoint = torch.load("checkpoint.pth")
    policy_net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epsilon = checkpoint['epsilon']
    start = checkpoint['episode']
"""

target_net = DQN()
target_net.load_state_dict(policy_net.state_dict())

#Action
def select_action(state):
    if random.random() < epsilon:
        return random.choice([0, 1, 2, 3])
    else:
        with torch.no_grad():
            state_t = torch.FloatTensor(state).view(1, -1)
            q_values = policy_net(state_t).squeeze()
            return q_values.argmax().item()

def optimize():
    if len(memory) < BATCH_SIZE:
        return

    batch = random.sample(memory, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.from_numpy(np.stack(states)).float()
    actions = torch.LongTensor(actions).unsqueeze(1)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.from_numpy(np.stack(next_states)).float()
    dones = torch.FloatTensor(dones)

    q_values = policy_net(states).gather(1, actions).squeeze()
    next_q_values = target_net(next_states).max(1)[0].detach()

    target = rewards + GAMMA * next_q_values * (1 - dones)

    loss = nn.MSELoss()(q_values, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

#Training
allMax = 0
try:
    for episode in range(start, EPISODES):
        state = env.reset()
        total_reward = 0
        current_game = []
        maxTile = 0

        while True:
            action = select_action(state)
            next_state, reward, done = env.step(action)
            current_game.append((state.copy(), action, reward))

            memory.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward
            maxTile = max(state)
            allMax = max(allMax, maxTile)
            optimize()

            if done:
                if total_reward > best_reward:
                    best_reward = total_reward
                    best_game = current_game.copy()
                    torch.save(best_game, "best_game.pth")
                break

        epsilon = max(0.1, epsilon * 0.995)

        if episode % 100 == 0:
            target_net.load_state_dict(policy_net.state_dict())
           # print(f"Episode {episode}, Reward: {total_reward}, Epsilon: {epsilon}, Max: {maxTile}, All Max: {allMax}")
            """
            torch.save({
                'model_state_dict': policy_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epsilon': epsilon,
                'episode': episode
            }, "checkpoint.pth")
            """

except KeyboardInterrupt:
    """
    torch.save({
        'model_state_dict': policy_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epsilon': epsilon,
        'episode': episode
    }, "checkpoint.pth")
    torch.save(best_game, "best_game.pth")
    """
    print("💾 Progress saved!")

def replay_best_game():
    game_data = torch.load("best_game.pth", weights_only=False)
    for i, (state, action, reward) in enumerate(game_data):
        print(state)
        print(action, reward)
        time.sleep(0.3)

#replay_best_game()

print(f"Best reward: {best_reward}, Best tile: {allMax}")