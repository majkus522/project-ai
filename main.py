import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from game import Game

GAMMA = 0.99
BATCH_SIZE = 100
EPISODES = 2000
epsilon = 1.0

#Neural network
class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9, 128),
            nn.ReLU(),
            nn.Linear(128, 9)
        )

    def forward(self, x):
        return self.net(x)

#Init
env = Game()
policy_net = DQN()
optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
memory = deque(maxlen=EPISODES)

#Load save
if os.path.isfile("checkpoint.pth"):
    checkpoint = torch.load("checkpoint.pth")
    policy_net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epsilon = checkpoint['epsilon']

target_net = DQN()
target_net.load_state_dict(policy_net.state_dict())

#Action
def select_action(state, epsilon):
    if random.random() < epsilon:
        return random.choice(env.available_actions())
    else:
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            q_values = policy_net(state_t).squeeze()

            # Mask invalid moves
            for i in range(9):
                if state[i] != 0:
                    q_values[i] = -1e9

            return q_values.argmax().item()

def optimize():
    if len(memory) < BATCH_SIZE:
        return

    batch = random.sample(memory, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions).unsqueeze(1)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones)

    q_values = policy_net(states).gather(1, actions).squeeze()
    next_q_values = target_net(next_states).max(1)[0].detach()

    target = rewards + GAMMA * next_q_values * (1 - dones)

    loss = nn.MSELoss()(q_values, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

#Training
for episode in range(EPISODES):
    state = env.reset()
    total_reward = 0

    while True:
        action = select_action(state, epsilon)
        next_state, reward, done = env.step(action)

        memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        optimize()

        if done:
            break

    epsilon = max(0.001, epsilon * 0.995)

    if episode % 100 == 0:
        target_net.load_state_dict(policy_net.state_dict())
        print(f"Episode {episode}, Reward: {total_reward}, Epsilon: {epsilon}")

#Save
torch.save({
    'model_state_dict': policy_net.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epsilon': epsilon,
}, "checkpoint.pth")

wins, losses, draws = 0, 0, 0

#Presentation
for _ in range(100):
    state = env.reset()

    while True:
        action = select_action(state, epsilon=0.0)
        state, reward, done = env.step(action)

        if done:
            if reward == 1:
                wins += 1
            elif reward == -1:
                losses += 1
            else:
                draws += 1
            break

print(f"Wins: {wins}, Losses: {losses}, Draws: {draws}")