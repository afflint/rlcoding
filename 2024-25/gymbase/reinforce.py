import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import cv2

# Hyperparameters
LEARNING_RATE = 3e-4
GAMMA = 0.99  # Discount factor
EPISODES = 3000  # Number of training episodes
BATCH_SIZE = 10  # Update policy after X episodes
HIDDEN_UNITS = 256  # Reduced hidden layer size for efficiency

def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (64, 64)) 
    frame = frame.astype(np.float32) / 255.0  
    return np.expand_dims(frame, axis=0)  # Shape: (1, 64, 64)

class PolicyNetwork(nn.Module):
    def __init__(self, action_dim, hidden_units: int = 256):
        super(PolicyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2)  
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2) 
        self.flattened_size = 32 * 14 * 14
        self.fc1 = nn.Linear(self.flattened_size, hidden_units)  
        self.fc2 = nn.Linear(hidden_units, action_dim)

        self.log_std = nn.Parameter(torch.ones(action_dim) * -1.0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  
        x = F.relu(self.fc1(x))
        mean = torch.tanh(self.fc2(x)) 
        std = torch.exp(self.log_std)  
        return mean, std

class PolicyGradientAgent:
    def __init__(self, action_dim, learning_rate: float = 3e-4, gamma: float = .99):
        self.device = torch.device("mps" if torch.mps.is_available() else "cpu")
        self.policy = PolicyNetwork(action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.memory = []  # Stores (log_prob, reward, state)
        self.gamma = gamma

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mean, std = self.policy(state)

        # Sample action from Gaussian policy
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        action = torch.clamp(action, -1, 1)  
        log_prob = dist.log_prob(action).sum(dim=-1)  # Compute log probability

        return action.cpu().numpy().squeeze(), log_prob

    def store_outcome(self, log_prob, reward, state):
        self.memory.append((log_prob, reward, state))

    def update_policy(self):
        R = 0  # Discounted return
        policy_loss = []
        returns = []

        # Compute discounted rewards
        for _, reward, _ in reversed(self.memory):
            R = reward + self.gamma * R
            returns.insert(0, R)  

        returns = torch.tensor(returns).to(self.device)

        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        baseline = sum(returns) / len(returns)

        # Compute policy gradient loss with advantage function
        for (log_prob, _, _), R in zip(self.memory, returns):
            advantage = R - baseline  
            policy_loss.append(-log_prob * advantage)  # Gradient ascent

        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()

        self.memory = []

