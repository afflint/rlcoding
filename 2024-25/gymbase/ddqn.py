import gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import cv2

# Preprocessing function converting to grayscale
def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (84, 84))
    # frame = np.expand_dims(frame, axis=0)
    return frame.astype(np.float32) / 255.0

# Neural network for Q-learning
class DQNCNN(nn.Module):
    def __init__(self, action_dim):
        super(DQNCNN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, action_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# DQN Agent with Double DQN
class DQNAgent:
    def __init__(self, action_dim, 
                 learning_rate: float = 1e-4, 
                 buffer_size: int = 100000,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.1,
                 epsilon_decay_rate: int = 1_000_000,
                 start_training: int = 10000,
                 batch_size: int = 32,
                 gamma: float = 0.99
                 ):
        self.action_dim = action_dim
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        self.q_network = DQNCNN(action_dim).to(self.device)
        self.target_network = DQNCNN(action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(buffer_size)

        # Exploration parameters
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay_rate
        self.epsilon_end = epsilon_end
        self.epsilon_start = epsilon_start

        self.start_training = start_training
        self.batch_size = batch_size

        self.gamma = gamma

        self.steps_done = 0

    def exploration_policy(self, state):
        self.steps_done += 1
        eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-1.0 * self.steps_done / self.epsilon_decay)
        if random.random() < eps_threshold:
            return random.randint(0, self.action_dim - 1)
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            #print("State shape before passing to CNN:", state.shape)
            with torch.no_grad():
                return torch.argmax(self.q_network(state)).item()

    def train(self):
        if len(self.memory) < self.start_training:
            return

        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        states_processed = self.q_network(states) # Evaluate all the states in the batch
        actions_index = actions.unsqueeze(1) # Transform the actions taken into (batch_size, 1)
        q_values_of_actions_taken = states_processed.gather(1, actions_index) # Get the values of the actions taken
        q_values = q_values_of_actions_taken.squeeze(1) # return to (batch_size, )

        # Compute target Q-values using Double DQN
        next_actions = self.q_network(next_states).argmax(1)
        next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values.detach()

        # Compute loss and optimize
        loss = F.mse_loss(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())


