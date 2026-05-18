"""Deep Q-Network agent for Lecture 5, Part 2.

This module is the reference implementation that the Part 2 notebook narrates.
It assembles three pieces:

* ``QNetwork`` -- the action-value approximator: a small multilayer perceptron
  that maps a state vector directly to one Q-value per action. It replaces the
  hand-crafted feature map x(s) of Lectures 3-4; the hidden layers learn the
  representation themselves.
* ``ReplayBuffer`` -- a fixed-capacity store of past transitions, sampled
  uniformly at random. It is the remedy for the correlation problem of Part 1:
  it decorrelates consecutive updates and lets each transition be reused.
* ``DQNAgent`` -- neural Q-learning wired to both fixes. The replay buffer
  supplies decorrelated minibatches; a periodically-synced target network keeps
  the bootstrap target stationary between syncs -- the remedy for the
  moving-target problem of Part 1.

The agent satisfies the course's agent protocol (``select_action``, ``update``,
``end_episode``) and so plugs into ``rlc.utils.training`` unchanged.
"""

from __future__ import annotations

import copy
import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """Action-value approximator: state vector in, one Q-value per action out.

    The neural counterpart of the linear approximator of Lectures 3-4. There,
    q(s, a) = w_a^T x(s): one weight vector per action on a fixed feature map
    x. Here the feature map is gone -- the network learns its own internal
    representation -- and a single forward pass returns the whole vector of
    action-values at once.
    """

    def __init__(self, n_state_features, n_actions, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_state_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, state):
        """Map a batch of state vectors to a batch of action-value vectors.

        Input shape (batch, n_state_features); output (batch, n_actions).
        Always called on a batch, even for a single state.
        """
        return self.net(state)


Transition = namedtuple("Transition",
                        ["state", "action", "reward", "next_state", "done"])


class ReplayBuffer:
    """Fixed-capacity store of transitions, sampled uniformly for learning.

    Backed by a deque with a maximum length: appending past capacity evicts
    the oldest transition automatically, so the buffer is always a sliding
    window over the most recent N transitions. sample() draws a minibatch
    uniformly at random, returned as batched tensors.
    """

    def __init__(self, capacity, *, seed=None):
        self._buffer = deque(maxlen=int(capacity))
        self._rng = random.Random(seed)

    def push(self, state, action, reward, next_state, done):
        """Store one transition. State arrays are copied defensively."""
        self._buffer.append(Transition(
            np.array(state, dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            bool(done),
        ))

    def sample(self, batch_size):
        """Draw a uniform random minibatch, returned as batched tensors.

        Returns (states, actions, rewards, next_states, dones) with shapes
        (B, d), (B,), (B,), (B, d), (B,). Actions are int64 (gather indices);
        dones are float32 so the term (1 - done) can multiply the bootstrap.
        """
        batch = self._rng.sample(self._buffer, batch_size)
        states      = torch.from_numpy(np.stack([t.state for t in batch]))
        actions     = torch.tensor([t.action for t in batch], dtype=torch.int64)
        rewards     = torch.tensor([t.reward for t in batch], dtype=torch.float32)
        next_states = torch.from_numpy(np.stack([t.next_state for t in batch]))
        dones       = torch.tensor([t.done for t in batch], dtype=torch.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Number of transitions currently stored."""
        return len(self._buffer)


class DQNAgent:
    """Deep Q-Network: neural Q-learning + experience replay + target network.

    Off-policy, epsilon-greedy. The Q-network maps a state directly to a vector
    of action-values (no hand-crafted feature map). Two components tame the
    instabilities of naive neural Q-learning: a ReplayBuffer decorrelates
    updates and reuses each transition; a target network makes the bootstrap
    target stationary between periodic syncs.
    """

    def __init__(self, n_state_features, n_actions, hidden_size=64,
                 lr=1e-3, gamma=0.99, epsilon_start=1.0, epsilon_min=0.05,
                 epsilon_decay=0.995, buffer_capacity=50_000, batch_size=64,
                 min_buffer_size=1000, target_sync_every=500, seed=None):
        self.n_actions = int(n_actions)
        self.gamma = float(gamma)
        self.epsilon_start = float(epsilon_start)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.epsilon = self.epsilon_start

        if seed is not None:
            torch.manual_seed(seed)
        self._rng = np.random.default_rng(seed)

        self.q_net = QNetwork(n_state_features, n_actions, hidden_size)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)

        self.target_net = copy.deepcopy(self.q_net)
        for p in self.target_net.parameters():
            p.requires_grad_(False)
        self.target_sync_every = int(target_sync_every)
        self._update_count = 0

        self.batch_size = int(batch_size)
        self.min_buffer_size = int(min_buffer_size)
        self.buffer = ReplayBuffer(buffer_capacity, seed=seed)

        # Diagnostics, appended per learning step.
        self.weight_norms = []
        self.grad_norms = []

    # --- action selection ---------------------------------------------------

    def q_values(self, state):
        state_t = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            return self.q_net(state_t).squeeze(0).numpy()

    def select_action(self, state, *, greedy=False):
        if not greedy and self._rng.random() < self.epsilon:
            return int(self._rng.integers(self.n_actions))
        q = self.q_values(state)
        max_value = q.max()
        candidates = np.flatnonzero(q == max_value)
        return int(self._rng.choice(candidates))

    # --- learning -----------------------------------------------------------

    def update(self, state, action, reward, next_state, terminated,
               *, next_action=None):
        self.buffer.push(state, action, reward, next_state, terminated)
        if len(self.buffer) < self.min_buffer_size:
            return

        states, actions, rewards, next_states, dones = \
            self.buffer.sample(self.batch_size)

        q_sa = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Bootstrap target, held constant (semi-gradient), computed with the
        # target network. (1 - dones) drops the bootstrap on terminal states.
        with torch.no_grad():
            q_next_max = self.target_net(next_states).max(dim=1).values
            target = rewards + self.gamma * (1.0 - dones) * q_next_max

        loss = ((q_sa - target) ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.sqrt(sum(p.grad.pow(2).sum() for p in self.q_net.parameters()))
        self.optimizer.step()

        self._update_count += 1
        if self._update_count % self.target_sync_every == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        self.grad_norms.append(float(grad_norm))
        with torch.no_grad():
            total = sum(p.pow(2).sum() for p in self.q_net.parameters())
            self.weight_norms.append(float(total.sqrt()))

    def end_episode(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # --- convenience views --------------------------------------------------

    def state_value(self, state):
        return float(self.q_values(state).max())

    def greedy_action(self, state):
        return self.select_action(state, greedy=True)