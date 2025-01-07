import numpy as np
import gymnasium as gym
from typing import Tuple, Optional


class Conservatorio(gym.Env):

    def __init__(self, distance: int, beta: float = .5, 
                 walk_cost: int = 6, bus_cost: int = 4, 
                 bus_wait: int = 12, metro_cost: int = 3):
        super().__init__()
        self.N = distance
        self._agent_location = 0
        self._target_location = self.N
        self.observation_space = gym.spaces.Discrete(self.N)
        self.a2id = {'walk': 0, 'bus': 1, 'metro': 2}
        self.id2a = dict([(v, k) for k, v in self.a2id.items()])
        self.action_space = gym.spaces.Discrete(len(self.a2id))
        self.beta = beta
        self.w, self.b_1, self.b_2, self.x = walk_cost, bus_cost, bus_wait, metro_cost
    
    def _move(self, state: int, action: int) -> Tuple[int, int]:
        """Transition function

        Args:
            state (int): starting location
            action (int): how to move

        Returns:
            Tuple[int, int]: arriving location, cost in minutes
        """
        if 'walk' == self.id2a[action]:
            return state + 1, self.w
        elif 'bus' == self.id2a[action]:
            destination = state + 2
            if np.random.uniform() <= self.beta:
                return destination, self.b_1
            else:
                return destination, self.b_2
        else:
            return state + 4, self.x 
    
    def _get_position(self):
        return {'agent': self._agent_location, 'target': self._target_location}
    
    def _get_distance(self):
        pos = self._get_position()
        return {'distance': pos['target'] - pos['agent']}
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._agent_location = 0
        return self._agent_location, self._get_distance()
    
    def step(self, action: int):
        s_prime, cost = self._move(self._agent_location, action)
        reward = -cost 
        terminated = False
        if s_prime == self.N:
            terminated = True
        elif s_prime > self.N:
            terminated = True
            reward = 2 * self.b_2
            self._agent_location = self.N
        self._agent_location = s_prime
        info = self._get_distance()
        return s_prime, reward, terminated, False, info

#Register the environment
gym.register(
    id="Conservatorio-v0",
    entry_point=lambda distance, beta, walk_cost, bus_cost, bus_wait, metro_cost: Conservatorio(distance, 
                                                                                        beta, 
                                                                                        walk_cost, 
                                                                                        bus_cost,
                                                                                        bus_wait, 
                                                                                        metro_cost),
    max_episode_steps=300,
)

