from typing import Set, Tuple

import numpy as np

from models import MDP


class GridMoveMDP(MDP):
    """
    We move on a `grid` of size $(n \times n)$, where $(i, j) : 0 \leq i \leq n, 0 \leq j \leq n$ is a
    `location`.
    We always start from $(0,0)$ and we need to reach as fast as possible one of the final
    locations $(\cdot, n)$.

    On the grid we can:
        - `walk`: this is always possible, and we can move East, North, West, or South,
            moving to the next location adjacent to the starting one.
        - `bus`: this is always possible, and we can move East, North, West, or South
            two locations away in the chosen direction.
            However, with probability $b$, we need to wait for the bus.
            Waitinig means that we do not move.
        - trains are available only in locations $(i, j)$ where $\mod(i, 2) = 0$ and $\mod(j, 2) = 0$.
            If a train is available, you can get it to one of the next stations.
            However, with probability $c$ you need to wait. Waiting means not moving.
        - Any movement requires the same amount of time.
        - You can never move outside the grid.
    """

    def __init__(self, size: Tuple[int,int], b: float = .5,
                 c: float = .2, gamma: float = .9):
        """
        Init size and probabilities
        :param size: size of the grid
        :param b: probability of waiting the bus
        :param c: probability of waiting the train
        :param gamma: discount
        """
        super().__init__(gamma=gamma)
        self.X = size[0]
        self.Y = size[1]
        self.directions = {
            'N': lambda state, step: (state[0], state[1]+step),
            'W': lambda state, step: (state[0]-step, state[1]),
            'E': lambda state, step: (state[0]+step, state[1]),
            'S': lambda state, step: (state[0], state[1]-step),
        }
        self.b = b
        self.c = c

    def train_next_stations(self, state):
        stations = set()
        if state[0] % 2 == 0 and state[1] % 2 == 0:
            for x in [state[0], state[0]+2, state[0]-2]:
                for y in [state[1], state[1]+2, state[1]-2]:
                    if 0 <= x < self.X and 0 <= y < self.Y:
                        stations.add((x, y))
        return stations

    def valid_states(self, state, distance=1):
        valid = set()
        for _, f in self.directions.items():
            x, y = f(state, distance)
            if 0 <= x < self.X and 0 <= y < self.Y:
                valid.add((x, y))
        return valid

    def start(self):
        return {(0, 0)}

    def states(self) -> Set:
        S = set()
        for i in range(self.X):
            for j in range(self.Y):
                S.add((i, j))
        return S

    def actions(self, state) -> Set:
        options = set()
        # walking
        for direction, f in self.directions.items():
            new_x, new_y = f(state, 1)
            if 0 <= new_x <= self.X and 0 <= new_y <= self.Y:
                options.add("W{}".format(direction))
        # bus
        for direction, f in self.directions.items():
            new_x, new_y = f(state, 2)
            if 0 <= new_x <= self.X and 0 <= new_y <= self.Y:
                options.add("B{}".format(direction))
        # train
        if state[0] % 2 == 0 and state[1] % 2 == 0:
            for x, y in self.train_next_stations(state):
                options.add("T_{}_{}".format(x, y))
        return options

    def transition(self, start_state, action, end_state) -> float:
        if action[0] == 'W': # walk (you move with probability 1
            if end_state in self.valid_states(start_state, distance=1):
                return 1.
            else:
                return 0.
        elif action[0] == 'B': # you stay in start_state with probability b
            if start_state == end_state:
                return self.b
            elif end_state in self.valid_states(start_state, distance=2):
                return 1 - self.b
            else:
                return 0.
        elif action[0] == 'T': # you stay in start state with probability c
            if start_state == end_state:
                return self.c
            elif end_state in self.train_next_stations(start_state):
                return 1 - self.c
            else:
                return 0.
        else:
            return 0.

    def reward(self, start_state, action, end_state) -> float:
        return -1

    def successors(self, state, action):
        """
        Override successors to take into account only the possible states
        :param state: current state
        :param action: action
        :return: [(s_prime, p, r), ...]
        """
        data = []
        if action[0] == 'W':
            distance = 1
            direction = action[1]
            s_prime = self.directions[direction](state, distance)
            if s_prime in self.valid_states(state, distance=distance):
                data.append((s_prime, 1., -1))
        elif action[0] == 'B':
            distance = 2
            direction = action[1]
            s_prime = self.directions[direction](state, distance)
            if s_prime in self.valid_states(state, distance=distance):
                data.append((s_prime, 1 - self.b, -1))
                data.append((state, self.b, -1))
            else:
                data.append((state, 1., -1))
        elif action[0] == 'T':
            parts = action.split('_')
            s_prime = (int(parts[1]), int(parts[2]))
            if s_prime in self.valid_states(state):
                data.append((s_prime, 1 - self.c, -1))
                data.append((state, self.c, -1))
            else:
                data.append((state, 1., -1))
        return data

    def end(self, state) -> bool:
        return state[1] == self.Y - 1

