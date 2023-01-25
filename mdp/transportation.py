from typing import Set

from models import MDP


class TransportationMDP(MDP):
    """
    You need to travel from a place `1` to a place `N`, moving always forward.
    At any place `i` with $0 \leq i \leq N$ you can choose one of the following transportation options:
    - `walk`: when you walk you always spend 2 minutes to go to $i + 1$;
    - `bus`: if you take a bus you spend 1 minute to go to $i + 2$,
        but with probability $\alpha$ you need to wait the bus.
        There's not bus traveling to $j$ if $j > N$;
    - `train` if you take the train you spend 1 minute to go to $2i$,
        but with probability $\beta$ you need to wait the train.
        There's not train traveling to $j$ if $j > N$;
    Your goal is to reach $N$ as fast as possible.
    """
    def __init__(self, n: int, alpha: float, beta: float, gamma: float,
                 w: int = -2, b: int = -1, t: int = -1):
        super().__init__(gamma)
        self.alpha = alpha
        self.beta = beta
        self.N = n
        self.w, self.b, self.t = w, b, t

    def start(self):
        return {1}

    def states(self) -> Set:
        return set(range(1, self.N + 1))

    def actions(self, state) -> Set:
        a = set()
        if state + 1 <= self.N:
            a.add('walk')
        if state + 2 <= self.N:
            a.add('bus')
        if state * 2 <= self.N:
            a.add('train')
        return a

    def transition(self, start_state, action, end_state) -> float:
        p = 0.
        valid_actions = self.actions(start_state)
        if not self.end(end_state) and action in valid_actions:
            if action == 'walk':
                if end_state == start_state + 1:
                    p = 1.
            elif action == 'bus':
                if start_state + 2 == end_state:
                    p = 1 - self.alpha
                elif end_state == start_state:
                    p = self.alpha
            elif action == 'train':
                if start_state * 2 == end_state:
                    p = 1 - self.beta
                elif end_state == start_state:
                    p = self.beta
        return p

    def reward(self, start_state, action, end_state) -> float:
        r = 0
        valid_actions = self.actions(start_state)
        if not self.end(end_state) and action in valid_actions:
            if action == 'walk':
                r = self.w
            elif action == 'bus':
                r = self.b
            elif action == 'train':
                r = self.t
        return r

    def successors(self, state, action):
        """
        Overrides the MDP successor function to avoid
        checking on all the possible states
        :param state: starting state
        :param action: action
        :return: [(s_prime, p, r), ...]
        """
        data = []
        if action == 'walk' and state + 1 <= self.N:
            data.append((state + 1, 1., self.w))
        elif action == 'bus' and state + 2 <= self.N:
            data.append((state + 2, 1 - self.alpha, self.b))
            data.append((state, self.alpha, self.b))
        elif action == 'train' and state * 2 <= self.N:
            data.append((state * 2, 1 - self.beta, self.t))
            data.append((state, self.beta, self.t))
        return data

    def end(self, state) -> bool:
        return state == self.N
