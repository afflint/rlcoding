import networkx as nx
from typing import Tuple, Callable, List
from abc import ABC, abstractmethod
import utils


class DDP(ABC):

    @property
    def start_state(self):
        """
        Give the start state
        :return: starting state
        """
        return None

    @abstractmethod
    def end(self, state):
        """
        Check for the end state
        :param state: current state
        :return: boolean
        """
        pass

    @abstractmethod
    def actions(self, state):
        """
        Possible actions from state
        :param state: state
        :return: list of possible actions
        """
        pass

    @abstractmethod
    def successor(self, state, action):
        """
        Given current state and action chosen returns
        the next state
        :param state:
        :param action:
        :return: next state
        """
        pass

    @staticmethod
    def cost(state, action):
        """
        Provides the cost of choosing the action from the current state
        :param state: current state
        :param action: action taken
        :return: cost
        """
        pass

    @abstractmethod
    def options(self, state):
        """
        Combines the previous methods to return a list
        of options given the current state
        :param state: current state
        :return: [action, next_state, cost]
        """
        pass


class FibonacciWalk(DDP):
    """
    We move forward along a line crossing $N$ points, starting from $0$ and ending in $N$.
    Possible movements are `walk` or `jump`, each with a specific cost.
    In particular, `walk` costs always $1$ and `jump` costs always $2$.
    By walking, we move from the point $i$ to the next point $i+1$.
    By jumping, we jump from the current point $i$ to the next point $j : j > i$
    that is in the Fibonacci series from $0$ to $N$.
    If such a point does not exists, we move to the next point $i+1$.
    We need to find the minimum cost to reach $N$.
    """

    def __init__(self, n: int):
        self.n = n
        self.fibonacci = list(utils.fibonacci(n))

    @property
    def start_state(self):
        return 0

    def end(self, state: int):
        return state == self.n

    def actions(self, state: int):
        if self.end(state):
            return []
        else:
            return ['W', 'J']

    @staticmethod
    def cost(state: int, action: str):
        if action == 'W':
            return 1
        else:
            return 2

    def successor(self, state: int, action: str):
        if self.end(state):
            return None
        else:
            if action == 'W':
                return state + 1
            else:
                try:
                    return min([x for x in self.fibonacci if state < x <= self.n])
                except ValueError:
                    return state + 1

    def options(self, state: int):
        """
        Combines successor and cost to return a list of triples of the form
        (action, next_stage, cost)
        :param state: current stage
        :return: list of (action, next_stage, cost)
        """
        result = []
        for action in self.actions(state):
            next_state = self.successor(state, action)
            result.append((action, next_state, FibonacciWalk.cost(state, action)))
        return result


class FieldMovement(DDP):
    """
    We move in a rectangular (n x m) grid, starting from tile (1,1) and aiming
    at reaching tile (n,m).
    state: one of the grid tiles
    actions: at each tile we can either move forward by moving north (N), east (E) or north-east (NE)
    cost: moving from state (i, j) costs 2i+j to go N, i+2*j to go E, 3*i*j to go NE
    end state: (n,m)
    """

    def __init__(self, n: int, m: int):
        self.x_size, self.y_size = n, m

    @property
    def start_state(self):
        return 1, 1

    def end(self, state: Tuple[int,int]):
        return state[0] == self.x_size and state[1] == self.y_size

    def actions(self, state: Tuple[int,int]) -> List[str]:
        """
        Returns only legal actions
        :param state: current state
        :return: actions that do not violate the grid
        """
        return [a for a in ['N', 'E', 'NE'] if self.successor(state, a) is not None]

    def successor(self, state: Tuple[int,int], action: str):
        x, y = state[0], state[1]
        if action == 'N' and y+1 <= self.y_size:
            return x, y + 1
        elif action == 'E' and x+1 <= self.x_size:
            return x + 1, y
        elif action == 'NE' and x+1 <= self.x_size and y+1 <= self.y_size:
            return x + 1, y + 1
        else:
            return None

    def cost(self, state: Tuple[int, int], action: str):
        i, j = state
        if action == 'NE':
            return 3*i*j
        elif action == 'E':
            return 2*j + i
        else:
            return 2*i + j

    def options(self, state: Tuple[int,int]):
        """
        Combines successor and cost to return a list of triples of the form
        (action, next_stage, cost)
        :param state: current stage
        :return: list of (action, next_stage, cost)
        """
        result = []
        for action in self.actions(state):
            next_state = self.successor(state, action)
            result.append((action, next_state, self.cost(state, action)))
        return result


