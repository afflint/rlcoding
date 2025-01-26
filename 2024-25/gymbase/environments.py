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


class BasicDuels(gym.Env):
    def __init__(self, starting_hp: int = 20, retreat_prob: float = 0.2, opponent_distr: Optional[np.ndarray] = None):
        super().__init__()

        self.ACTION_TO_MOVES = ["melee", "ranged", "spell", "retreat", "heal"]
        self.NUM_ACTIONS = len(self.ACTION_TO_MOVES)
        self.EFFECTIVENESS_TABLE = np.array([
            [-4, -6, -2,  0,  0],
            [-2, -4, -4,  0,  0],
            [-6, -2, -4,  0,  0],
            [-2, -6, -4,  0,  0],
            [-2, -2, -2, +1, +1],
        ])

        self._starting_hp = starting_hp
        self._retreat_prob = retreat_prob
        self._opponent_distr = opponent_distr
        
        # The agent can choose between melee, range, spell attack, retreat or heal
        self.action_space = gym.spaces.Discrete(self.NUM_ACTIONS)

        # The agent observers their hp and the opponent's
        self.observation_space = gym.spaces.Dict({
            "agent": gym.spaces.Discrete(starting_hp + 1),
            "opponent": gym.spaces.Discrete(starting_hp + 1)
        })

        if opponent_distr is not None and opponent_distr.size != self.NUM_ACTIONS:
            raise ValueError(f"The opponent distribution must be of length {self.NUM_ACTIONS}")

        if opponent_distr is not None and not np.isclose(np.sum(opponent_distr), 1):
            raise ValueError("The opponent distribution must sum to 1")

    def _get_obs(self):
        return {"agent": self._agent_location, "opponent": self._target_location}

    def _get_info(self, agent_action, opponent_action):
        agent_move = self.ACTION_TO_MOVES[agent_action]
        opponent_move = self.ACTION_TO_MOVES[opponent_action]
        return {"agent": agent_move, "opponent": opponent_move}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed = seed)
        np.random.seed(seed)

        self._agent_location = self._starting_hp
        self._target_location = self._starting_hp

        if self._opponent_distr is None:
            # Get an opponent that follows a distribution over the actions chosen at random
            random_vector = np.random.rand(self.NUM_ACTIONS)
            self._opponent_distr = random_vector / sum(random_vector)

        return self._get_obs(), {}

    def _compute_reward(self, retreat):
        agent_dead = self._agent_location <= 0
        opponent_dead = self._target_location <= 0

        if retreat:
            # If the agent fled, give a lower reward
            return 0.5

        if agent_dead == opponent_dead:
            # If both are dead or alive, no reward
            return 0
        
        if opponent_dead: return 1
        return -1

    def step(self, action: int):
        # Draw opponent action from their distribution
        opponent_action = np.random.choice(self.NUM_ACTIONS, p = self._opponent_distr)

        # Retrieve the effect of the two moves on the agent
        self._agent_location += self.EFFECTIVENESS_TABLE[action, opponent_action]
        self._agent_location = np.clip(self._agent_location, 0, self._starting_hp)

        # Retrieve the effect of the two moves on the opponent
        # note that the game's matrix must be inverted for the opponent
        self._target_location += self.EFFECTIVENESS_TABLE[opponent_action, action]
        self._target_location = np.clip(self._target_location, 0, self._starting_hp)

        # If either player is out of hit points, the match is over
        terminated = self._agent_location <= 0 or self._target_location <= 0

        agent_retreat = self.ACTION_TO_MOVES[action] == "retreat"
        opponent_retreat = self.ACTION_TO_MOVES[opponent_action] == "retreat"

        truncated = False
        if agent_retreat or opponent_retreat:
            # A retreat is not always successfull
            truncated = np.random.rand() <= self._retreat_prob

        reward = self._compute_reward(truncated and agent_retreat)
        info = self._get_info(action, opponent_action)
        observation = self._get_obs()

        return observation, reward, terminated, truncated, info


#Register the environment
gym.register(
    id="Duels-v0",
    entry_point=BasicDuels,
)



class TaxationEnv(gym.Env):
    """
    Gymnasium environment for the politician taxation game.
    """
    def __init__(self, years: int = 5, transitions: np.ndarray = None, rewards: np.ndarray = None):
        super(TaxationEnv, self).__init__()

        self.years = years

        # Action space: 0 = High Taxation, 1 = Moderate Taxation
        self.action_space = gym.spaces.Discrete(2)

        # Observation space:
        # - Economic growth: 0 = Low, 1 = High
        # - Years remaining: 0 to 5
        self.observation_space = gym.spaces.Tuple((
            gym.spaces.Discrete(2),  # Economic growth: 0 (Low), 1 (High)
            gym.spaces.Discrete(self.years + 1)   # Years remaining: 0 to years
        ))
        
        # Economy goes low except for high with moderate taxes
        if transitions is None:
            self.transitions = np.ones((self.observation_space[0].n, 
                                self.action_space.n), dtype=np.int64)
            self.transitions[0, 1] = 0
        else:
            self.transitions = transitions.astype(np.int64)

        if rewards is None:
            self.rewards = np.array([
                [15, 10],
                [8, 5]
            ])
        else:
            self.rewards = rewards

        # Initialize the environment
        self.reset()

    def step(self, action):
        """
        Execute one step in the environment.
        """
        # Get current state
        economic_growth, years_left = self.state

        # Calculate the reward based on the action and economic growth
        reward = self.rewards[int(economic_growth), int(action)]
        next_growth = self.transitions[int(economic_growth), int(action)]

        # Update years remaining
        years_left -= 1

        # Update the state
        self.state = (next_growth, years_left)

        # Check if the episode is done
        done = years_left == 0

        # Additional info (optional, can be used for debugging)
        info = {}

        return self.state, reward, done, False, info

    def reset(self, seed=None, options=None):
        """
        Reset the environment to the initial state.
        """
        super().reset(seed=seed)

        # Initial state: High Economic Growth and 5 years remaining
        self.state = (0, self.years)
        return self.state, {}

    def render(self):
        """
        Render the current state of the environment.
        """
        economic_growth_str = "High" if self.state[0] == 0 else "Low"
        print(f"📈 Economic Growth: {economic_growth_str}, 📅 Years Remaining: {self.state[1]}")

    def close(self):
        """
        Perform any cleanup when closing the environment.
        """
        pass

#Register the environment
gym.register(
    id="TaxationGame-v0",
    entry_point=lambda years, transitions, rewards: TaxationEnv(years, transitions, rewards),
)
