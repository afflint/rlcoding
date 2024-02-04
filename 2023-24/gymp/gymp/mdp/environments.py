"""Show how to create a custom environment (see https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/)"""
import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register



class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self, render_mode=None, size=5) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.size = size #size of the square grid
        self.window_size = 512 #for pygame rendering
        
        #Observations provide info about the agent and target locations
        self.observation_space = spaces.Dict({
            'agent': spaces.Box(0, size - 1, shape=(2,), dtype=int),
            'target': spaces.Box(0, size - 1, shape=(2,), dtype=int),
        })
        
        #Four discrete actions for right, up, left, down
        self.action_space = spaces.Discrete(4)
        
        #Map between actions and actual directions in the 2d space
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1])
        }
        self.window = None
        self.clock = None
    
    def _get_obs(self):
        """Returns the observation from the environment state"""
        return {'agent': self._agent_location, 'target': self._target_location}
    
    def _get_info(self):
        """Returns the manhattan ditance from agent location to target location"""
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }
    
    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

#Register the environment
register(
    id="gym_examples/GridWorld-v0",
    entry_point=lambda: GridWorldEnv(),
    max_episode_steps=300,
)

class ForestExampleEnv(gym.Env):
    """The environment represents a forset where, from each location (state),
    we have paths leading to another location.
    The possible directions are always 4: North, South, West, East
    Structure
    A south H
    A east C
    A west B
    B east A
    B west D
    C west A
    C east E
    D north B
    D south F
    E south C
    E north G
    F west D
    G south E
    H west A
    """

    # Actions: 0=north, 1=south, 2=west, 3=east.
    action_space = gym.spaces.Discrete(4)
    # States from A to H as from 0 to 7.
    observation_space = gym.spaces.Discrete(8)

    def __init__(self):
        self.state_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        self.s = lambda x: self.state_names.index(x)
        # State available actions
        self.available_actions = np.array([
            [0, 1, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
        ])
        # map state: {action: next_state}
        self.direction = {
            0: {2: self.s('B'), 3: self.s('C'), 1: self.s('H')},
            1: {3: self.s('A'), 2: self.s('D')},
            2: {2: self.s('A'), 3: self.s('E')},
            3: {0: self.s('B'), 1: self.s('F')},
            4: {0: self.s('G'), 1: self.s('C')},
            5: {2: self.s('D')},
            6: {1: self.s('E')},
            7: {2: self.s('A')},
        }
        self.observation_space = spaces.Discrete(8)
        
    def _rewards(self, state):
        if state == self.s('H'):
            return 1
        elif state in [self.s('F'), self.s('G')]:
            return -1
        else:
            return 0

    def _get_obs(self):
        """Returns the observation from the environment state"""
        return self._agent_location
    
    def _get_info(self):
        """Returns the available actions from agent state"""
        return {'actions': self.unwrapped.available_actions[self._agent_location]}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        # Choose the agent's to start always on location A
        self._agent_location = 0
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        assert self.action_space.contains(action)
        assert self.unwrapped.available_actions[self._agent_location, action] == 1 # check if the action is possible
        next_state = self.direction[self._agent_location][action]
        self._agent_location = next_state
        info = self._get_info()
        terminal = next_state == self.s('H')
        return self._get_obs(), self._rewards(next_state), terminal, False, info


#Register the environment
register(
    id="ForestExampleEnv-v0",
    entry_point=lambda: ForestExampleEnv(),
    max_episode_steps=300,
)