import os
from typing import Optional

import numpy as np
import random

import gymnasium as gym
from gymnasium import spaces
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO


class NumberLineEnv(gym.Env):
    """
A custom Gym environment simulating a number line.

Actions:
    - 0: Move left (decrease the current position by 1, if greater than 0).
    - 1: Move right (increase the current position by 1, if less than max_position).

Observation:
    - An RGB image representation of the current and goal positions on the number line.
    - The image has a shape of (300, 300, 3) and pixel values ranging from 0 to 255.

Termination:
    - The episode ends when the current position reaches the goal position.

Reward:
    - A reward of 1 is given if the current position is the same as the goal position.
    - A reward of -1 is given if the action does not move the current position closer to the goal position.
    - A reward of 0 is given otherwise.

Initialization Options:
    - max_position: The maximum value on the number line (default is 5).
      Both the start and goal positions are randomly initialized between 0 and max_position.

Special Notes:
    - The start and goal positions are re-randomized at the beginning of each episode.
    - If after 2 * max_position steps the target still is not meet, the episode is terminated and the environment is reset.
    - The environment ensures that the start and goal positions are not the same initially.
    - An observation is returned as an RGB image with the current and goal positions labeled.

"""

    def __init__(self, max_position=5):
        super(NumberLineEnv, self).__init__()
        self.max_position = max_position

        # Define action and observation space
        self.action_space = spaces.Discrete(2)  # 0: Move left, 1: Move right
        self.observation_space = spaces.Box(low=0, high=255, shape=(300, 300, 3), dtype=np.uint8)

        # Randomize start and goal
        self.start_position = random.randint(0, self.max_position)
        self.goal_position = random.randint(0, self.max_position)
        if self.start_position == self.goal_position:
            self.goal_position = (self.goal_position + 1) % self.max_position
        self.position = self.start_position
        self.steps_made = 0

    def step(self, action):
        if action==-1:
            return self._get_observation(), 0.0, False, False, {"Target": self.goal_position, "Current": self.position, "won": False}
        self.steps_made += 1
        prev_distance = abs(self.position - self.goal_position)
        if action == 0 and self.position > 0:
            self.position -= 1
        elif action == 1 and self.position < self.max_position:
            self.position += 1
        current_distance = abs(self.position - self.goal_position)

        done = False
        if current_distance == 0:
            reward = 1
            done = True
        elif current_distance > prev_distance or current_distance == prev_distance:
            reward = -1
        else:
            reward = 0

        won = reward > 0
        info = {"Target": self.goal_position, "Current": self.position, "won": won}

        observation = self._get_observation()
        if self.start_position == self.goal_position:
            self.goal_position = (self.goal_position + 1) % self.max_position
        truncate = False
        if not done and self.steps_made >= self.max_position * 2:
            truncate = True
        return observation, reward, done, truncate, info

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        random.seed(seed)
        # Randomize start and goal
        self.start_position = random.randint(0, self.max_position)
        self.goal_position = random.randint(0, self.max_position)
        if self.start_position == self.goal_position:
            self.goal_position = (self.goal_position + 1) % self.max_position
        self.position = self.start_position
        self.steps_made = 0
        info = {"Target": self.goal_position, "Current": self.position, "won": False}
        return self._get_observation(), info

    def _get_observation(self):
        # Draw the "Goal" and "Current" strings
        img = Image.new('RGB', (300,300), color="white")
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype('dejavu/DejaVuSans-Bold.ttf', 36)
        goal_text = f"Target: {self.goal_position}"
        current_text = f"Current: {self.position}"
        draw.text((30, 60), goal_text, fill='black',font=font)
        draw.text((30, 180), current_text, fill='black',font=font)
        return np.array(img)

    def close(self):
        pass
