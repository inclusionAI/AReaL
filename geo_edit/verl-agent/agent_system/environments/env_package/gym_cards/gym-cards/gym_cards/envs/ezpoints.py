import os
from typing import Optional

import numpy as np
import random

import gymnasium as gym
from gymnasium import spaces
from PIL import Image, ImageDraw, ImageFont


def get_image(card_name):
    path = f"img/{card_name}.png"
    cwd = os.path.dirname(__file__)
    image = Image.open(os.path.join(cwd, path))
    return image

# Constants for actions
NUMBER_ACTIONS_TEN = list(range(1, 11))
OPERATOR_ACTIONS = ['+', '*', '=']

class EZPointEnv(gym.Env):
    """
    A custom Gym environment for an easier version of solving the "24 Game".

    Actions:
        0: 1
        1: 2
        2: 3
        3: 4
        4: 5
        5: 6
        6: 7
        7: 8
        8: 9
        9: 10
        10: '+'
        11: '*'
        12: '='

    Termination:
        - If the formula length exceeds 5.
        - If '=' action is taken, the formula is evaluated.

    Reward:
        - 10 if the formula evaluates to the target_points.
        - 0 if an invalid action is taken.
        - 0 otherwise

    """
    def __init__(self, target_points=12):
        self.target_points = target_points
        self.set_action_space()
        self.canvas_width, self.canvas_height = 300, 300
        self.observation_space = spaces.Box(low=0, high=255, shape=(300, 300, 3), dtype=np.uint8)

    def set_action_space(self):
        numbers = NUMBER_ACTIONS_TEN
        self.allowed_numbers = numbers
        self.action_space = spaces.Discrete(len(numbers) + len(OPERATOR_ACTIONS))

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        random.seed(seed)
        self.cards_num, self.cards = self._generate_cards()
        self.card_imgs = []
        self.card_width = int(self.canvas_width / len(self.cards) * 0.9)  # Adjust as needed
        self.card_height = int(self.card_width * 7/5)  # Assuming a 5:7 card ratio; adjust if different
        for i, card in enumerate(self.cards):
            pil_img = get_image(card).resize((self.card_width, self.card_height))  # Resize the card
            self.card_imgs.append(pil_img)
        self.formula = []
        self.used_cards = []
        info = {"Cards": self.cards, "Numbers": self.cards_num, "Formula": self.formula, "won": False}
        return self._get_observation(), info

    def step(self, action):
        if action==-1:
            return self._get_observation(), 0, False, False, {"Cards": self.cards, "Numbers": self.cards_num, "Formula": self.formula, "won": False}
        terminated, reward, info = False, 0, {}
        chosen_action = self.allowed_numbers[action] if action < len(self.allowed_numbers) else OPERATOR_ACTIONS[action - len(self.allowed_numbers)]

        ## Terminate first if the formula is too long.
        if len(self.formula) > 5:
            return self._terminate_step(0, 'time_limit_reached', is_truncated=True)

        if not self._is_valid_action(chosen_action):
            ## Add a space to the formula, to make sure the formula length increases.
            return self._get_observation(), 0, False, False, {"Cards": self.cards, "Numbers": self.cards_num, "Formula": self.formula, "won": False}
        elif chosen_action in self.allowed_numbers:
            self.used_cards.append(chosen_action)

        if chosen_action == '=':
            return self._evaluate_formula()

        self.formula.append(chosen_action)
        info = {"Cards": self.cards, "Numbers": self.cards_num, "Formula": self.formula, "won": False}
        return self._get_observation(), reward, terminated, False, info

    def _generate_cards(self):
        # Generate the first card
        cards_num = [random.randint(2, 13) for _ in range(1)]
        suits = ["H", "S", "D", "C"]
        cards_suit = [random.choice(suits) for _ in range(1)]
        cards = [y + self._card_num_to_str(x) for x, y in zip(cards_num, cards_suit)]
        # Set face cards to 10
        cards_num = [min(x, 10) for x in cards_num]
        # Check whether the generated card is divisable by 12
        divisable_list = [1, 2, 3, 4, 6]
        if cards_num[0] in divisable_list and random.random() <= 0.5:
            # If the number is divisable by 2, 3, 4, 6, then we generate another number to 12 by multiplication with 50% chance
            cards_num.append(12 // cards_num[0])
        else:
            cards_num.append(12 - cards_num[0])

        cards.append(random.choice(suits)+self._card_num_to_str(cards_num[1]))
        if random.random() <= 0.5:
            # randomly swap the order of the two cards
            cards_num = cards_num[::-1]
            cards = cards[::-1]
        return cards_num, cards

    def _card_num_to_str(self, num):
        face_cards = {1: 'A', 10: 'T', 11: 'J', 12: 'Q', 13: 'K'}
        return face_cards.get(num, str(num))



    def _is_valid_action(self,action):
        if action not in self.allowed_numbers:
            # We don't check for operators
            return True
        else:
            new_used_cards = self.used_cards + [action]
            is_valid = not any(new_used_cards.count(x) > self.cards_num.count(x) for x in new_used_cards)
            return is_valid


    def _evaluate_formula(self):
        try:
            formula_str = ''.join(map(str, self.formula))
            reward = 10 if eval(formula_str) == self.target_points else 0
        except Exception:
            # The formula is invalid
            reward = 0
        finally:
            if len(self.used_cards) != 2:
                # Not all cards are used.
                reward = 0
        won = reward > 0
        info = {"Cards": self.cards, "Numbers": self.cards_num, "Formula": self.formula, "won": won}
        return self._get_observation(), reward, True, False, info

    def _terminate_step(self, reward, info_key, is_truncated=False):
        return self._get_observation(), reward, not is_truncated, is_truncated, {"Cards": self.cards, "Numbers": self.cards_num, "Formula": self.formula, "won": False}

    def _get_observation(self):
        # Create a blank white canvas
        canvas = Image.new('RGB', (self.canvas_width, self.canvas_height), '#35654d')

        # Paste each card onto the canvas
        for i, pil_img in enumerate(self.card_imgs):
            # Calculate position for pasting
            x_offset = 5+ int(i * pil_img.width * 1.1)  # adjust this multiplier (1.1) for spacing
            y_offset = int((self.canvas_height - pil_img.height) / 2)  # center vertically
            canvas.paste(pil_img, (x_offset, y_offset))

        # Draw formula onto the canvas
        draw = ImageDraw.Draw(canvas)
        text_formula = 'Formula: '
        text = f'{" ".join(map(str, self.formula))}'
        text_formula = text_formula + text
        font = ImageFont.truetype('dejavu/DejaVuSans.ttf', 20)
        draw.text((10, self.canvas_height*0.86), text_formula, fill="white", font=font)  # adjust position and other properties as needed
        # Convert PIL image to numpy array if required
        image_array = np.array(canvas)

        return image_array
