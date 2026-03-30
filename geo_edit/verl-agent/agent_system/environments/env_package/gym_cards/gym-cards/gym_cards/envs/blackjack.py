import os
from typing import Optional

import numpy as np

import gymnasium as gym
from gymnasium import spaces
from PIL import Image, ImageDraw, ImageFont

image_cache = {}

def get_image(card_name):
    if card_name in image_cache:
        return image_cache[card_name]
    path = f"img/{card_name}.png"
    cwd = os.path.dirname(__file__)
    image = Image.open(os.path.join(cwd, path))
    image_cache[card_name] = image
    return image

def draw_card_with_info(np_random):
    card_value = draw_card(np_random)
    suit = np_random.choice(["C", "D", "H", "S"])
    if card_value == 1:
        face = 'A'
    elif card_value == 10:
        face = np_random.choice(["J", "Q", "K"])
    else:
        face = str(card_value)
    return card_value, face, suit

def draw_hand_with_info(np_random):
    return [draw_card_with_info(np_random) for _ in range(2)]


def cmp(a, b):
    r = float(a > b) - float(a < b)
    r = 0 if r < 0 else r
    return r


# 1 = Ace, 2-10 = Number cards, Jack/Queen/King = 10
deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]


def draw_card(np_random):

    return int(np_random.choice(deck))


def draw_hand(np_random):
    return [draw_card(np_random), draw_card(np_random)]


def usable_ace(hand_values):  # Does this hand have a usable ace?
    return int(1 in hand_values and sum(hand_values) + 10 <= 21)


def sum_hand(hand):  # Return current hand total
    hand_values = [card[0] for card in hand]
    if usable_ace(hand_values):
        return sum(hand_values) + 10
    return sum(hand_values)


def is_bust(hand):  # Is this hand a bust?
    values = [card[0] for card in hand]
    return sum_hand(hand) > 21


def score(hand):  # What is the score of this hand (0 if bust)
    return 0 if is_bust(hand) else sum_hand(hand)


def is_natural(hand):  # Is this hand a natural blackjack?
    return sorted(hand) == [1, 10]


class BlackjackEnv(gym.Env):
    """
    Blackjack is a card game where the goal is to beat the dealer by obtaining cards
    that sum to closer to 21 (without going over 21) than the dealers cards.

    ## Description
    The game starts with the dealer having one face up and one face down card,
    while the player has two face up cards. All cards are drawn from an infinite deck
    (i.e. with replacement).

    The card values are:
    - Face cards (Jack, Queen, King) have a point value of 10.
    - Aces can either count as 11 (called a 'usable ace') or 1.
    - Numerical cards (2-9) have a value equal to their number.

    The player has the sum of cards held. The player can request
    additional cards (hit) until they decide to stop (stick) or exceed 21 (bust,
    immediate loss).

    After the player sticks, the dealer reveals their facedown card, and draws cards
    until their sum is 17 or greater. If the dealer goes bust, the player wins.

    If neither the player nor the dealer busts, the outcome (win, lose, draw) is
    decided by whose sum is closer to 21.

    This environment corresponds to the version of the blackjack problem
    described in Example 5.1 in Reinforcement Learning: An Introduction
    by Sutton and Barto [<a href="#blackjack_ref">1</a>].

    ## Action Space
    The action shape is `(1,)` in the range `{0, 1}` indicating
    whether to stick or hit.

    - 0: Stick
    - 1: Hit

    ## Observation Space
    The observation is a pixel image of the current state of the game.
    spaces.Box(low=0, high=255, shape=(300, 300, 3), dtype=np.uint8)

    The observation is returned as `(int(), int(), int())`.

    ## Starting State
    The starting state is initialised in the following range.

    | Observation               | Min  | Max  |
    |---------------------------|------|------|
    | Player current sum        |  4   |  12  |
    | Dealer showing card value |  2   |  11  |
    | Usable Ace                |  0   |  1   |

    ## Rewards
    - win game: +1
    - lose game: -1
    - draw game: 0
    - win game with natural blackjack:
    +1.5 (if <a href="#nat">natural</a> is True)
    +1 (if <a href="#nat">natural</a> is False)

    ## Episode End
    The episode ends if the following happens:

    - Termination:
    1. The player hits and the sum of hand exceeds 21.
    2. The player sticks.

    An ace will always be counted as usable (11) unless it busts the player.

    ## Information

    No additional information is returned.

    ## Arguments

    ```python
    import gymnasium as gym
    gym.make('Blackjack-v1', natural=False, sab=False)
    ```

    <a id="nat"></a>`natural=False`: Whether to give an additional reward for
    starting with a natural blackjack, i.e. starting with an ace and ten (sum is 21).

    <a id="sab"></a>`sab=False`: Whether to follow the exact rules outlined in the book by
    Sutton and Barto. If `sab` is `True`, the keyword argument `natural` will be ignored.
    If the player achieves a natural blackjack and the dealer does not, the player
    will win (i.e. get a reward of +1). The reverse rule does not apply.
    If both the player and the dealer get a natural, it will be a draw (i.e. reward 0).

    ## References
    <a id="blackjack_ref"></a>[1] R. Sutton and A. Barto, “Reinforcement Learning:
    An Introduction” 2020. [Online]. Available: [http://www.incompleteideas.net/book/RLbook2020.pdf](http://www.incompleteideas.net/book/RLbook2020.pdf)

    ## Version History
    * v1: Fix the natural handling in Blackjack
    * v0: Initial version release
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 4,
    }

    def __init__(self, render_mode: Optional[str] = None, natural=False, sab=False, is_pixel: bool = True):
        self.is_pixel = is_pixel
        self.action_space = spaces.Discrete(2)
        if self.is_pixel:
            self.observation_space = spaces.Box(low=0, high=255, shape=(300, 300, 3), dtype=np.uint8)
        else:
            self.observation_space = spaces.Tuple(
            (spaces.Discrete(32), spaces.Discrete(11), spaces.Discrete(2)))

        # Flag to payout 1.5 on a "natural" blackjack win, like casino rules
        # Ref: http://www.bicyclecards.com/how-to-play/blackjack/
        self.natural = natural

        # Flag for full agreement with the (Sutton and Barto, 2018) definition. Overrides self.natural
        self.sab = sab

        self.render_mode = render_mode
        return


    def step(self, action):
        if action==-1:
            return self._get_obs(), 0.0, False, False, {"Dealer Card": self.dealer, "Player Card": self.player, 'won': False}
        assert self.action_space.contains(action)
        if action:  # hit: add a card to players hand and return
            self.player.append(draw_card_with_info(self.np_random))
            won = False
            if is_bust(self.player):
                terminated = True
                reward = 0.0
            else:
                terminated = False
                reward = 0.0
        else:  # stick: play out the dealers hand, and score
            terminated = True
            while sum_hand(self.dealer) < 17:
                self.dealer.append(draw_card_with_info(self.np_random))
            reward = cmp(score(self.player), score(self.dealer))
            if self.sab and is_natural(self.player) and not is_natural(self.dealer):
                # Player automatically wins. Rules consistent with S&B
                reward = 1.0
            elif (
                not self.sab
                and self.natural
                and is_natural(self.player)
                and reward == 1.0
            ):
                # Natural gives extra points, but doesn't autowin. Legacy implementation
                reward = 1.5
            
            won = reward > 0
        info = {"Dealer Card": self.dealer, "Player Card": self.player, 'won': won}

        return self._get_obs(), reward, terminated, False, info



    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.dealer = draw_hand_with_info(self.np_random)
        self.player = draw_hand_with_info(self.np_random)

        player_values = [card[0] for card in self.player]
        _, dealer_card_value, _ = (sum_hand(self.player), self.dealer[0][0], usable_ace(player_values))

        suits = ["C", "D", "H", "S"]
        self.dealer_top_card_suit = self.np_random.choice(suits)

        if dealer_card_value == 1:
            self.dealer_top_card_value_str = "A"
        elif dealer_card_value == 10:
            self.dealer_top_card_value_str = self.np_random.choice(["J", "Q", "K"])
        else:
            self.dealer_top_card_value_str = str(dealer_card_value)

        if self.render_mode == "human":
            self.render()
        info = {"Dealer Card": self.dealer, "Player Card": self.player, 'won': False}
        return self._get_obs(), info

    def _get_obs(self):
        if self.is_pixel:
            # Define image size and background color
            img_size = (300, 300)  # Adjust size as needed
            card_size = (70, 98)
            spacing = 4
            background_color = '#35654d'

            # Create a new image with the defined size and background color
            img = Image.new('RGB', img_size, color=background_color)
            draw = ImageDraw.Draw(img)
            font = ImageFont.truetype('dejavu/DejaVuSans-Bold.ttf', 16)
            # Load and paste dealer card image
            draw.text((5, 5), f"Dealer", fill='white', font=font)
            dealer_card = f"{self.dealer[0][2]}{self.dealer[0][1]}"
            dealer_card_img = get_image(dealer_card).resize(card_size)
            img.paste(dealer_card_img, (5, 25))  # Adjust position as needed
            back_card_img = get_image("card").resize(card_size)
            img.paste(back_card_img, (78, 25))

            # Load and paste player card images
            x_offset, y_offset = 5, 150  # Starting position for player cards
            draw.text((5, 130), f"Player", fill='white', font=font)
            for idx, (_, face, suit) in enumerate(self.player):
                card_name = f"{suit}{face}"
                card_img = get_image(card_name).resize(card_size)
                img.paste(card_img, (x_offset, y_offset))
                x_offset += card_img.width + spacing  # Adjust spacing and position as needed
                if idx == 4:
                    y_offset += card_img.height + spacing
                    x_offset = 5

            # Convert the PIL image to a NumPy array if needed
            image = np.array(img).astype(np.uint8)

            return image
        else:
            return (sum_hand(self.player), self.dealer[0][0], usable_ace(self.player))