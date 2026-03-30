from gym_cards.envs.points import Point24Env
from gym_cards.envs.ezpoints import EZPointEnv
from gym_cards.envs.blackjack import BlackjackEnv
from gym_cards.envs.numberline import NumberLineEnv
from gymnasium.envs.registration import register

register(
    id='gym_cards/Blackjack-v0',
    entry_point='gym_cards.envs:BlackjackEnv',
    max_episode_steps=300,
)

register(
    id='gym_cards/Points24-v0',
    entry_point='gym_cards.envs:Point24Env',
    max_episode_steps=300,
)

register(
    id='gym_cards/EZPoints-v0',
    entry_point='gym_cards.envs:EZPointEnv',
    max_episode_steps=300,
)

register(
    id='gym_cards/NumberLine-v0',
    entry_point='gym_cards.envs:NumberLineEnv',
    max_episode_steps=300,
)