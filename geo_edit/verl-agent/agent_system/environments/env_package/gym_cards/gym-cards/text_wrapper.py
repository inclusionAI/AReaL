import random
from typing import List

def info_to_text_obs(env_name, info):
    """
    This function directly parse the info from the gym_cards envs
    to customized text observation.

    gym_cards envs: https://github.com/RL4VLM/RL4VLM/tree/main/gym-cards

    please adjust text_obs accordingly as needed
    """
    if env_name == "gym_cards/NumberLine-v0":
        text_obs = f"Target number: {info['Target']}. Current number: {info['Current']}"
    elif env_name == "gym_cards/EZPoints-v0":
        """
        J, Q, and K count as 10, you can choose to increase the difficulty
        only telling the language model J, Q, K, and describe in the prompt that they count as 10
        """
        text_obs = f"Cards: {info['Cards']}. Numbers: {info['Numbers']}."
        current_formula = ''.join(str(element) for element in info['Formula'])
        text_obs = text_obs + f" Current formula: {current_formula}."
    elif env_name == "gym_cards/Points24-v0":
        text_obs = f"Cards: {info['Cards']}. Numbers: {info['Numbers']}."
        current_formula = ''.join(str(element) for element in info['Formula'])
        text_obs = text_obs + f" Current formula: {current_formula}."
    elif env_name == "gym_cards/Blackjack-v0":
        """
        the info for each card is represented in (card_value, face, suit)
        e.g., (10, 'J', 'H') represents the Jack of Hearts
        and we shall only include one card for the dealer
        since we cannot see all dealer's cards
        """
        ## note that the first [0] means we only select the first card
        ## the [1] means that we are using the card value, not the number
        ## feel free to change the [1] to [0] to reduce the complexity
        dealer_card = info['Dealer Card'][0][1]
        ## the [1] means that we are using the card value, not the number
        ## feel free to change the [1] to [0] to reduce the complexity
        player_cards = ', '.join(str(element[1]) for element in info['Player Card'])
        text_obs = f"Dealer's card: {dealer_card}. Player's cards: {player_cards}."
    else:
        raise NotImplementedError("Environment not implemented.")
    return text_obs
### Below are direclty copied from https://github.com/RL4VLM/RL4VLM/blob/main/VLM_PPO/a2c_ppo_acktr/rl_utils.py

# Define the function that processes the list of strings according to the specified rules
def text_projection(text_actions: List[str], env_name):
    output_indices = []
    if env_name == 'gym_cards/NumberLine-v0':
        action_list = ["-", "+"]
    elif env_name == 'gym_cards/Blackjack-v0':
        action_list = ["stand", "hit"]
    elif env_name == 'gym_cards/EZPoints-v0':
        action_list = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
                       "+", "*", "="]
    elif env_name == 'gym_cards/Points24-v0':
        action_list = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
                       "+", "-", "*", "/", "(", ")", "="]
    else:
        raise NotImplementedError("Action list not implemented for this env!")
    for string in text_actions:
        if not isinstance(string, str):
            # directly output a random action if the string is not a string
            output_indices.append(random.randint(0, len(action_list) - 1))
            continue
        string = string.lower()
        action_index = string.find('"action":')
        # Extract everything after "action":
        string = string[action_index:]
        contained_actions = []
        # For the 'gym_cards/Points24-v0' environment, handle '10' separately
        if 'points' in env_name.lower() and '10' in string:
            contained_actions.append('10')
            string = string.replace('10', '')  # Remove '10' to prevent it from being counted as '1'
        # Find all actions that are contained in the string
        for action in action_list:
            if action in string:
                contained_actions.append(action)
        # Remove duplicates by converting to a set and back to a list
        contained_actions = list(set(contained_actions))
        if len(contained_actions) == 1 and contained_actions[0] in action_list:
            # Only one keyword from action_list is in the string
            output_indices.append(action_list.index(contained_actions[0]))
        else:
            # The string contains none or multiple keywords, randomly select an index from action_list
            output_indices.append(random.randint(0, len(action_list) - 1))
    ## Please adjust the output dtypes accordingly
    return output_indices