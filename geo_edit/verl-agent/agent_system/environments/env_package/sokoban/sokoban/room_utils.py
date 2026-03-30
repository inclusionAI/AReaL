import random
import numpy as np
import marshal
import copy
from collections import deque

import matplotlib.pyplot as plt
import matplotlib.animation as animation

def get_shortest_action_path(room_fixed, room_state, MAX_DEPTH=100):
        """
        Get the shortest action path to push all boxes to the target spots.
        Use BFS to find the shortest path.
        NOTE currently only support one player, only one shortest solution
        =========================================================
        Parameters:
            room_state (np.ndarray): the state of the room
                - 0: wall
                - 1: empty space
                - 2: box target
                - 3: box on target
                - 4: box not on target
                - 5: player
            room_fixed (np.ndarray): the fixed part of the room
                - 0: wall
                - 1: empty space
                - 2: box target
            MAX_DEPTH (int): the maximum depth of the search
        =========================================================
        Returns:
            action_sequence (list): the action sequence to push all boxes to the target spots
        """
        
        # BFS queue stores (room_state, path)
        queue = deque([(copy.deepcopy(room_state), [])])
        explored_states = set()
        
        # Possible moves: up, down, left, right
        moves = [(-1,0), (1,0), (0,-1), (0,1)]
        actions = [1, 2, 3, 4] # Corresponding action numbers
        
        while queue:
            room_state, path = queue.popleft()
            if len(path) > MAX_DEPTH:
                return [] # No solution found

            # reduce the search space by checking if the state has been explored
            state_tohash = marshal.dumps(room_state)
            if state_tohash in explored_states:
                continue
            explored_states.add(state_tohash)
            

            # get information of the room
            player_pos = tuple(np.argwhere(room_state == 5)[0])
            boxes_on_target = set(map(tuple, np.argwhere((room_state == 3))))
            boxes_not_on_target = set(map(tuple, np.argwhere((room_state == 4))))
            boxes = boxes_on_target | boxes_not_on_target


            # Check if all boxes are on targets
            if not boxes_not_on_target:
                return path
                
            # Try each direction
            for move, action in zip(moves, actions):
                new_room_state = copy.deepcopy(room_state)
                new_player_pos = (player_pos[0] + move[0], player_pos[1] + move[1])
                
                # Check is new player position is wall or out of bound
                if new_player_pos[0] < 0 or new_player_pos[0] >= room_fixed.shape[0] \
                    or new_player_pos[1] < 0 or new_player_pos[1] >= room_fixed.shape[1] \
                    or room_fixed[new_player_pos] == 0:
                    continue
                    
                # If there's a box, check if we can push it
                if new_player_pos in boxes:
                    box_pos = new_player_pos # the original box position
                    new_box_pos = (new_player_pos[0] + move[0], new_player_pos[1] + move[1])
                    
                    # Can't push if hitting wall or another box or out of bound
                    if room_fixed[new_box_pos] == 0 or new_box_pos in boxes \
                        or new_box_pos[0] < 0 or new_box_pos[0] >= room_fixed.shape[0] \
                        or new_box_pos[1] < 0 or new_box_pos[1] >= room_fixed.shape[1]:
                        continue
                        
                    # move the box
                    
                    new_room_state[box_pos] = room_fixed[box_pos]
                    if room_fixed[new_box_pos] == 2:
                        new_room_state[new_box_pos] = 3
                    else:
                        new_room_state[new_box_pos] = 4
                
                # player moves
                new_room_state[player_pos] = room_fixed[player_pos]
                new_room_state[new_player_pos] = 5
                queue.append((new_room_state, path + [action]))
                        
        return [] # No solution found

# def plot_animation(imgs):
#     fig, ax = plt.subplots()
#     im = ax.imshow(imgs[0])
#     def init():
#         im.set_data(imgs[0])
#         return [im]
#     def update(i):
#         im.set_data(imgs[i])
#         return [im]
#     ani = animation.FuncAnimation(fig, update, frames=len(imgs), init_func=init, blit=True)
#     return ani

def plot_animation(imgs):
    height, width = imgs[0].shape[:2]
    fig = plt.figure(figsize=(width/100, height/100), dpi=500)
    
    ax = fig.add_axes([0, 0, 1, 1])
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    
    im = ax.imshow(imgs[0])
    def init():
        im.set_data(imgs[0])
        return [im]
    def update(i):
        im.set_data(imgs[i])
        return [im]
    ani = animation.FuncAnimation(fig, update, frames=len(imgs), init_func=init, blit=True)
    return ani

def solve_sokoban(env, saved_animation_path):
    """
    Solve the given sokoban environment and save the animation
    """
    actions = get_shortest_action_path(env.room_fixed, env.room_state)
    print(f"Found {len(actions)} actions: {actions}")
    imgs = []
    img_before_action = env.render('rgb_array')
    imgs.append(img_before_action)
    for action in actions:
        env.step(action)
        img_after_action = env.render('rgb_array')
        imgs.append(img_after_action)
    ani = plot_animation(imgs)
    ani.save(saved_animation_path)



        





def add_random_player_movement(room_state, room_structure, move_probability=0.5, continue_probability=0.5, max_steps=3):
    """
    Randomly move the player after reverse_playing to make the level more challenging, also fix the problem that in generated map, the player is always adjacent to the box
    
    Parameters:
        room_state (np.ndarray): Current state of the room
        room_structure (np.ndarray): Fixed structure of the room
        move_probability (float): Probability of moving the player at all (0.0-1.0)
        continue_probability (float): Probability of continuing to move after each step (0.0-1.0)
        max_steps (int): Maximum number of steps the player can move (1-3)
    
    Returns:
        np.ndarray: Updated room state with randomly moved player
    """
    # Check if we should move the player at all
    if random.random() > move_probability:
        return room_state
    
    # Find player position
    player_pos = np.where(room_state == 5)
    player_pos = np.array([player_pos[0][0], player_pos[1][0]])
    
    # Keep track of previous positions to avoid moving back
    previous_positions = [tuple(player_pos)]
    
    # Make 1-3 random moves
    steps_taken = 0
    while steps_taken < max_steps:
        # Get all valid moves (can't move into walls or boxes)
        valid_moves = []
        for action in range(4):  # 0: up, 1: down, 2: left, 3: right
            change = CHANGE_COORDINATES[action]
            next_pos = player_pos + change
            
            # Check if next position is valid (empty space or target) and not a previous position
            if (room_state[next_pos[0], next_pos[1]] in [1, 2] and 
                tuple(next_pos) not in previous_positions):
                valid_moves.append((action, next_pos))
        
        # If no valid moves, break
        if not valid_moves:
            break
        
        # Choose a random valid move
        chosen_action, next_pos = random.choice(valid_moves)
        # print(f"player_pos: {player_pos}, next_pos: {next_pos}")
        
        # Move player
        room_state[player_pos[0], player_pos[1]] = room_structure[player_pos[0], player_pos[1]]
        room_state[next_pos[0], next_pos[1]] = 5
        
        # Update player position and track previous position
        player_pos = next_pos
        previous_positions.append(tuple(player_pos))
        
        steps_taken += 1
        
        # Decide whether to continue moving
        if steps_taken >= max_steps or random.random() > continue_probability:
            break
    
    return room_state



"""
Following code is adapted from the nicely written gym_sokoban repo
"""

def generate_room(dim=(13, 13), p_change_directions=0.35, num_steps=25, num_boxes=3, tries=4, second_player=False, search_depth=100):
    """
    Generates a Sokoban room, represented by an integer matrix. The elements are encoded as follows:
    wall = 0
    empty space = 1
    box target = 2
    box not on target = 3
    box on target = 4
    player = 5

    :param dim:
    :param p_change_directions:
    :param num_steps:
    :param num_boxes:
    :param tries:
    :param second_player:
    :return: Numpy 2d Array, box mapping, action sequence
    """
    room_state = np.zeros(shape=dim)
    room_structure = np.zeros(shape=dim)

    # Some times rooms with a score == 0 are the only possibility.
    # In these case, we try another model.
    for t in range(tries):
        room = room_topology_generation(dim, p_change_directions, num_steps)
        room = place_boxes_and_player(room, num_boxes=num_boxes, second_player=second_player)

        # Room fixed represents all not movable parts of the room
        room_structure = np.copy(room)
        room_structure[room_structure == 5] = 1

        # Room structure represents the current state of the room including movable parts
        room_state = room.copy()
        room_state[room_state == 2] = 4

        room_state, box_mapping, action_sequence = reverse_playing(room_state, room_structure, search_depth)
        room_state[room_state == 3] = 4

        if box_displacement_score(box_mapping) > 0:
            break

    if box_displacement_score(box_mapping) == 0:
        raise RuntimeWarning('Generated Model with score == 0')

    # Add random player movement after reverse_playing
    if box_displacement_score(box_mapping) == 1:
        move_probability = 0.8
    else:
        move_probability = 0.5
    room_state = add_random_player_movement(
        room_state, 
        room_structure,
        move_probability=move_probability,       # 50% chance the player will move
        continue_probability=0.5,   # 50% chance to continue moving after each step
        max_steps=3                 # Maximum of 3 steps
    )

    return room_structure, room_state, box_mapping, action_sequence


def room_topology_generation(dim=(10, 10), p_change_directions=0.35, num_steps=15):
    """
    Generate a room topology, which consits of empty floors and walls.

    :param dim:
    :param p_change_directions:
    :param num_steps:
    :return:
    """
    dim_x, dim_y = dim

    # The ones in the mask represent all fields which will be set to floors
    # during the random walk. The centered one will be placed over the current
    # position of the walk.
    masks = [
        [
            [0, 0, 0],
            [1, 1, 1],
            [0, 0, 0]
        ],
        [
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0]
        ],
        [
            [0, 0, 0],
            [1, 1, 0],
            [0, 1, 0]
        ],
        [
            [0, 0, 0],
            [1, 1, 0],
            [1, 1, 0]
        ],
        [
            [0, 0, 0],
            [0, 1, 1],
            [0, 1, 0]
        ]
    ]

    # Possible directions during the walk
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    direction = random.sample(directions, 1)[0]

    # Starting position of random walk
    position = np.array([
        random.randint(1, dim_x - 1),
        random.randint(1, dim_y - 1)]
    )

    level = np.zeros(dim, dtype=int)

    for s in range(num_steps):

        # Change direction randomly
        if random.random() < p_change_directions:
            direction = random.sample(directions, 1)[0]

        # Update position
        position = position + direction
        position[0] = max(min(position[0], dim_x - 2), 1)
        position[1] = max(min(position[1], dim_y - 2), 1)

        # Apply mask
        mask = random.sample(masks, 1)[0]
        mask_start = position - 1
        level[mask_start[0]:mask_start[0] + 3, mask_start[1]:mask_start[1] + 3] += mask

    level[level > 0] = 1
    level[:, [0, dim_y - 1]] = 0
    level[[0, dim_x - 1], :] = 0

    return level


def place_boxes_and_player(room, num_boxes, second_player):
    """
    Places the player and the boxes into the floors in a room.

    :param room:
    :param num_boxes:
    :return:
    """
    # Get all available positions
    possible_positions = np.where(room == 1)
    num_possible_positions = possible_positions[0].shape[0]
    num_players = 2 if second_player else 1

    if num_possible_positions <= num_boxes + num_players:
        raise RuntimeError('Not enough free spots (#{}) to place {} player and {} boxes.'.format(
            num_possible_positions,
            num_players,
            num_boxes)
        )

    # Place player(s)
    ind = np.random.randint(num_possible_positions)
    player_position = possible_positions[0][ind], possible_positions[1][ind]
    room[player_position] = 5

    if second_player:
        ind = np.random.randint(num_possible_positions)
        player_position = possible_positions[0][ind], possible_positions[1][ind]
        room[player_position] = 5

    # Place boxes
    for n in range(num_boxes):
        possible_positions = np.where(room == 1)
        num_possible_positions = possible_positions[0].shape[0]

        ind = np.random.randint(num_possible_positions)
        box_position = possible_positions[0][ind], possible_positions[1][ind]
        room[box_position] = 2

    return room


# Global variables used for reverse playing.
explored_states = set()
num_boxes = 0
best_room_score = -1
best_room = None
best_box_mapping = None


def reverse_playing(room_state, room_structure, search_depth=100):
    """
    This function plays Sokoban reverse in a way, such that the player can
    move and pull boxes.
    It ensures a solvable level with all boxes not being placed on a box target.
    :param room_state:
    :param room_structure:
    :param search_depth:
    :return: 2d array, box mapping, action sequence
    """
    global explored_states, num_boxes, best_room_score, best_room, best_box_mapping, best_action_sequence

    # Box_Mapping is used to calculate the box displacement for every box
    box_mapping = {}
    box_locations = np.where(room_structure == 2)
    num_boxes = len(box_locations[0])
    for l in range(num_boxes):
        box = (box_locations[0][l], box_locations[1][l])
        box_mapping[box] = box

    # explored_states globally stores the best room state and score found during search
    explored_states = set()
    best_room_score = -1
    best_room = None
    best_box_mapping = box_mapping
    best_action_sequence = []

    depth_first_search(room_state, room_structure, box_mapping, box_swaps=0, last_pull=(-1, -1), ttl=search_depth, action_sequence=[])

    return best_room, best_box_mapping, best_action_sequence


def depth_first_search(room_state, room_structure, box_mapping, box_swaps=0, last_pull=(-1, -1), ttl=300, action_sequence=[]):
    """
    Searches through all possible states of the room.
    This is a recursive function, which stops if the ttl is reduced to 0 or
    over 1.000.000 states have been explored.
    :param room_state:
    :param room_structure:
    :param box_mapping:
    :param box_swaps:
    :param last_pull:
    :param ttl:
    :param action_sequence:
    :return:
    """
    global explored_states, num_boxes, best_room_score, best_room, best_box_mapping, best_action_sequence

    ttl -= 1
    if ttl <= 0 or len(explored_states) >= 300000:
        return

    state_tohash = marshal.dumps(room_state)

    # Only search this state, if it not yet has been explored
    if not (state_tohash in explored_states):

        # Add current state and its score to explored states
        room_score = box_swaps * box_displacement_score(box_mapping)
        if np.where(room_state == 2)[0].shape[0] != num_boxes:
            room_score = 0

        if room_score > best_room_score:
            best_room = room_state.copy()
            best_room_score = room_score
            best_box_mapping = box_mapping.copy()
            best_action_sequence = action_sequence.copy()

        explored_states.add(state_tohash)

        for action in ACTION_LOOKUP.keys():
            # The state and box mapping need to be copied to ensure
            # every action starts from a similar state.

            # TODO: A tentitive try here to make less moves
            if action >= 4:
                continue

            room_state_next = room_state.copy()
            box_mapping_next = box_mapping.copy()

            room_state_next, box_mapping_next, last_pull_next = \
                reverse_move(room_state_next, room_structure, box_mapping_next, last_pull, action)

            box_swaps_next = box_swaps
            if last_pull_next != last_pull:
                box_swaps_next += 1
            
            action_sequence_next = action_sequence + [action]
            # action_sequence_next = action_sequence + [(action, box_mapping_next != box_mapping)] # add whether a box is moved
            depth_first_search(room_state_next, room_structure, box_mapping_next, box_swaps_next, last_pull_next, ttl, action_sequence_next)
            

def reverse_move(room_state, room_structure, box_mapping, last_pull, action):
    """
    Perform reverse action. Where all actions in the range [0, 3] correspond to
    push actions and the ones greater 3 are simmple move actions.
    :param room_state:
    :param room_structure:
    :param box_mapping:
    :param last_pull:
    :param action:
    :return:
    """
    player_position = np.where(room_state == 5)
    player_position = np.array([player_position[0][0], player_position[1][0]])

    change = CHANGE_COORDINATES[action % 4]
    next_position = player_position + change

    # Check if next position is an empty floor or an empty box target
    if room_state[next_position[0], next_position[1]] in [1, 2]:

        # Move player, independent of pull or move action.
        room_state[player_position[0], player_position[1]] = room_structure[player_position[0], player_position[1]]
        room_state[next_position[0], next_position[1]] = 5

        # In addition try to pull a box if the action is a pull action
        if action < 4:
            possible_box_location = change[0] * -1, change[1] * -1
            possible_box_location += player_position

            if room_state[possible_box_location[0], possible_box_location[1]] in [3, 4]:
                # Perform pull of the adjacent box
                room_state[player_position[0], player_position[1]] = 3
                room_state[possible_box_location[0], possible_box_location[1]] = room_structure[
                    possible_box_location[0], possible_box_location[1]]

                # Update the box mapping
                for k in box_mapping.keys():
                    if box_mapping[k] == (possible_box_location[0], possible_box_location[1]):
                        box_mapping[k] = (player_position[0], player_position[1])
                        last_pull = k

    return room_state, box_mapping, last_pull


def box_displacement_score(box_mapping):
    """
    Calculates the sum of all Manhattan distances, between the boxes
    and their origin box targets.
    :param box_mapping:
    :return:
    """
    score = 0
    
    for box_target in box_mapping.keys():
        box_location = np.array(box_mapping[box_target])
        box_target = np.array(box_target)
        dist = np.sum(np.abs(box_location - box_target))
        score += dist

    return score


TYPE_LOOKUP = {
    0: 'wall',
    1: 'empty space',
    2: 'box target',
    3: 'box on target',
    4: 'box not on target',
    5: 'player'
}

ACTION_LOOKUP = {
    0: 'push up',
    1: 'push down',
    2: 'push left',
    3: 'push right',
    4: 'move up',
    5: 'move down',
    6: 'move left',
    7: 'move right',
}

# Moves are mapped to coordinate changes as follows
# 0: Move up
# 1: Move down
# 2: Move left
# 3: Move right
CHANGE_COORDINATES = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1)
}
