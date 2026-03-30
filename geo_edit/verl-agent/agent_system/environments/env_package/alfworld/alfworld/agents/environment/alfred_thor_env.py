import os
import json
import numpy as np

import traceback
import threading
from queue import Queue
from threading import Thread
import random

import alfworld.agents
from alfworld.agents.utils.misc import get_templated_task_desc
from alfworld.env.thor_env import ThorEnv
from alfworld.agents.expert import HandCodedThorAgent, HandCodedAgentTimeout
from alfworld.agents.detector.mrcnn import load_pretrained_model
from alfworld.agents.controller import OracleAgent, OracleAStarAgent, MaskRCNNAgent, MaskRCNNAStarAgent


TASK_TYPES = {1: "pick_and_place_simple",
              2: "look_at_obj_in_light",
              3: "pick_clean_then_place_in_recep",
              4: "pick_heat_then_place_in_recep",
              5: "pick_cool_then_place_in_recep",
              6: "pick_two_obj_and_place"}


class AlfredThorEnv(object):
    '''
    Interface for Embodied (THOR) environment
    '''

    class Thor(threading.Thread):
        def __init__(self, queue, train_eval="train"):
            Thread.__init__(self)
            self.action_queue = queue
            self.mask_rcnn = None
            self.env =  None
            self.train_eval = train_eval
            self.controller_type = "oracle"

        def run(self):
            while True:
                action, reset, task_file = self.action_queue.get()
                try:
                    if reset:
                        self.reset(task_file)
                    else:
                        self.step(action)
                finally:
                    self.action_queue.task_done()

        def init_env(self, config):
            self.config = config

            screen_height = config['env']['thor']['screen_height']
            screen_width = config['env']['thor']['screen_width']
            smooth_nav = config['env']['thor']['smooth_nav']
            save_frames_to_disk = config['env']['thor']['save_frames_to_disk']

            if not self.env:
                self.env = ThorEnv(player_screen_height=screen_height,
                                   player_screen_width=screen_width,
                                   smooth_nav=smooth_nav,
                                   save_frames_to_disk=save_frames_to_disk)
            self.controller_type = self.config['controller']['type']
            self._done = False
            self._res = ()
            self._feedback = ""
            self.expert = HandCodedThorAgent(self.env, max_steps=200)
            self.prev_command = ""
            self.load_mask_rcnn()

        def load_mask_rcnn(self):
            # load pretrained MaskRCNN model if required
            if 'mrcnn' in self.config['controller']['type'] and not self.mask_rcnn:
                model_path = os.path.expandvars(self.config['mask_rcnn']['pretrained_model_path'])
                self.mask_rcnn = load_pretrained_model(model_path)

        def set_task(self, task_file):
            self.task_file = task_file
            self.traj_root = os.path.dirname(task_file)
            with open(task_file, 'r') as f:
                self.traj_data = json.load(f)

        def reset(self, task_file):
            assert self.env
            assert self.controller_type

            self.set_task(task_file)

            # scene setup
            scene_num = self.traj_data['scene']['scene_num']
            object_poses = self.traj_data['scene']['object_poses']
            dirty_and_empty = self.traj_data['scene']['dirty_and_empty']
            object_toggles = self.traj_data['scene']['object_toggles']
            scene_name = 'FloorPlan%d' % scene_num
            self.env.reset(scene_name)
            self.env.restore_scene(object_poses, object_toggles, dirty_and_empty)

            # recording
            save_frames_path = self.config['env']['thor']['save_frames_path']
            self.env.save_frames_path = os.path.join(save_frames_path, self.traj_root.replace('../', ''))

            # initialize to start position
            self.env.step(dict(self.traj_data['scene']['init_action']))            # print goal instr
            task_desc = get_templated_task_desc(self.traj_data)
            print("Task: %s" % task_desc)
            # print("Task: %s" % (self.traj_data['turk_annotations']['anns'][0]['task_desc']))

            # setup task for reward
            class args: pass
            args.reward_config = os.path.join(alfworld.agents.__path__[0], 'config/rewards.json')
            self.env.set_task(self.traj_data, args, reward_type='dense')

            # set controller
            self.controller_type = self.config['controller']['type']
            self.goal_desc_human_anns_prob = self.config['env']['goal_desc_human_anns_prob']
            load_receps = self.config['controller']['load_receps']
            debug = self.config['controller']['debug']

            if self.controller_type == 'oracle':
                self.controller = OracleAgent(self.env, self.traj_data, self.traj_root,
                                              load_receps=load_receps, debug=debug,
                                              goal_desc_human_anns_prob=self.goal_desc_human_anns_prob)
            elif self.controller_type == 'oracle_astar':
                self.controller = OracleAStarAgent(self.env, self.traj_data, self.traj_root,
                                                   load_receps=load_receps, debug=debug,
                                                   goal_desc_human_anns_prob=self.goal_desc_human_anns_prob)
            elif self.controller_type == 'mrcnn':
                self.controller = MaskRCNNAgent(self.env, self.traj_data, self.traj_root,
                                                pretrained_model=self.mask_rcnn,
                                                load_receps=load_receps, debug=debug,
                                                goal_desc_human_anns_prob=self.goal_desc_human_anns_prob,
                                                save_detections_to_disk=self.env.save_frames_to_disk, save_detections_path=self.env.save_frames_path)
            elif self.controller_type == 'mrcnn_astar':
                self.controller = MaskRCNNAStarAgent(self.env, self.traj_data, self.traj_root,
                                                     pretrained_model=self.mask_rcnn,
                                                     load_receps=load_receps, debug=debug,
                                                     goal_desc_human_anns_prob=self.goal_desc_human_anns_prob,
                                                     save_detections_to_disk=self.env.save_frames_to_disk, save_detections_path=self.env.save_frames_path)
            else:
                raise NotImplementedError()

            # zero steps
            self.steps = 0

            # reset expert state
            self.expert.reset(task_file)
            self.prev_command = ""

            # return intro text
            self._feedback = self.controller.feedback
            self._res = self.get_info()

            return self._feedback

        def step(self, action):
            if not self._done:
                # take action
                self.prev_command = str(action)
                self._feedback = self.controller.step(action)
                self._res = self.get_info()
                if self.env.save_frames_to_disk:
                    self.record_action(action)
            self.steps += 1

        def get_results(self):
            return self._res

        def record_action(self, action):
            txt_file = os.path.join(self.env.save_frames_path, 'action.txt')
            with open(txt_file, 'a+') as f:
                f.write("%s\r\n" % str(action))

        def get_info(self):
            won = self.env.get_goal_satisfied()
            pcs = self.env.get_goal_conditions_met()
            goal_condition_success_rate = pcs[0] / float(pcs[1])
            acs = self.controller.get_admissible_commands()

            # expert action
            if self.train_eval == "train":
                game_state = {
                    'admissible_commands': acs,
                    'feedback': self._feedback,
                    'won': won
                }
                expert_actions = ["look"]
                try:
                    if not self.prev_command:
                        self.expert.observe(game_state['feedback'])
                    else:
                        next_action = self.expert.act(game_state, 0, won, self.prev_command)
                        if next_action in acs:
                            expert_actions = [next_action]
                except HandCodedAgentTimeout:
                    print("Expert Timeout")
                except Exception as e:
                    print(e)
                    traceback.print_exc()
            else:
                expert_actions = []

            training_method = self.config["general"]["training_method"]
            if training_method == "dqn":
                max_nb_steps_per_episode = self.config["rl"]["training"]["max_nb_steps_per_episode"]
            elif training_method == "dagger":
                max_nb_steps_per_episode = self.config["dagger"]["training"]["max_nb_steps_per_episode"]
            else:
                raise NotImplementedError
            self._done = won or self.steps > max_nb_steps_per_episode
            return (self._feedback, self._done, acs, won, goal_condition_success_rate, expert_actions)

        def get_last_frame(self):
            return self.env.last_event.frame[:,:,::-1]

        def get_exploration_frames(self):
            return self.controller.get_exploration_frames()

    def __init__(self, config, train_eval="train"):
        print("Initialize AlfredThorEnv...")
        self.config = config
        self.train_eval = train_eval
        self.random_seed = 123
        self.batch_size = 1
        self.envs = []
        self.action_queues = []
        self.get_env_paths()

    def close(self):
        for env in self.envs:
            env.env.stop()

    def seed(self, seed):
        self.random_seed = seed

    def get_env_paths(self):
        self.json_file_list = []

        if self.train_eval == "train":
            data_path = os.path.expandvars(self.config['dataset']['data_path'])
        elif self.train_eval == "eval_in_distribution":
            data_path = os.path.expandvars(self.config['dataset']['eval_id_data_path'])
        elif self.train_eval == "eval_out_of_distribution":
            data_path = os.path.expandvars(self.config['dataset']['eval_ood_data_path'])
        else:
            raise Exception("Invalid split. Must be either train or eval")

        # get task types
        assert len(self.config['env']['task_types']) > 0
        task_types = []
        for tt_id in self.config['env']['task_types']:
            if tt_id in TASK_TYPES:
                task_types.append(TASK_TYPES[tt_id])

        for root, dirs, files in os.walk(data_path, topdown=False):
            if 'traj_data.json' in files:
                # Skip movable and slice objects object tasks
                if 'movable' in root or 'Sliced' in root:
                    continue

                # File paths
                json_path = os.path.join(root, 'traj_data.json')
                game_file_path = os.path.join(root, "game.tw-pddl")

                # Load trajectory file
                with open(json_path, 'r') as f:
                    traj_data = json.load(f)

                # Check for any task_type constraints
                if not traj_data['task_type'] in task_types:
                    continue

                self.json_file_list.append(json_path)

                # # Only add solvable games
                # if os.path.exists(game_file_path):
                #     with open(game_file_path, 'r') as f:
                #         gamedata = json.load(f)
                #
                #     if 'solvable' in gamedata and gamedata['solvable']:
                #         self.json_file_list.append(json_path)

        print("Overall we have %s games..." % (str(len(self.json_file_list))))
        self.num_games = len(self.json_file_list)

        if self.train_eval == "train":
            num_train_games = self.config['dataset']['num_train_games'] if self.config['dataset']['num_train_games'] > 0 else len(self.json_file_list)
            self.json_file_list = self.json_file_list[:num_train_games]
            self.num_games = len(self.json_file_list)
            print("Training with %d games" % (len(self.json_file_list)))
        else:
            num_eval_games = self.config['dataset']['num_eval_games'] if self.config['dataset']['num_eval_games'] > 0 else len(self.json_file_list)
            self.json_file_list = self.json_file_list[:num_eval_games]
            self.num_games = len(self.json_file_list)
            print("Evaluating with %d games" % (len(self.json_file_list)))

    def init_env(self, batch_size):
        self.get_env_paths()
        self.batch_size = batch_size
        self.action_queues = []
        self.task_order = [""] * self.batch_size
        for n in range(self.batch_size):
            queue = Queue()
            env = self.Thor(queue, self.train_eval)
            self.action_queues.append(queue)
            self.envs.append(env)
            env.daemon = True
            env.start()
            env.init_env(self.config)
        return self

    def reset(self):
        # set tasks
        batch_size = self.batch_size
        # reset envs

        if self.train_eval == 'train':
            tasks = random.sample(self.json_file_list, k=batch_size)
        else:
            if len(self.json_file_list)-batch_size > batch_size:
                tasks = [self.json_file_list.pop(random.randrange(len(self.json_file_list))) for _ in range(batch_size)]
            else:
                tasks = random.sample(self.json_file_list, k=batch_size)
                self.get_env_paths()

        for n in range(batch_size):
            self.action_queues[n].put((None, True, tasks[n]))

        obs, dones, infos = self.wait_and_get_info()
        return obs, infos

    def step(self, actions):
        '''
        executes actions in parallel and waits for all env to finish
        '''

        batch_size = self.batch_size
        for n in range(batch_size):
            self.action_queues[n].put((actions[n], False, ""))

        obs, dones, infos = self.wait_and_get_info()
        return obs, None, dones, infos

    def wait_and_get_info(self):
        obs, dones, admissible_commands, wons, gamefiles, expert_plans, gc_srs = [], [], [], [], [], [], []

        # wait for all threads
        for n in range(self.batch_size):
            self.action_queues[n].join()
            feedback, done, acs, won, gc_sr, expert_actions = self.envs[n].get_results()
            obs.append(feedback)
            dones.append(done)
            admissible_commands.append(acs)
            wons.append(won)
            gc_srs.append(gc_sr)
            gamefiles.append(self.envs[n].traj_root)
            expert_plans.append(expert_actions)

        infos = {'admissible_commands': admissible_commands,
                 'won': wons,
                 'goal_condition_success_rate': gc_srs,
                 'extra.gamefile': gamefiles,
                 'extra.expert_plan': expert_plans}
        return obs, dones, infos

    def get_frames(self):
        images = []
        for n in range(self.batch_size):
            images.append(self.envs[n].get_last_frame())
        return np.array(images)

    def get_exploration_frames(self):
        images = []
        for n in range(self.batch_size):
            images.append(self.envs[n].get_exploration_frames())
        return images
