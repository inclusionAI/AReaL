import logging
import numpy as np

try:
    import torch
    from transformers import DistilBertModel, DistilBertTokenizer
except ImportError:
    raise ImportError("torch or transformers not found. Please install them via `pip install alfworld[full]`.")

logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)

import alfworld.agents.modules.memory as memory
from alfworld.agents.modules.model import Policy
from alfworld.agents.modules.generic import to_np, to_pt, _words_to_ids, pad_sequences, preproc, max_len, ez_gather_dim_1, LinearSchedule, BeamSearchNode


class ObservationPool(object):

    def __init__(self, capacity=1):
        if capacity == 0:
            self.capacity = 1
            self.disable_observation = True
        elif capacity > 0:
            self.capacity = capacity
            self.disable_observation = False
        else:
            raise NotImplementedError

    def identical_with_history(self, new_stuff, list_of_old_stuff):
        for i in range(len(list_of_old_stuff)):
            if new_stuff == list_of_old_stuff[i]:
                return True
        return False

    def push_batch(self, stuff):
        assert len(stuff) == len(self.memory)
        for i in range(len(stuff)):
            self.push_one(i, stuff[i])

    def push_one(self, which, stuff):
        assert which < len(self.memory)
        if len(self.memory[which]) == 0 and stuff.endswith("restart"):
            return
        if self.disable_observation:
            if "[SEP]" in stuff:
                action = stuff.split("[SEP]", 1)[-1].strip()
                self.memory[which].append(action)
        else:
            if "Nothing happens" in stuff:
                return
            if not self.identical_with_history(stuff, self.memory[which]):
                self.memory[which].append(stuff)
        if len(self.memory[which]) > self.capacity:
            self.memory[which] = self.memory[which][-self.capacity:]

    def push_first_sight(self, stuff):
        assert len(stuff) == len(self.memory)
        for i in range(len(stuff)):
            self.first_sight.append(stuff[i])

    def get(self, which=None):
        if which is not None:
            assert which < len(self.memory)
            output = [self.first_sight[which]]
            for idx in range(len(self.memory[which])):
                output.append(self.memory[which][idx])
            return " [SEP] ".join(output)

        output = []
        for i in range(len(self.memory)):
            output.append(self.get(which=i))
        return output

    def reset(self, batch_size):
        self.memory = []
        self.first_sight = []
        for _ in range(batch_size):
            self.memory.append([])

    def __len__(self):
        return len(self.memory)


class BaseAgent:
    '''
    Base class for agents
    '''

    def __init__(self, config):
        self.mode = "train"
        self.config = config
        print(self.config)
        self.load_config()

        # bert tokenizer and model
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
        self.word2id = self.tokenizer.get_vocab()
        self.word_vocab = {value:key for key, value in self.word2id.items()}
        bert_model = DistilBertModel.from_pretrained('distilbert-base-cased')
        bert_model.transformer = None
        bert_model.encoder = None
        for param in bert_model.parameters():
            param.requires_grad = False

        self.online_net = Policy(config=self.config, bert_model=bert_model, word_vocab_size=len(self.word2id))
        self.target_net = Policy(config=self.config, bert_model=bert_model, word_vocab_size=len(self.word2id))
        self.online_net.train()
        self.target_net.train()
        self.update_target_net()
        for param in self.target_net.parameters():
            param.requires_grad = False
        if self.use_cuda:
            self.online_net.cuda()
            self.target_net.cuda()

        # optimizer
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=self.config['general']['training']['optimizer']['learning_rate'])
        self.clip_grad_norm = self.config['general']['training']['optimizer']['clip_grad_norm']

        # losses
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def load_config(self):
        self.task = self.config['general']['task']
        self.observation_pool_capacity = self.config['general']['observation_pool_capacity']
        self.observation_pool = ObservationPool(capacity=self.observation_pool_capacity)
        self.hide_init_receptacles = self.config['general']['hide_init_receptacles']
        self.training_method = self.config['general']['training_method']

        self.init_learning_rate = self.config['general']['training']['optimizer']['learning_rate']
        self.clip_grad_norm = self.config['general']['training']['optimizer']['clip_grad_norm']
        self.batch_size = self.config['general']['training']['batch_size']
        self.max_episode = self.config['general']['training']['max_episode']
        self.smoothing_eps = self.config['general']['training']['smoothing_eps']

        self.run_eval = self.config['general']['evaluate']['run_eval']
        self.eval_batch_size = self.config['general']['evaluate']['batch_size']

        # Set the random seed manually for reproducibility.
        self.random_seed = self.config['general']['random_seed']
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            if not self.config['general']['use_cuda']:
                print("WARNING: CUDA device detected but 'use_cuda: false' found in config.yaml")
                self.use_cuda = False
            else:
                torch.backends.cudnn.deterministic = True
                torch.cuda.manual_seed(self.random_seed)
                self.use_cuda = True
        else:
            self.use_cuda = False

        self.experiment_tag = self.config['general']['checkpoint']['experiment_tag']
        self.report_frequency = self.config['general']['checkpoint']['report_frequency']
        self.load_pretrained = self.config['general']['checkpoint']['load_pretrained']
        self.load_from_tag = self.config['general']['checkpoint']['load_from_tag']

        self.recurrent = self.config['general']['model']['recurrent']

        # RL specific
        # epsilon greedy
        self.epsilon_anneal_episodes = self.config['rl']['epsilon_greedy']['epsilon_anneal_episodes']
        self.epsilon_anneal_from = self.config['rl']['epsilon_greedy']['epsilon_anneal_from']
        self.epsilon_anneal_to = self.config['rl']['epsilon_greedy']['epsilon_anneal_to']
        self.epsilon = self.epsilon_anneal_from
        self.epsilon_scheduler = LinearSchedule(schedule_timesteps=self.epsilon_anneal_episodes, initial_p=self.epsilon_anneal_from, final_p=self.epsilon_anneal_to)
        self.noisy_net = self.config['rl']['epsilon_greedy']['noisy_net']
        if self.noisy_net:
            # disable epsilon greedy
            self.epsilon_anneal_episodes = -1
            self.epsilon = 0.0

        # replay buffer and updates
        self.accumulate_reward_from_final = self.config['rl']['replay']['accumulate_reward_from_final']
        self.discount_gamma_game_reward = self.config['rl']['replay']['discount_gamma_game_reward']
        self.discount_gamma_count_reward = self.config['rl']['replay']['discount_gamma_count_reward']
        self.discount_gamma_novel_object_reward = self.config['rl']['replay']['discount_gamma_novel_object_reward']
        self.replay_batch_size = self.config['rl']['replay']['replay_batch_size']
        self.dqn_memory = memory.PrioritizedReplayMemory(self.config['rl']['replay']['replay_memory_capacity'],
                                                         priority_fraction=self.config['rl']['replay']['replay_memory_priority_fraction'],
                                                         discount_gamma_game_reward=self.discount_gamma_game_reward,
                                                         discount_gamma_count_reward=self.discount_gamma_count_reward,
                                                         discount_gamma_novel_object_reward=self.discount_gamma_novel_object_reward,
                                                         accumulate_reward_from_final=self.accumulate_reward_from_final)
        self.update_per_k_game_steps = self.config['rl']['replay']['update_per_k_game_steps']
        self.multi_step = self.config['rl']['replay']['multi_step']
        self.count_reward_lambda = self.config['rl']['replay']['count_reward_lambda']
        self.novel_object_reward_lambda = self.config['rl']['replay']['novel_object_reward_lambda']
        self.rl_replay_sample_history_length = self.config['rl']['replay']['replay_sample_history_length']
        self.rl_replay_sample_update_from = self.config['rl']['replay']['replay_sample_update_from']

        # rl train and eval
        self.learn_start_from_this_episode = self.config['rl']['training']['learn_start_from_this_episode']
        self.target_net_update_frequency = self.config['rl']['training']['target_net_update_frequency']

        # dagger
        self.fraction_assist_anneal_episodes = self.config['dagger']['fraction_assist']['fraction_assist_anneal_episodes']
        self.fraction_assist_anneal_from = self.config['dagger']['fraction_assist']['fraction_assist_anneal_from']
        self.fraction_assist_anneal_to = self.config['dagger']['fraction_assist']['fraction_assist_anneal_to']
        self.fraction_assist = self.fraction_assist_anneal_from
        self.fraction_assist_scheduler = LinearSchedule(schedule_timesteps=self.fraction_assist_anneal_episodes, initial_p=self.fraction_assist_anneal_from, final_p=self.fraction_assist_anneal_to)

        self.fraction_random_anneal_episodes = self.config['dagger']['fraction_random']['fraction_random_anneal_episodes']
        self.fraction_random_anneal_from = self.config['dagger']['fraction_random']['fraction_random_anneal_from']
        self.fraction_random_anneal_to = self.config['dagger']['fraction_random']['fraction_random_anneal_to']
        self.fraction_random = self.fraction_random_anneal_from
        self.fraction_random_scheduler = LinearSchedule(schedule_timesteps=self.fraction_random_anneal_episodes, initial_p=self.fraction_random_anneal_from, final_p=self.fraction_random_anneal_to)

        self.dagger_memory = memory.DaggerReplayMemory(self.config['dagger']['replay']['replay_memory_capacity'])
        self.dagger_update_per_k_game_steps = self.config['dagger']['replay']['update_per_k_game_steps']
        self.dagger_replay_batch_size = self.config['dagger']['replay']['replay_batch_size']
        self.dagger_replay_sample_history_length = self.config['dagger']['replay']['replay_sample_history_length']
        self.dagger_replay_sample_update_from = self.config['dagger']['replay']['replay_sample_update_from']

        if self.training_method == "dagger":
            self.max_target_length = self.config['dagger']['max_target_length']
            self.generate_top_k = self.config['dagger']['generate_top_k']
            self.beam_width = self.config['dagger']['beam_width']
            self.action_space = self.config['dagger']['action_space']
            self.max_nb_steps_per_episode = self.config['dagger']['training']['max_nb_steps_per_episode']
            self.unstick_by_beam_search = self.config['dagger']['unstick_by_beam_search']
        elif self.training_method == "dqn":
            self.max_target_length = self.config['rl']['max_target_length']
            self.generate_top_k = self.config['rl']['generate_top_k']
            self.beam_width = self.config['rl']['beam_width']
            self.action_space = self.config['rl']['action_space']
            self.max_nb_steps_per_episode = self.config['rl']['training']['max_nb_steps_per_episode']
            self.unstick_by_beam_search = None
        else:
            raise NotImplementedError

    def train(self):
        """
        Tell the agent that it's training phase.
        """
        self.mode = "train"
        self.online_net.train()

    def eval(self):
        """
        Tell the agent that it's evaluation phase.
        """
        self.mode = "eval"
        self.online_net.eval()

    def update_target_net(self):
        if self.target_net is not None:
            self.target_net.load_state_dict(self.online_net.state_dict())

    def reset_noise(self):
        if self.noisy_net:
            # Resets noisy weights in all linear layers (of online net only)
            self.online_net.reset_noise()

    def load_pretrained_model(self, load_from):
        """
        Load pretrained checkpoint from file.

        Arguments:
            load_from: File name of the pretrained model checkpoint.
        """
        print("loading model from %s\n" % (load_from))
        try:
            if self.use_cuda:
                pretrained_dict = torch.load(load_from)
            else:
                pretrained_dict = torch.load(load_from, map_location='cpu')

            model_dict = self.online_net.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.online_net.load_state_dict(model_dict)
            print("The loaded parameters are:")
            keys = [key for key in pretrained_dict]
            print(", ".join(keys))
            print("--------------------------")
        except:
            print("Failed to load checkpoint...")

    def save_model_to_path(self, save_to):
        torch.save(self.online_net.state_dict(), save_to)
        print("Saved checkpoint to %s..." % (save_to))

    def init(self, batch_size):
        self.observation_pool.reset(batch_size)

    def get_word_input(self, input_strings):
        word_id_list = [self.tokenizer.encode(item, add_special_tokens=False) for item in input_strings]
        return self.get_word_input_from_ids(word_id_list)

    def get_word_input_from_ids(self, word_id_list):
        input_word = pad_sequences(word_id_list, maxlen=max_len(word_id_list) + 3, dtype='int32')  # 3 --> see layer.DepthwiseSeparableConv.padding
        input_word = to_pt(input_word, self.use_cuda)
        return input_word

    def get_action_candidate_representations(self, action_candidate_list, use_model="online"):
        # in case there are too many candidates in certain data point, we compute their candidate representations by small batches
        batch_size = len(action_candidate_list)
        max_num_candidate = max_len(action_candidate_list)
        res_representations = torch.zeros(batch_size, max_num_candidate, self.online_net.block_hidden_dim)
        res_mask = torch.zeros(batch_size, max_num_candidate)
        if self.use_cuda:
            res_representations = res_representations.cuda()
            res_mask = res_mask.cuda()

        squeezed_candidate_list, from_which_original_batch = [], []
        for b in range(batch_size):
            squeezed_candidate_list += action_candidate_list[b]
            for i in range(len(action_candidate_list[b])):
                from_which_original_batch.append((b, i))

        tmp_batch_size = 64
        n_tmp_batches = (len(squeezed_candidate_list) + tmp_batch_size - 1) // tmp_batch_size
        for tmp_batch_id in range(n_tmp_batches):
            tmp_batch_cand = squeezed_candidate_list[tmp_batch_id * tmp_batch_size: (tmp_batch_id + 1) * tmp_batch_size]  # tmp_batch of candidates
            tmp_batch_from = from_which_original_batch[tmp_batch_id * tmp_batch_size: (tmp_batch_id + 1) * tmp_batch_size]

            tmp_batch_cand_representation_sequence, tmp_batch_cand_mask = self.encode_text(tmp_batch_cand, use_model=use_model)  # tmp_batch x num_word x hid, tmp_batch x num_word

            # masked mean the num_word dimension
            _mask = torch.sum(tmp_batch_cand_mask, -1)  # batch
            tmp_batch_cand_representation = torch.sum(tmp_batch_cand_representation_sequence, -2)  # batch x hid
            tmp = torch.eq(_mask, 0).float()
            if tmp_batch_cand_representation.is_cuda:
                tmp = tmp.cuda()
            _mask = _mask + tmp
            tmp_batch_cand_representation = tmp_batch_cand_representation / _mask.unsqueeze(-1)  # batch x hid
            tmp_batch_cand_mask = tmp_batch_cand_mask.byte().any(-1).float()  # batch

            for i in range(len(tmp_batch_from)):
                res_representations[tmp_batch_from[i][0], tmp_batch_from[i][1], :] = tmp_batch_cand_representation[i]
                res_mask[tmp_batch_from[i][0], tmp_batch_from[i][1]] = tmp_batch_cand_mask[i]

        return res_representations, res_mask

    def choose_model(self, use_model="online"):
        if use_model == "online":
            model = self.online_net
        elif use_model == "target":
            model = self.target_net
        else:
            raise NotImplementedError
        return model

    def encode_text(self, observation_strings, use_model):
        model = self.choose_model(use_model)
        input_obs = self.get_word_input(observation_strings)
        # encode
        obs_encoding_sequence, obs_mask = model.encode_text(input_obs)
        return obs_encoding_sequence, obs_mask

    def finish_of_episode(self, episode_no, batch_size):
        # fraction_assist annealing
        self.fraction_assist = self.fraction_assist_scheduler.value(episode_no)
        self.fraction_assist = max(self.fraction_assist, 0.0)

        # fraction_random annealing
        self.fraction_random = self.fraction_random_scheduler.value(episode_no)
        self.fraction_random = max(self.fraction_random, 0.0)

        # Update target network
        if (episode_no + batch_size) % self.target_net_update_frequency <= episode_no % self.target_net_update_frequency:
            self.update_target_net()
        # decay lambdas
        if episode_no < self.learn_start_from_this_episode:
            return
        if episode_no < self.epsilon_anneal_episodes + self.learn_start_from_this_episode:
            self.epsilon = self.epsilon_scheduler.value(episode_no - self.learn_start_from_this_episode)
            self.epsilon = max(self.epsilon, 0.0)

    def preprocess_task(self, task_strings):
        return [preproc(item) for item in task_strings]

    def preprocess_observation(self, observation_strings):
        res = []
        for i in range(len(observation_strings)):
            obs = observation_strings[i]
            obs = preproc(obs)
            # sort objects
            if "you see" in obs:
                # -= Welcome to TextWorld, ALFRED! =- You are in the middle of a room. Looking quickly around you, you see a armchair 2, a diningtable 1, a diningtable 2, a coffeetable 1, a armchair 1, a tvstand 1, a sidetable 2, a garbagecan 1, a sidetable 1, a sofa 1, a drawer 1, and a drawer 2.
                before_you_see, after_you_see = obs.split("you see", 1)
                before_you_see, after_you_see = before_you_see.strip(), after_you_see.strip()
                before_you_see = " ".join([before_you_see, "you see"])
                object_list, after_period = after_you_see.split(".", 1)
                object_list, after_period = object_list.strip(), after_period.strip()
                after_period += "."
                object_list = object_list.replace(", and ", ", ")
                object_list = object_list.split(", ")
                object_list = sorted(object_list)
                if len(object_list) == 1:
                    object_string = object_list[0]
                else:
                    object_string = ", ".join(object_list[:-1]) + ", and " + object_list[-1]
                obs = " ".join([before_you_see, object_string]) + after_period
            res.append(obs)
        return res

    def preprocess_action_candidates(self, action_candidate_list):
        batch_size = len(action_candidate_list)
        preproced_action_candidate_list = []
        for b in range(batch_size):
            ac = [preproc(item) for item in action_candidate_list[b]]
            preproced_action_candidate_list.append(ac)
        return preproced_action_candidate_list

    def preprocessing(self, observation_strings, action_candidate_list):
        preproced_observation_strings = self.preprocess_observation(observation_strings)
        preproced_action_candidate_list = self.preprocess_action_candidates(action_candidate_list)
        return preproced_observation_strings, preproced_action_candidate_list

    def get_task_and_obs(self, observation_strings):
        batch_size = len(observation_strings)
        task_desc_strings, no_goal_observation_strings = [], []
        for b in range(batch_size):
            task_desc = observation_strings[b].partition("Your task is to: ")[-1]
            no_goal_obs_str = observation_strings[b].replace("Your task is to: %s" % task_desc, "")
            if self.hide_init_receptacles:
                no_goal_obs_str = no_goal_obs_str.partition("Looking quickly around you, you see")[0]
            task_desc_strings.append(task_desc)
            no_goal_observation_strings.append(no_goal_obs_str)
        return task_desc_strings, no_goal_observation_strings

    def encode(self, observation_strings, use_model):
        obs_enc_seq, obs_mask = self.encode_text(observation_strings, use_model=use_model)
        return obs_enc_seq, obs_mask

    def action_scoring(self, action_candidate_list, h_obs, obs_mask, h_td, td_mask, previous_dynamics, use_model=None):
        model = self.choose_model(use_model)
        average_action_candidate_representations, action_candidate_mask = self.get_action_candidate_representations(action_candidate_list, use_model=use_model)  # batch x num_cand x hid, batch x num_cand
        aggregated_obs_representation = model.aggretate_information(h_obs, obs_mask, h_td, td_mask)  # batch x obs_length x hid

        if self.recurrent:
            averaged_representation = model.masked_mean(aggregated_obs_representation, obs_mask)  # batch x hid
            current_dynamics = model.rnncell(averaged_representation, previous_dynamics) if previous_dynamics is not None else model.rnncell(averaged_representation)
        else:
            current_dynamics = None

        action_scores, action_masks = model.score_actions(average_action_candidate_representations, action_candidate_mask,
                                                          aggregated_obs_representation, obs_mask, current_dynamics)  # batch x num_actions
        return action_scores, action_masks, current_dynamics

    def beam_search_candidate_scoring(self, action_candidate_list, aggregated_obs_representation, obs_mask, current_dynamics, use_model=None):
        model = self.choose_model(use_model)
        average_action_candidate_representations, action_candidate_mask = self.get_action_candidate_representations(action_candidate_list, use_model=use_model)  # batch x num_cand x hid, batch x num_cand

        action_scores, action_masks = model.score_actions(average_action_candidate_representations, action_candidate_mask,
                                                          aggregated_obs_representation, obs_mask, current_dynamics, fix_shared_components=True)  # batch x num_actions
        return action_scores, action_masks

