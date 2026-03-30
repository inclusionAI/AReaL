import copy
import operator
import logging
from queue import PriorityQueue

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    raise ImportError("torch not found. Please install them via `pip install alfworld[full]`.")
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)

from alfworld.agents.agent import BaseAgent
from alfworld.agents.modules.generic import to_np, to_pt, _words_to_ids, pad_sequences, preproc, max_len, ez_gather_dim_1, LinearSchedule, BeamSearchNode
from alfworld.agents.modules.layers import NegativeLogLoss, masked_mean, compute_mask, GetGenerationQValue


class TextDQNAgent(BaseAgent):
    '''
    TextAgent trained with DQN (Reinforcement Learning)
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.training_method == "dqn"

    def choose_random_action(self, action_rank, action_unpadded=None):
        """
        Select an action randomly.
        """
        batch_size = action_rank.size(0)
        action_space_size = action_rank.size(1)
        if action_unpadded is None:
            indices = np.random.choice(action_space_size, batch_size)
        else:
            indices = []
            for j in range(batch_size):
                indices.append(np.random.choice(len(action_unpadded[j])))
            indices = np.array(indices)
        return indices

    def choose_maxQ_action(self, action_rank, action_mask=None):
        """
        Generate an action by maximum q values.
        """
        action_rank = action_rank - torch.min(action_rank, -1, keepdim=True)[0] + 1e-2  # minus the min value, so that all values are non-negative
        if action_mask is not None:
            assert action_mask.size() == action_rank.size(), (action_mask.size().shape, action_rank.size())
            action_rank = action_rank * action_mask
        action_indices = torch.argmax(action_rank, -1)  # batch
        return to_np(action_indices)

    # choosing from list of admissible commands
    def admissible_commands_act_greedy(self, observation_strings, task_desc_strings, action_candidate_list, previous_dynamics):
        with torch.no_grad():
            h_obs, obs_mask = self.encode(observation_strings, use_model="online")
            h_td, td_mask = self.encode(task_desc_strings, use_model="online")
            action_scores, action_masks, current_dynamics = self.action_scoring(action_candidate_list,
                                                                                h_obs, obs_mask,
                                                                                h_td, td_mask,
                                                                                previous_dynamics,
                                                                                use_model="online")
            action_indices_maxq = self.choose_maxQ_action(action_scores, action_masks)
            chosen_indices = action_indices_maxq
            chosen_indices = chosen_indices.astype(int)
            chosen_actions = [item[idx] for item, idx in zip(action_candidate_list, chosen_indices)]

            return chosen_actions, chosen_indices, current_dynamics

    def admissible_commands_act_random(self, observation_strings, task_desc_strings, action_candidate_list, previous_dynamics):
        with torch.no_grad():
            h_obs, obs_mask = self.encode(observation_strings, use_model="online")
            h_td, td_mask = self.encode(task_desc_strings, use_model="online")
            action_scores, action_masks, current_dynamics = self.action_scoring(action_candidate_list,
                                                                                h_obs, obs_mask,
                                                                                h_td, td_mask,
                                                                                previous_dynamics,
                                                                                use_model="online")
            action_indices_random = self.choose_random_action(action_scores, action_candidate_list)

            chosen_indices = action_indices_random
            chosen_indices = chosen_indices.astype(int)
            chosen_actions = [item[idx] for item, idx in zip(action_candidate_list, chosen_indices)]
            return chosen_actions, chosen_indices, current_dynamics

    def admissible_commands_act(self, observation_strings, task_desc_strings, action_candidate_list, previous_dynamics, random=False):

        with torch.no_grad():
            if self.mode == "eval":
                return self.admissible_commands_act_greedy(observation_strings, task_desc_strings, action_candidate_list, previous_dynamics)
            if random:
                return self.admissible_commands_act_random(observation_strings, task_desc_strings, action_candidate_list, previous_dynamics)
            batch_size = len(observation_strings)

            h_obs, obs_mask = self.encode(observation_strings, use_model="online")
            h_td, td_mask = self.encode(task_desc_strings, use_model="online")
            action_scores, action_masks, current_dynamics = self.action_scoring(action_candidate_list,
                                                                                h_obs, obs_mask,
                                                                                h_td, td_mask,
                                                                                previous_dynamics,
                                                                                use_model="online")

            action_indices_maxq = self.choose_maxQ_action(action_scores, action_masks)
            action_indices_random = self.choose_random_action(action_scores, action_candidate_list)

            # random number for epsilon greedy
            rand_num = np.random.uniform(low=0.0, high=1.0, size=(batch_size,))
            less_than_epsilon = (rand_num < self.epsilon).astype("float32")  # batch
            greater_than_epsilon = 1.0 - less_than_epsilon

            chosen_indices = less_than_epsilon * action_indices_random + greater_than_epsilon * action_indices_maxq
            chosen_indices = chosen_indices.astype(int)
            chosen_actions = [item[idx] for item, idx in zip(action_candidate_list, chosen_indices)]

            return chosen_actions, chosen_indices, current_dynamics

    # choosing from output of beam search (without re-compute some intermediate representations)
    def beam_search_choice_act_greedy(self, observation_strings, task_desc_strings, previous_dynamics):
        with torch.no_grad():
            action_candidate_list, current_dynamics, obs_mask, aggregated_obs_representation = self.command_generation_by_beam_search(observation_strings, task_desc_strings, previous_dynamics)
            action_scores, action_masks = self.beam_search_candidate_scoring(action_candidate_list,
                                                                             aggregated_obs_representation,
                                                                             obs_mask,
                                                                             current_dynamics,
                                                                             use_model="online")
            action_indices_maxq = self.choose_maxQ_action(action_scores, action_masks)
            chosen_indices = action_indices_maxq
            chosen_indices = chosen_indices.astype(int)
            chosen_actions = [item[idx] for item, idx in zip(action_candidate_list, chosen_indices)]

            return chosen_actions, chosen_indices, current_dynamics, action_candidate_list

    def beam_search_choice_act_random(self, observation_strings, task_desc_strings, previous_dynamics):
        with torch.no_grad():
            action_candidate_list, current_dynamics, obs_mask, aggregated_obs_representation = self.command_generation_by_beam_search(observation_strings, task_desc_strings, previous_dynamics)
            action_scores, _ = self.beam_search_candidate_scoring(action_candidate_list,
                                                                  aggregated_obs_representation,
                                                                  obs_mask,
                                                                  current_dynamics,
                                                                  use_model="online")
            action_indices_random = self.choose_random_action(action_scores, action_candidate_list)

            chosen_indices = action_indices_random
            chosen_indices = chosen_indices.astype(int)
            chosen_actions = [item[idx] for item, idx in zip(action_candidate_list, chosen_indices)]
            return chosen_actions, chosen_indices, current_dynamics, action_candidate_list

    def beam_search_choice_act(self, observation_strings, task_desc_strings, previous_dynamics, random=False):
        with torch.no_grad():
            if self.mode == "eval":
                return self.beam_search_choice_act_greedy(observation_strings, task_desc_strings, previous_dynamics)
            if random:
                return self.beam_search_choice_act_random(observation_strings, task_desc_strings, previous_dynamics)
            batch_size = len(observation_strings)
            action_candidate_list, current_dynamics, obs_mask, aggregated_obs_representation = self.command_generation_by_beam_search(observation_strings, task_desc_strings, previous_dynamics)
            action_scores, action_masks = self.beam_search_candidate_scoring(action_candidate_list,
                                                                             aggregated_obs_representation,
                                                                             obs_mask,
                                                                             current_dynamics,
                                                                             use_model="online")

            action_indices_maxq = self.choose_maxQ_action(action_scores, action_masks)
            action_indices_random = self.choose_random_action(action_scores, action_candidate_list)

            # random number for epsilon greedy
            rand_num = np.random.uniform(low=0.0, high=1.0, size=(batch_size,))
            less_than_epsilon = (rand_num < self.epsilon).astype("float32")  # batch
            greater_than_epsilon = 1.0 - less_than_epsilon

            chosen_indices = less_than_epsilon * action_indices_random + greater_than_epsilon * action_indices_maxq
            chosen_indices = chosen_indices.astype(int)
            chosen_actions = [item[idx] for item, idx in zip(action_candidate_list, chosen_indices)]

            return chosen_actions, chosen_indices, current_dynamics, action_candidate_list

    # generating token by token
    def command_generation_by_beam_search(self, observation_strings, task_desc_strings, previous_dynamics):
        with torch.no_grad():
            batch_size = len(observation_strings)
            beam_width = self.beam_width
            generate_top_k = self.generate_top_k
            chosen_actions = []

            input_obs = self.get_word_input(observation_strings)
            h_obs, obs_mask = self.encode(observation_strings, use_model="online")
            h_td, td_mask = self.encode(task_desc_strings, use_model="online")
            aggregated_obs_representation = self.online_net.aggretate_information(h_obs, obs_mask, h_td, td_mask)  # batch x obs_length x hid

            if self.recurrent:
                averaged_representation = self.online_net.masked_mean(aggregated_obs_representation, obs_mask)  # batch x hid
                current_dynamics = self.online_net.rnncell(averaged_representation, previous_dynamics) if previous_dynamics is not None else self.online_net.rnncell(averaged_representation)
            else:
                current_dynamics = None

            for b in range(batch_size):
                # starts from CLS tokens
                __input_target_list = [self.word2id["[CLS]"]]
                __input_obs = input_obs[b: b + 1]  # 1 x obs_len
                __obs_mask = obs_mask[b: b + 1]  # 1 x obs_len
                __aggregated_obs_representation = aggregated_obs_representation[b: b + 1]  # 1 x obs_len x hid
                if current_dynamics is not None:
                    __current_dynamics = current_dynamics[b: b + 1]  # 1 x hid
                else:
                    __current_dynamics = None
                ended_nodes = []

                # starting node -  previous node, input target, logp, length
                node = BeamSearchNode(None, __input_target_list, 0, 1)
                nodes_queue = PriorityQueue()
                # start the queue
                nodes_queue.put((node.val, node))
                queue_size = 1
                while(True):
                    # give up when decoding takes too long
                    if queue_size > 2000:
                        break
                    # fetch the best node
                    score, n = nodes_queue.get()
                    __input_target_list = n.input_target

                    if (n.input_target[-1] == self.word2id["[SEP]"] or n.length >= self.max_target_length) and n.previous_node != None:
                        ended_nodes.append((score, n))
                        # if we reached maximum # of sentences required
                        if len(ended_nodes) >= generate_top_k:
                            break
                        else:
                            continue

                    input_target = pad_sequences([__input_target_list], dtype='int32')
                    input_target = to_pt(input_target, self.use_cuda)
                    target_mask = compute_mask(input_target)
                    # decode for one step using decoder
                    pred = self.online_net.decode(input_target, target_mask, __aggregated_obs_representation, __obs_mask, __current_dynamics, __input_obs)  # 1 x target_length x vocab
                    pred = pred[0][-1].cpu()
                    gt_zero = torch.gt(pred, 0.0).float()  # vocab
                    epsilon = torch.le(pred, 0.0).float() * 1e-8  # vocab
                    log_pred = torch.log(pred + epsilon) * gt_zero  # vocab

                    top_beam_width_log_probs, top_beam_width_indicies = torch.topk(log_pred, beam_width)
                    next_nodes = []

                    for new_k in range(beam_width):
                        pos = top_beam_width_indicies[new_k]
                        log_p = top_beam_width_log_probs[new_k].item()
                        node = BeamSearchNode(n, __input_target_list + [pos], n.log_prob + log_p, n.length + 1)
                        next_nodes.append((node.val, node))

                    # put them into queue
                    for i in range(len(next_nodes)):
                        score, nn = next_nodes[i]
                        nodes_queue.put((score, nn))
                    # increase qsize
                    queue_size += len(next_nodes) - 1

                # choose n best paths
                if len(ended_nodes) == 0:
                    ended_nodes = [nodes_queue.get() for _ in range(generate_top_k)]

                utterances = []
                for score, n in sorted(ended_nodes, key=operator.itemgetter(0)):
                    utte = n.input_target
                    utte_string = self.tokenizer.decode(utte)
                    utterances.append(utte_string)

                utterances = [item.replace("[CLS]", "").replace("[SEP]", "").strip() for item in utterances]
                chosen_actions.append(utterances)
            return chosen_actions, current_dynamics, obs_mask, aggregated_obs_representation

    def command_generation_act_greedy(self, observation_strings, task_desc_strings, previous_dynamics):
        with torch.no_grad():
            batch_size = len(observation_strings)

            input_obs = self.get_word_input(observation_strings)
            h_obs, obs_mask = self.encode(observation_strings, use_model="online")
            h_td, td_mask = self.encode(task_desc_strings, use_model="online")
            aggregated_obs_representation = self.online_net.aggretate_information(h_obs, obs_mask, h_td, td_mask)  # batch x obs_length x hid

            if self.recurrent:
                averaged_representation = self.online_net.masked_mean(aggregated_obs_representation, obs_mask)  # batch x hid
                current_dynamics = self.online_net.rnncell(averaged_representation, previous_dynamics) if previous_dynamics is not None else self.online_net.rnncell(averaged_representation)
            else:
                current_dynamics = None

            # greedy generation
            input_target_list = [[self.word2id["[CLS]"]] for i in range(batch_size)]
            eos = np.zeros(batch_size)
            for _ in range(self.max_target_length):

                input_target = copy.deepcopy(input_target_list)
                input_target = pad_sequences(input_target, maxlen=max_len(input_target)).astype('int32')
                input_target = to_pt(input_target, self.use_cuda)
                target_mask = compute_mask(input_target)  # mask of ground truth should be the same
                pred = self.online_net.decode(input_target, target_mask, aggregated_obs_representation, obs_mask, current_dynamics, input_obs)  # batch x target_length x vocab
                # pointer softmax
                pred = to_np(pred[:, -1])  # batch x vocab
                pred = np.argmax(pred, -1)  # batch
                for b in range(batch_size):
                    new_stuff = [pred[b]] if eos[b] == 0 else []
                    input_target_list[b] = input_target_list[b] + new_stuff
                    if pred[b] == self.word2id["[SEP]"]:
                        eos[b] = 1
                if np.sum(eos) == batch_size:
                    break
            chosen_actions = [self.tokenizer.decode(item) for item in input_target_list]
            chosen_actions = [item.replace("[CLS]", "").replace("[SEP]", "").strip() for item in chosen_actions]
            chosen_indices = [item[1:] for item in input_target_list]
            for i in range(len(chosen_indices)):
                if chosen_indices[i][-1] == self.word2id["[SEP]"]:
                    chosen_indices[i] = chosen_indices[i][:-1]
            return chosen_actions, chosen_indices, current_dynamics

    def command_generation_act_random(self, observation_strings, task_desc_strings, previous_dynamics):
        with torch.no_grad():

            batch_size = len(observation_strings)
            beam_width = self.beam_width
            generate_top_k = self.generate_top_k
            chosen_actions, chosen_indices = [], []

            input_obs = self.get_word_input(observation_strings)
            h_obs, obs_mask = self.encode(observation_strings, use_model="online")
            h_td, td_mask = self.encode(task_desc_strings, use_model="online")
            aggregated_obs_representation = self.online_net.aggretate_information(h_obs, obs_mask, h_td, td_mask)  # batch x obs_length x hid

            if self.recurrent:
                averaged_representation = self.online_net.masked_mean(aggregated_obs_representation, obs_mask)  # batch x hid
                current_dynamics = self.online_net.rnncell(averaged_representation, previous_dynamics) if previous_dynamics is not None else self.online_net.rnncell(averaged_representation)
            else:
                current_dynamics = None

            for b in range(batch_size):
                # starts from CLS tokens
                __input_target_list = [self.word2id["[CLS]"]]
                __input_obs = input_obs[b: b + 1]  # 1 x obs_len
                __obs_mask = obs_mask[b: b + 1]  # 1 x obs_len
                __aggregated_obs_representation = aggregated_obs_representation[b: b + 1]  # 1 x obs_len x hid
                if current_dynamics is not None:
                    __current_dynamics = current_dynamics[b: b + 1]  # 1 x hid
                else:
                    __current_dynamics = None
                ended_nodes = []
                # starting node -  previous node, input target, logp, length
                node = BeamSearchNode(None, __input_target_list, 0, 1)
                nodes_queue = PriorityQueue()
                # start the queue
                nodes_queue.put((node.val, node))
                queue_size = 1

                while(True):
                    # give up when decoding takes too long
                    if queue_size > 2000:
                        break
                    # fetch the best node
                    score, n = nodes_queue.get()
                    __input_target_list = n.input_target

                    if (n.input_target[-1] == self.word2id["[SEP]"] or n.length >= self.max_target_length) and n.previous_node != None:
                        ended_nodes.append((score, n))
                        # if we reached maximum # of sentences required
                        if len(ended_nodes) >= generate_top_k:
                            break
                        else:
                            continue

                    input_target = pad_sequences([__input_target_list], dtype='int32')
                    input_target = to_pt(input_target, self.use_cuda)
                    target_mask = compute_mask(input_target)
                    # decode for one step using decoder
                    pred = self.online_net.decode(input_target, target_mask, __aggregated_obs_representation, __obs_mask, __current_dynamics, __input_obs)  # 1 x target_length x vocab
                    pred = pred[0][-1].cpu()
                    gt_zero = torch.gt(pred, 0.0).float()  # vocab
                    epsilon = torch.le(pred, 0.0).float() * 1e-8  # vocab
                    log_pred = torch.log(pred + epsilon) * gt_zero  # vocab

                    top_beam_width_log_probs, top_beam_width_indicies = torch.topk(log_pred, beam_width)
                    next_nodes = []

                    for new_k in range(beam_width):
                        pos = top_beam_width_indicies[new_k]
                        log_p = top_beam_width_log_probs[new_k].item()
                        node = BeamSearchNode(n, __input_target_list + [pos], n.log_prob + log_p, n.length + 1)
                        next_nodes.append((node.val, node))

                    # put them into queue
                    for i in range(len(next_nodes)):
                        score, nn = next_nodes[i]
                        nodes_queue.put((score, nn))
                    # increase qsize
                    queue_size += len(next_nodes) - 1

                # choose n best paths
                if len(ended_nodes) == 0:
                    ended_nodes = [nodes_queue.get() for _ in range(generate_top_k)]

                indicies, utterances = [], []
                for score, n in sorted(ended_nodes, key=operator.itemgetter(0)):
                    utte = n.input_target
                    utte_string = self.tokenizer.decode(utte)
                    utterances.append(utte_string)
                    indicies.append(utte)

                utterances = [item.replace("[CLS]", "").replace("[SEP]", "").strip() for item in utterances]
                indicies = [item[1:] for item in indicies]
                for i in range(len(indicies)):
                    if indicies[i][-1] == self.word2id["[SEP]"]:
                        indicies[i] = indicies[i][:-1]

                # sample one from all generated beams
                rand_idx = np.random.choice(len(indicies))
                chosen_actions.append(utterances[rand_idx])
                chosen_indices.append(indicies[rand_idx])
            return chosen_actions, chosen_indices, current_dynamics

    def command_generation_act(self, observation_strings, task_desc_strings, previous_dynamics, random=False):
        with torch.no_grad():
            if self.mode == "eval":
                return self.command_generation_act_greedy(observation_strings, task_desc_strings, previous_dynamics)
            if random:
                return self.command_generation_act_random(observation_strings, task_desc_strings, previous_dynamics)
            batch_size = len(observation_strings)
            greedy_actions, greedy_indices, greedy_current_dynamics = self.command_generation_act_greedy(observation_strings, task_desc_strings, previous_dynamics)

            # random number for epsilon greedy
            chosen_actions, chosen_indices, current_dynamics = [], [], []
            rand_num = np.random.uniform(low=0.0, high=1.0, size=(batch_size,))
            for b in range(batch_size):
                if rand_num[b] < self.epsilon:
                    # random
                    random_actions, random_indices, random_current_dynamics = self.command_generation_act_random(observation_strings[b: b + 1], task_desc_strings[b: b + 1], None if previous_dynamics is None else previous_dynamics[b: b + 1])
                    chosen_actions.append(random_actions[0])
                    chosen_indices.append(random_indices[0])
                    if self.recurrent:
                        current_dynamics.append(random_current_dynamics[0])
                else:
                    # greedy
                    chosen_actions.append(greedy_actions[b])
                    chosen_indices.append(greedy_indices[b])
                    if self.recurrent:
                        current_dynamics.append(greedy_current_dynamics[b])
            current_dynamics = torch.stack(current_dynamics, 0) if self.recurrent else None  # batch x hidden
            return chosen_actions, chosen_indices, current_dynamics

    # update: admissible commands
    def get_dqn_loss_admissible_commands(self):
        """
        Update neural model in agent. In this example we follow algorithm
        of updating model in dqn with replay memory.
        """
        if len(self.dqn_memory) < self.replay_batch_size:
            return None, None
        data = self.dqn_memory.get_batch(self.replay_batch_size, multi_step=self.multi_step)
        if data is None:
            return None, None

        obs_list, task_list, candidate_list, action_indices, rewards, next_obs_list, next_candidate_list, actual_ns = data
        if self.use_cuda:
            rewards = rewards.cuda()

        h_obs, obs_mask = self.encode(obs_list, use_model="online")
        h_td, td_mask = self.encode(task_list, use_model="online")
        action_scores, _, _ = self.action_scoring(candidate_list,
                                                  h_obs, obs_mask,
                                                  h_td, td_mask,
                                                  None,
                                                  use_model="online")

        # ps_a
        action_indices = to_pt(action_indices, enable_cuda=self.use_cuda, type='long').unsqueeze(-1)
        q_value = ez_gather_dim_1(action_scores, action_indices).squeeze(1)  # batch

        with torch.no_grad():
            if self.noisy_net:
                self.target_net.reset_noise()  # Sample new target net noise
            # pns Probabilities p(s_t+n, ·; θonline)
            h_obs, obs_mask = self.encode(next_obs_list, use_model="online")
            next_action_scores, next_action_masks, _ = self.action_scoring(next_candidate_list,
                                                                           h_obs, obs_mask,
                                                                           h_td.detach(), td_mask.detach(),
                                                                           None,
                                                                           use_model="online")

            # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; θonline))]
            next_action_indices = self.choose_maxQ_action(next_action_scores, next_action_masks)  # batch
            next_action_indices = to_pt(next_action_indices, enable_cuda=self.use_cuda, type='long').unsqueeze(-1)
            # pns # Probabilities p(s_t+n, ·; θtarget)
            h_obs, obs_mask = self.encode(next_obs_list, use_model="target")
            h_td_t, td_mask_t = self.encode(task_list, use_model="target")
            next_action_scores, _, _ = self.action_scoring(next_candidate_list,
                                                           h_obs, obs_mask,
                                                           h_td_t, td_mask_t,
                                                           None,
                                                           use_model="target")

            # pns_a # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)
            next_q_value = ez_gather_dim_1(next_action_scores, next_action_indices).squeeze(1)  # batch
            discount = to_pt((np.ones_like(actual_ns) * self.discount_gamma_game_reward) ** actual_ns, self.use_cuda, type="float")

        rewards = rewards + next_q_value * discount  # batch
        loss = F.smooth_l1_loss(q_value, rewards)
        return loss, q_value

    def get_drqn_loss_admissible_commands(self):
        """
        Update neural model in agent. In this example we follow algorithm
        of updating model in dqn with replay memory.
        """
        if len(self.dqn_memory) < self.replay_batch_size:
            return None, None
        data, contains_first_step = self.dqn_memory.get_batch_of_sequences(self.replay_batch_size, sample_history_length=self.rl_replay_sample_history_length)
        if data is None:
            return None, None

        seq_obs, task, seq_candidates, seq_chosen_indices, seq_reward, seq_next_obs, seq_next_candidates = data
        loss_list, q_value_list = [], []
        prev_dynamics = None

        h_td, td_mask = self.encode(task, use_model="online")
        with torch.no_grad():
            h_td_t, td_mask_t = self.encode(task, use_model="target")

        for step_no in range(self.rl_replay_sample_history_length):
            obs, candidates, chosen_indices, reward, next_obs, next_candidates = seq_obs[step_no], seq_candidates[step_no], seq_chosen_indices[step_no], seq_reward[step_no], seq_next_obs[step_no], seq_next_candidates[step_no]
            if self.use_cuda:
                reward = reward.cuda()

            h_obs, obs_mask = self.encode(obs, use_model="online")
            action_scores, _, current_dynamics = self.action_scoring(candidates, h_obs, obs_mask, h_td, td_mask,
                                                                     prev_dynamics, use_model="online")
            # ps_a
            chosen_indices = to_pt(chosen_indices, enable_cuda=self.use_cuda, type='long').unsqueeze(-1)
            q_value = ez_gather_dim_1(action_scores, chosen_indices).squeeze(1)  # batch

            prev_dynamics = current_dynamics
            if (not contains_first_step) and step_no < self.rl_replay_sample_update_from:
                q_value = q_value.detach()
                prev_dynamics = prev_dynamics.detach()
                continue

            with torch.no_grad():
                if self.noisy_net:
                    self.target_net.reset_noise()  # Sample new target net noise
                # pns Probabilities p(s_t+n, ·; θonline)

                h_obs, obs_mask = self.encode(next_obs, use_model="online")
                next_action_scores, next_action_masks, _ = self.action_scoring(next_candidates, h_obs, obs_mask, h_td, td_mask,
                                                                               prev_dynamics, use_model="online")

                # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; θonline))]
                next_action_indices = self.choose_maxQ_action(next_action_scores, next_action_masks)  # batch
                next_action_indices = to_pt(next_action_indices, enable_cuda=self.use_cuda, type='long').unsqueeze(-1)

                # pns # Probabilities p(s_t+n, ·; θtarget)
                h_obs, obs_mask = self.encode(next_obs, use_model="target")
                next_action_scores, _, _ = self.action_scoring(next_candidates, h_obs, obs_mask, h_td_t, td_mask_t,
                                                               prev_dynamics, use_model="target")

                # pns_a # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)
                next_q_value = ez_gather_dim_1(next_action_scores, next_action_indices).squeeze(1)  # batch

            reward = reward + next_q_value * self.discount_gamma_game_reward  # batch
            loss = F.smooth_l1_loss(q_value, reward)  # 1
            loss_list.append(loss)
            q_value_list.append(q_value)

        loss = torch.stack(loss_list).mean()
        q_value = torch.stack(q_value_list).mean()

        return loss, q_value

    def update_dqn_admissible_commands(self):
        # update neural model by replaying snapshots in replay memory
        if self.recurrent:
            dqn_loss, q_value = self.get_drqn_loss_admissible_commands()
        else:
            dqn_loss, q_value = self.get_dqn_loss_admissible_commands()

        if dqn_loss is None:
            return None, None
        # Backpropagate
        self.online_net.zero_grad()
        self.optimizer.zero_grad()
        dqn_loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.clip_grad_norm)
        self.optimizer.step()  # apply gradients
        return to_np(torch.mean(dqn_loss)), to_np(torch.mean(q_value))

    # update: beam search choice
    def get_dqn_loss_beam_search_choice(self):
        """
        Update neural model in agent. In this example we follow algorithm
        of updating model in dqn with replay memory.
        """
        if len(self.dqn_memory) < self.replay_batch_size:
            return None, None
        data = self.dqn_memory.get_batch(self.replay_batch_size, multi_step=self.multi_step)
        if data is None:
            return None, None

        obs_list, task_list, candidate_list, action_indices, rewards, next_obs_list, next_candidate_list, actual_ns = data
        if self.use_cuda:
            rewards = rewards.cuda()

        with torch.no_grad():
            h_obs, obs_mask = self.encode(obs_list, use_model="online")
            h_td, td_mask = self.encode(task_list, use_model="online")
            aggregated_obs_representation = self.online_net.aggretate_information(h_obs, obs_mask, h_td, td_mask)  # batch x obs_length x hid
        action_scores, _ = self.beam_search_candidate_scoring(candidate_list, aggregated_obs_representation, obs_mask, None, use_model="online")

        # ps_a
        action_indices = to_pt(action_indices, enable_cuda=self.use_cuda, type='long').unsqueeze(-1)
        q_value = ez_gather_dim_1(action_scores, action_indices).squeeze(1)  # batch

        with torch.no_grad():
            if self.noisy_net:
                self.target_net.reset_noise()  # Sample new target net noise
            # pns Probabilities p(s_t+n, ·; θonline)
            h_obs, obs_mask = self.encode(next_obs_list, use_model="online")
            aggregated_obs_representation = self.online_net.aggretate_information(h_obs, obs_mask, h_td, td_mask)  # batch x obs_length x hid
            next_action_scores, next_action_masks = self.beam_search_candidate_scoring(next_candidate_list, aggregated_obs_representation, obs_mask, None, use_model="online")

            # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; θonline))]
            next_action_indices = self.choose_maxQ_action(next_action_scores, next_action_masks)  # batch
            next_action_indices = to_pt(next_action_indices, enable_cuda=self.use_cuda, type='long').unsqueeze(-1)
            # pns # Probabilities p(s_t+n, ·; θtarget)
            h_obs, obs_mask = self.encode(next_obs_list, use_model="target")
            h_td_t, td_mask_t = self.encode(task_list, use_model="target")
            aggregated_obs_representation = self.target_net.aggretate_information(h_obs, obs_mask, h_td_t, td_mask_t)  # batch x obs_length x hid
            next_action_scores, _ = self.beam_search_candidate_scoring(next_candidate_list, aggregated_obs_representation, obs_mask, None, use_model="target")

            # pns_a # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)
            next_q_value = ez_gather_dim_1(next_action_scores, next_action_indices).squeeze(1)  # batch
            discount = to_pt((np.ones_like(actual_ns) * self.discount_gamma_game_reward) ** actual_ns, self.use_cuda, type="float")

        rewards = rewards + next_q_value * discount  # batch
        loss = F.smooth_l1_loss(q_value, rewards)
        return loss, q_value

    def get_drqn_loss_beam_search_choice(self):
        """
        Update neural model in agent. In this example we follow algorithm
        of updating model in dqn with replay memory.
        """
        if len(self.dqn_memory) < self.replay_batch_size:
            return None, None
        data, contains_first_step = self.dqn_memory.get_batch_of_sequences(self.replay_batch_size, sample_history_length=self.rl_replay_sample_history_length)
        if data is None:
            return None, None

        seq_obs, task, seq_candidates, seq_chosen_indices, seq_reward, seq_next_obs, seq_next_candidates = data
        loss_list, q_value_list = [], []
        prev_dynamics = None

        with torch.no_grad():
            h_td, td_mask = self.encode(task, use_model="online")
            h_td_t, td_mask_t = self.encode(task, use_model="target")

        for step_no in range(self.rl_replay_sample_history_length):
            obs, candidates, chosen_indices, reward, next_obs, next_candidates = seq_obs[step_no], seq_candidates[step_no], seq_chosen_indices[step_no], seq_reward[step_no], seq_next_obs[step_no], seq_next_candidates[step_no]
            if self.use_cuda:
                reward = reward.cuda()

            with torch.no_grad():
                h_obs, obs_mask = self.encode(obs, use_model="online")
                aggregated_obs_representation = self.online_net.aggretate_information(h_obs, obs_mask, h_td, td_mask)  # batch x obs_length x hid
                averaged_representation = self.online_net.masked_mean(aggregated_obs_representation, obs_mask)  # batch x hid
                current_dynamics = self.online_net.rnncell(averaged_representation, prev_dynamics) if prev_dynamics is not None else self.online_net.rnncell(averaged_representation)
            action_scores, _ = self.beam_search_candidate_scoring(candidates, aggregated_obs_representation, obs_mask, current_dynamics, use_model="online")
            # ps_a
            chosen_indices = to_pt(chosen_indices, enable_cuda=self.use_cuda, type='long').unsqueeze(-1)
            q_value = ez_gather_dim_1(action_scores, chosen_indices).squeeze(1)  # batch

            prev_dynamics = current_dynamics
            if (not contains_first_step) and step_no < self.rl_replay_sample_update_from:
                q_value = q_value.detach()
                prev_dynamics = prev_dynamics.detach()
                continue

            with torch.no_grad():
                if self.noisy_net:
                    self.target_net.reset_noise()  # Sample new target net noise
                # pns Probabilities p(s_t+n, ·; θonline)

                h_obs, obs_mask = self.encode(next_obs, use_model="online")
                aggregated_obs_representation = self.online_net.aggretate_information(h_obs, obs_mask, h_td, td_mask)  # batch x obs_length x hid
                averaged_representation = self.online_net.masked_mean(aggregated_obs_representation, obs_mask)  # batch x hid
                next_dynamics = self.online_net.rnncell(averaged_representation, current_dynamics) if current_dynamics is not None else self.online_net.rnncell(averaged_representation)
                next_action_scores, next_action_masks = self.beam_search_candidate_scoring(next_candidates, aggregated_obs_representation, obs_mask, next_dynamics, use_model="online")

                # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; θonline))]
                next_action_indices = self.choose_maxQ_action(next_action_scores, next_action_masks)  # batch
                next_action_indices = to_pt(next_action_indices, enable_cuda=self.use_cuda, type='long').unsqueeze(-1)

                # pns # Probabilities p(s_t+n, ·; θtarget)
                h_obs, obs_mask = self.encode(next_obs, use_model="target")
                aggregated_obs_representation = self.target_net.aggretate_information(h_obs, obs_mask, h_td_t, td_mask_t)  # batch x obs_length x hid
                averaged_representation = self.target_net.masked_mean(aggregated_obs_representation, obs_mask)  # batch x hid
                next_dynamics = self.target_net.rnncell(averaged_representation, current_dynamics) if current_dynamics is not None else self.target_net.rnncell(averaged_representation)
                next_action_scores, _ = self.beam_search_candidate_scoring(next_candidates, aggregated_obs_representation, obs_mask, next_dynamics, use_model="target")

                # pns_a # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)
                next_q_value = ez_gather_dim_1(next_action_scores, next_action_indices).squeeze(1)  # batch

            reward = reward + next_q_value * self.discount_gamma_game_reward  # batch
            loss = F.smooth_l1_loss(q_value, reward)  # 1
            loss_list.append(loss)
            q_value_list.append(q_value)

        loss = torch.stack(loss_list).mean()
        q_value = torch.stack(q_value_list).mean()

        return loss, q_value

    def update_dqn_beam_search_choice(self):
        # update neural model by replaying snapshots in replay memory
        if self.recurrent:
            dqn_loss, q_value = self.get_drqn_loss_beam_search_choice()
        else:
            dqn_loss, q_value = self.get_dqn_loss_beam_search_choice()

        if dqn_loss is None:
            return None, None
        # Backpropagate
        self.online_net.zero_grad()
        self.optimizer.zero_grad()
        dqn_loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.clip_grad_norm)
        self.optimizer.step()  # apply gradients
        return to_np(torch.mean(dqn_loss)), to_np(torch.mean(q_value))

    # update: command generation
    def get_dqn_loss_command_generation(self):
        """
        Update neural model in agent. In this example we follow algorithm
        of updating model in dqn with replay memory.
        """
        if len(self.dqn_memory) < self.replay_batch_size:
            return None, None
        data = self.dqn_memory.get_batch(self.replay_batch_size, multi_step=self.multi_step)
        if data is None:
            return None, None

        observation_strings, task_desc_strings, _, action_indices, rewards, next_observation_strings, _, actual_ns = data
        batch_size = len(observation_strings)
        if self.use_cuda:
            rewards = rewards.cuda()

        input_target = [[self.word2id["[CLS]"]] + item for item in action_indices]
        ground_truth = [item + [self.word2id["[SEP]"]] for item in action_indices]
        input_target = self.get_word_input_from_ids(input_target)
        ground_truth = self.get_word_input_from_ids(ground_truth)

        input_obs = self.get_word_input(observation_strings)
        next_input_obs = self.get_word_input(next_observation_strings)
        h_obs, obs_mask = self.encode(observation_strings, use_model="online")
        h_td, td_mask = self.encode(task_desc_strings, use_model="online")
        aggregated_obs_representation = self.online_net.aggretate_information(h_obs, obs_mask, h_td, td_mask)  # batch x obs_length x hid

        target_mask = compute_mask(input_target)  # mask of ground truth should be the same
        pred = self.online_net.decode(input_target, target_mask, aggregated_obs_representation, obs_mask, None, input_obs)  # batch x target_length x vocab
        q_value = GetGenerationQValue(pred * target_mask.unsqueeze(-1), ground_truth, target_mask)

        with torch.no_grad():
            if self.noisy_net:
                self.target_net.reset_noise()  # Sample new target net noise
            # pns Probabilities p(s_t+n, ·; θonline)
            next_h_obs, next_obs_mask = self.encode(next_observation_strings, use_model="online")
            next_aggregated_obs_representation = self.online_net.aggretate_information(next_h_obs, next_obs_mask, h_td, td_mask)  # batch x obs_length x hid

            # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; θonline))]
            # greedy generation
            input_target_list = [[self.word2id["[CLS]"]] for i in range(batch_size)]
            eos = np.zeros(batch_size)
            for _ in range(self.max_target_length):
                input_target = copy.deepcopy(input_target_list)
                input_target = pad_sequences(input_target, maxlen=max_len(input_target)).astype('int32')
                input_target = to_pt(input_target, self.use_cuda)
                target_mask = compute_mask(input_target)  # mask of ground truth should be the same
                pred = self.online_net.decode(input_target, target_mask, next_aggregated_obs_representation, next_obs_mask, None, next_input_obs)  # batch x target_length x vocab
                # pointer softmax
                pred = to_np(pred[:, -1])  # batch x vocab
                pred = np.argmax(pred, -1)  # batch
                for b in range(batch_size):
                    new_stuff = [pred[b]] if eos[b] == 0 else []
                    input_target_list[b] = input_target_list[b] + new_stuff
                    if pred[b] == self.word2id["[SEP]"]:
                        eos[b] = 1
                if np.sum(eos) == batch_size:
                    break
            chosen_indices = [item[1:] for item in input_target_list]
            for i in range(len(chosen_indices)):
                if chosen_indices[i][-1] == self.word2id["[SEP]"]:
                    chosen_indices[i] = chosen_indices[i][:-1]

            # pns # Probabilities p(s_t+n, ·; θtarget)
            next_input_target = [[self.word2id["[CLS]"]] + item for item in chosen_indices]
            next_ground_truth = [item + [self.word2id["[SEP]"]] for item in chosen_indices]
            next_input_target = self.get_word_input_from_ids(next_input_target)
            next_ground_truth = self.get_word_input_from_ids(next_ground_truth)

            next_h_obs, next_obs_mask = self.encode(next_observation_strings, use_model="target")
            next_h_td, next_td_mask = self.encode(task_desc_strings, use_model="target")
            next_aggregated_obs_representation = self.target_net.aggretate_information(next_h_obs, next_obs_mask, next_h_td, next_td_mask)  # batch x obs_length x hid

            next_target_mask = compute_mask(next_input_target)  # mask of ground truth should be the same
            next_pred = self.target_net.decode(next_input_target, next_target_mask, next_aggregated_obs_representation, next_obs_mask, None, next_input_obs)  # batch x target_length x vocab
            next_q_value = GetGenerationQValue(next_pred * next_target_mask.unsqueeze(-1), next_ground_truth, next_target_mask)  # batch

            discount = to_pt((np.ones_like(actual_ns) * self.discount_gamma_game_reward) ** actual_ns, self.use_cuda, type="float")

        rewards = rewards + next_q_value * discount  # batch
        loss = F.smooth_l1_loss(q_value, rewards)
        return loss, q_value

    def get_drqn_loss_command_generation(self):
        if len(self.dqn_memory) < self.replay_batch_size:
            return None, None
        data, contains_first_step = self.dqn_memory.get_batch_of_sequences(self.replay_batch_size, sample_history_length=self.rl_replay_sample_history_length)
        if data is None:
            return None, None

        seq_obs, task_desc_strings, _, seq_chosen_indices, seq_reward, seq_next_obs, _ = data
        batch_size = len(seq_obs[0])
        loss_list, q_value_list = [], []
        previous_dynamics = None

        h_td, td_mask = self.encode(task_desc_strings, use_model="online")
        with torch.no_grad():
            h_td_t, td_mask_t = self.encode(task_desc_strings, use_model="target")

        for step_no in range(self.rl_replay_sample_history_length):
            observation_strings, action_indices, reward, next_observation_strings = seq_obs[step_no], seq_chosen_indices[step_no], seq_reward[step_no], seq_next_obs[step_no]
            if self.use_cuda:
                reward = reward.cuda()

            input_target = [[self.word2id["[CLS]"]] + item for item in action_indices]
            ground_truth = [item + [self.word2id["[SEP]"]] for item in action_indices]
            input_target = self.get_word_input_from_ids(input_target)
            ground_truth = self.get_word_input_from_ids(ground_truth)

            input_obs = self.get_word_input(observation_strings)
            next_input_obs = self.get_word_input(next_observation_strings)
            h_obs, obs_mask = self.encode(observation_strings, use_model="online")
            aggregated_obs_representation = self.online_net.aggretate_information(h_obs, obs_mask, h_td, td_mask)  # batch x obs_length x hid

            averaged_representation = self.online_net.masked_mean(aggregated_obs_representation, obs_mask)  # batch x hid
            current_dynamics = self.online_net.rnncell(averaged_representation, previous_dynamics) if previous_dynamics is not None else self.online_net.rnncell(averaged_representation)

            target_mask = compute_mask(input_target)  # mask of ground truth should be the same
            pred = self.online_net.decode(input_target, target_mask, aggregated_obs_representation, obs_mask, current_dynamics, input_obs)  # batch x target_length x vocab
            q_value = GetGenerationQValue(pred * target_mask.unsqueeze(-1), ground_truth, target_mask)

            previous_dynamics = current_dynamics
            if (not contains_first_step) and step_no < self.rl_replay_sample_update_from:
                q_value = q_value.detach()
                previous_dynamics = previous_dynamics.detach()
                continue

            with torch.no_grad():
                if self.noisy_net:
                    self.target_net.reset_noise()  # Sample new target net noise
                # pns Probabilities p(s_t+n, ·; θonline)
                next_h_obs, next_obs_mask = self.encode(next_observation_strings, use_model="online")
                next_aggregated_obs_representation = self.online_net.aggretate_information(next_h_obs, next_obs_mask, h_td, td_mask)  # batch x obs_length x hid
                next_averaged_representation = self.online_net.masked_mean(next_aggregated_obs_representation, next_obs_mask)  # batch x hid
                next_dynamics = self.online_net.rnncell(averaged_representation, current_dynamics) if current_dynamics is not None else self.online_net.rnncell(next_averaged_representation)

                # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; θonline))]
                # greedy generation
                input_target_list = [[self.word2id["[CLS]"]] for i in range(batch_size)]
                eos = np.zeros(batch_size)
                for _ in range(self.max_target_length):
                    input_target = copy.deepcopy(input_target_list)
                    input_target = pad_sequences(input_target, maxlen=max_len(input_target)).astype('int32')
                    input_target = to_pt(input_target, self.use_cuda)
                    target_mask = compute_mask(input_target)  # mask of ground truth should be the same
                    pred = self.online_net.decode(input_target, target_mask, next_aggregated_obs_representation, next_obs_mask, next_dynamics, next_input_obs)  # batch x target_length x vocab
                    # pointer softmax
                    pred = to_np(pred[:, -1])  # batch x vocab
                    pred = np.argmax(pred, -1)  # batch
                    for b in range(batch_size):
                        new_stuff = [pred[b]] if eos[b] == 0 else []
                        input_target_list[b] = input_target_list[b] + new_stuff
                        if pred[b] == self.word2id["[SEP]"]:
                            eos[b] = 1
                    if np.sum(eos) == batch_size:
                        break
                chosen_indices = [item[1:] for item in input_target_list]
                for i in range(len(chosen_indices)):
                    if chosen_indices[i][-1] == self.word2id["[SEP]"]:
                        chosen_indices[i] = chosen_indices[i][:-1]

                # pns # Probabilities p(s_t+n, ·; θtarget)
                next_input_target = [[self.word2id["[CLS]"]] + item for item in chosen_indices]
                next_ground_truth = [item + [self.word2id["[SEP]"]] for item in chosen_indices]
                next_input_target = self.get_word_input_from_ids(next_input_target)
                next_ground_truth = self.get_word_input_from_ids(next_ground_truth)

                next_h_obs, next_obs_mask = self.encode(next_observation_strings, use_model="target")
                next_aggregated_obs_representation = self.target_net.aggretate_information(next_h_obs, next_obs_mask, h_td_t, td_mask_t)  # batch x obs_length x hid
                next_averaged_representation = self.target_net.masked_mean(next_aggregated_obs_representation, next_obs_mask)  # batch x hid
                next_dynamics = self.target_net.rnncell(averaged_representation, current_dynamics) if current_dynamics is not None else self.target_net.rnncell(next_averaged_representation)

                next_target_mask = compute_mask(next_input_target)  # mask of ground truth should be the same
                next_pred = self.target_net.decode(next_input_target, next_target_mask, next_aggregated_obs_representation, next_obs_mask, next_dynamics, next_input_obs)  # batch x target_length x vocab
                next_q_value = GetGenerationQValue(next_pred * next_target_mask.unsqueeze(-1), next_ground_truth, next_target_mask)  # batch

            reward = reward + next_q_value * self.discount_gamma_game_reward  # batch
            loss = F.smooth_l1_loss(q_value, reward)  # 1
            loss_list.append(loss)
            q_value_list.append(q_value)

        loss = torch.stack(loss_list).mean()
        q_value = torch.stack(q_value_list).mean()
        return loss, q_value

    def update_dqn_command_generation(self):
        # update neural model by replaying snapshots in replay memory
        if self.recurrent:
            dqn_loss, q_value = self.get_drqn_loss_command_generation()
        else:
            dqn_loss, q_value = self.get_dqn_loss_command_generation()

        if dqn_loss is None:
            return None, None
        # Backpropagate
        self.online_net.zero_grad()
        self.optimizer.zero_grad()
        dqn_loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.clip_grad_norm)
        self.optimizer.step()  # apply gradients
        return to_np(torch.mean(dqn_loss)), to_np(torch.mean(q_value))

    def update_dqn(self):
        if self.action_space == "generation":
            return self.update_dqn_command_generation()
        elif self.action_space == "beam_search_choice":
            return self.update_dqn_beam_search_choice()
        elif self.action_space in ["admissible", "exhaustive"]:
            return self.update_dqn_admissible_commands()
        else:
            raise NotImplementedError()