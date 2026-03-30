# part of the code are from https://github.com/hill-a/stable-baselines/
import random
from collections import namedtuple
import numpy as np
import torch


# a snapshot of state to be stored in replay memory
Transition = namedtuple('Transition', ('observation_list', 'task_list', 'action_candidate_list', 'chosen_indices', 'reward', 'count_reward', 'novel_object_reward'))
# a snapshot of state to be stored in replay memory for question answering
dagger_transition = namedtuple('dagger_transition', ('observation_list', 'task_list', 'action_candidate_list', 'target_list', 'target_indices'))


class PrioritizedReplayMemory(object):

    def __init__(self, capacity=100000, priority_fraction=0.0,
                 discount_gamma_game_reward=1.0, discount_gamma_count_reward=0.0, discount_gamma_novel_object_reward=0.0,
                 accumulate_reward_from_final=False):
        # prioritized replay memory
        self.priority_fraction = priority_fraction
        self.alpha_capacity = int(capacity * priority_fraction)
        self.beta_capacity = capacity - self.alpha_capacity
        self.alpha_memory, self.beta_memory = [], []
        self.alpha_rewards, self.beta_rewards = [], []
        self.accumulate_reward_from_final = accumulate_reward_from_final
        self.discount_gamma_game_reward = discount_gamma_game_reward
        self.discount_gamma_count_reward = discount_gamma_count_reward
        self.discount_gamma_novel_object_reward = discount_gamma_novel_object_reward

    def push(self, is_prior, reward, t):
        """Saves a transition."""
        if self.priority_fraction == 0.0:
            is_prior = False
        trajectory = []
        for i in range(len(t)):
            trajectory.append(Transition(t[i][0], t[i][1], t[i][2], t[i][3], t[i][4], t[i][5], t[i][6]))
        if is_prior:
            self.alpha_memory.append(trajectory)
            self.alpha_rewards.append(reward)
            if len(self.alpha_memory) > self.alpha_capacity:
                remove_id = np.random.randint(self.alpha_capacity)
                self.alpha_memory = self.alpha_memory[:remove_id] + self.alpha_memory[remove_id + 1:]
                self.alpha_rewards = self.alpha_rewards[:remove_id] + self.alpha_rewards[remove_id + 1:]
        else:
            self.beta_memory.append(trajectory)
            self.beta_rewards.append(reward)
            if len(self.beta_memory) > self.beta_capacity:
                remove_id = np.random.randint(self.beta_capacity)
                self.beta_memory = self.beta_memory[:remove_id] + self.beta_memory[remove_id + 1:]
                self.beta_rewards = self.beta_rewards[:remove_id] + self.beta_rewards[remove_id + 1:]

    def _get_single_transition(self, n, which_memory):
        if len(which_memory) == 0:
            return None
        assert n > 0
        trajectory_id = np.random.randint(len(which_memory))
        trajectory = which_memory[trajectory_id]

        if len(trajectory) <= n:
            return None
        head = np.random.randint(0, len(trajectory) - n)
        final = len(trajectory) - 1

        # all good
        obs = trajectory[head].observation_list
        task = trajectory[head].task_list
        candidates = trajectory[head].action_candidate_list
        chosen_indices = trajectory[head].chosen_indices

        next_obs = trajectory[head + n].observation_list
        next_candidates = trajectory[head + n].action_candidate_list

        # 1 2 [3] 4 5 (6) 7 8 9f
        how_long = final - head + 1 if self.accumulate_reward_from_final else n + 1
        accumulated_rewards = [self.discount_gamma_game_reward ** i * trajectory[head + i].reward for i in range(how_long)]
        accumulated_rewards = accumulated_rewards[:n + 1]
        game_reward = torch.sum(torch.stack(accumulated_rewards))

        accumulated_count_rewards = [self.discount_gamma_count_reward ** i * trajectory[head + i].count_reward for i in range(n + 1)]
        accumulated_count_rewards = accumulated_count_rewards[:n + 1]
        count_reward = torch.sum(torch.stack(accumulated_count_rewards))
        
        accumulated_novel_object_rewards = [self.discount_gamma_novel_object_reward ** i * trajectory[head + i].novel_object_reward for i in range(n + 1)]
        accumulated_novel_object_rewards = accumulated_novel_object_rewards[:n + 1]
        novel_object_reward = torch.sum(torch.stack(accumulated_novel_object_rewards))

        return (obs, task, candidates, chosen_indices, game_reward + count_reward + novel_object_reward, next_obs, next_candidates, n)

    def _get_batch(self, n_list, which_memory):
        res = []
        for i in range(len(n_list)):
            output = self._get_single_transition(n_list[i], which_memory)
            if output is None:
                continue
            res.append(output)
        if len(res) == 0:
            return None
        return res

    def get_batch(self, batch_size, multi_step=1):
        from_alpha = min(int(self.priority_fraction * batch_size), len(self.alpha_memory))
        from_beta = min(batch_size - from_alpha, len(self.beta_memory))
        res = []
        if from_alpha == 0:
            res_alpha = None
        else:
            res_alpha = self._get_batch(np.random.randint(1, multi_step + 1, size=from_alpha), self.alpha_memory)
        if from_beta == 0:
            res_beta = None
        else:
            res_beta = self._get_batch(np.random.randint(1, multi_step + 1, size=from_beta), self.beta_memory)
        if res_alpha is None and res_beta is None:
            return None
        if res_alpha is not None:
            res += res_alpha
        if res_beta is not None:
            res += res_beta
        random.shuffle(res)

        obs_list, task_list, action_candidate_list, chosen_indices_list, reward_list, actual_n_list = [], [], [], [], [], []
        next_obs_list, next_action_candidate_list = [], []

        for item in res:
            obs, task, candidates, chosen_indices, reward, next_obs, next_candidates, n = item
            obs_list.append(obs)
            task_list.append(task)
            action_candidate_list.append(candidates)
            chosen_indices_list.append(chosen_indices)
            reward_list.append(reward)
            next_obs_list.append(next_obs)
            next_action_candidate_list.append(next_candidates)
            actual_n_list.append(n)

        chosen_indices_list = np.array(chosen_indices_list)  # batch
        reward_list = torch.stack(reward_list, 0)  # batch
        actual_n_list = np.array(actual_n_list)

        return [obs_list, task_list, action_candidate_list, chosen_indices_list, reward_list, next_obs_list, next_action_candidate_list, actual_n_list]

    def _get_single_sequence_transition(self, which_memory, sample_history_length, contains_first_step):
        if len(which_memory) == 0:
            return None
        assert sample_history_length > 1
        trajectory_id = np.random.randint(len(which_memory))
        trajectory = which_memory[trajectory_id]

        if len(trajectory) <= sample_history_length:
            return None
        
        # 0 1 2 3 4 5 6 7
        if contains_first_step:
            head = 0
        else:
            if 1 >= len(trajectory) - sample_history_length:
                return None
            head = np.random.randint(1, len(trajectory) - sample_history_length)
        # tail = head + sample_history_length - 1
        final = len(trajectory) - 1

        seq_obs, seq_candidates, seq_chosen_indices, seq_reward, seq_next_obs, seq_next_candidates = [], [], [], [], [], []
        task = trajectory[head].task_list
        for j in range(sample_history_length):
            seq_obs.append(trajectory[head + j].observation_list)
            seq_candidates.append(trajectory[head + j].action_candidate_list)
            seq_chosen_indices.append(trajectory[head + j].chosen_indices)
            seq_next_obs.append(trajectory[head + j + 1].observation_list)
            seq_next_candidates.append(trajectory[head + j + 1].action_candidate_list)
            
            how_long = final - (head + j) + 1 if self.accumulate_reward_from_final else 1
            accumulated_rewards = [self.discount_gamma_game_reward ** i * trajectory[head + j + i].reward for i in range(how_long)]
            accumulated_rewards = accumulated_rewards[:1]
            game_reward = torch.sum(torch.stack(accumulated_rewards))
            
            accumulated_count_rewards = [self.discount_gamma_count_reward ** i * trajectory[head + j + i].count_reward for i in range(how_long)]
            accumulated_count_rewards = accumulated_count_rewards[:1]
            count_reward = torch.sum(torch.stack(accumulated_count_rewards))
            
            accumulated_novel_object_rewards = [self.discount_gamma_novel_object_reward ** i * trajectory[head + j + i].novel_object_reward for i in range(how_long)]
            accumulated_novel_object_rewards = accumulated_novel_object_rewards[:1]
            novel_object_reward = torch.sum(torch.stack(accumulated_novel_object_rewards))
            seq_reward.append(game_reward + count_reward + novel_object_reward)

        return [seq_obs, seq_candidates, seq_chosen_indices, seq_reward, seq_next_obs, seq_next_candidates, task]

    def _get_batch_of_sequences(self, which_memory, batch_size, sample_history_length, contains_first_step):
        assert sample_history_length > 1
        
        obs, task, candidates, chosen_indices, reward, next_obs, next_candidates = [], [], [], [], [], [], []
        for _ in range(sample_history_length):
            obs.append([])
            candidates.append([])
            chosen_indices.append([])
            reward.append([])
            next_obs.append([])
            next_candidates.append([])

        # obs, candidate, chosen_indices, graph_triplets, reward, next_obs, next_candidate, next_graph_triplets
        for _ in range(batch_size):
            t = self._get_single_sequence_transition(which_memory, sample_history_length, contains_first_step)
            if t is None:
                continue
            task.append(t[6])
            for step in range(sample_history_length):
                obs[step].append(t[0][step])
                candidates[step].append(t[1][step])
                chosen_indices[step].append(t[2][step])
                reward[step].append(t[3][step])
                next_obs[step].append(t[4][step])
                next_candidates[step].append(t[5][step])

        if len(task) == 0:
            return None
        
        return [obs, task, candidates, chosen_indices, reward, next_obs, next_candidates]

    def get_batch_of_sequences(self, batch_size, sample_history_length):

        from_alpha = min(int(self.priority_fraction * batch_size), len(self.alpha_memory))
        from_beta = min(batch_size - from_alpha, len(self.beta_memory))

        random_number = np.random.uniform(low=0.0, high=1.0, size=(1,))
        contains_first_step = random_number[0] < 0.05  # hard coded here. So 5% of the sampled batches will have first step

        if from_alpha == 0:
            res_alpha = None
        else:
            res_alpha = self._get_batch_of_sequences(self.alpha_memory, from_alpha, sample_history_length, contains_first_step)
        if from_beta == 0:
            res_beta = None
        else:
            res_beta = self._get_batch_of_sequences(self.beta_memory, from_beta, sample_history_length, contains_first_step)
        if res_alpha is None and res_beta is None:
            return None

        obs, task, candidates, chosen_indices, reward, next_obs, next_candidates = [], [], [], [], [], [], []
        for _ in range(sample_history_length):
            obs.append([])
            candidates.append([])
            chosen_indices.append([])
            reward.append([])
            next_obs.append([])
            next_candidates.append([])

        if res_alpha is not None:
            __obs, __task, __candidates, __chosen_indices, __reward, __next_obs, __next_candidates = res_alpha
            task += __task
            for i in range(sample_history_length):
                obs[i] += __obs[i]
                candidates[i] += __candidates[i]
                chosen_indices[i] += __chosen_indices[i]
                reward[i] += __reward[i]
                next_obs[i] += __next_obs[i]
                next_candidates[i] += __next_candidates[i]

        if res_beta is not None:
            __obs, __task, __candidates, __chosen_indices, __reward, __next_obs, __next_candidates = res_beta
            task += __task
            for i in range(sample_history_length):
                obs[i] += __obs[i]
                candidates[i] += __candidates[i]
                chosen_indices[i] += __chosen_indices[i]
                reward[i] += __reward[i]
                next_obs[i] += __next_obs[i]
                next_candidates[i] += __next_candidates[i]

        for i in range(sample_history_length):
            reward[i] = torch.stack(reward[i], 0)  # batch
            chosen_indices[i] = np.array(chosen_indices[i])  # batch

        return [obs, task, candidates, chosen_indices, reward, next_obs, next_candidates], contains_first_step

    def get_avg_rewards(self):
        if len(self.alpha_rewards) == 0 and len(self.beta_rewards) == 0 :
            return 0.0
        return np.mean(self.alpha_rewards + self.beta_rewards)

    def __len__(self):
        return len(self.alpha_memory) + len(self.beta_memory)


class DaggerReplayMemory(object):

    def __init__(self, capacity=100000):
        # replay memory
        self.capacity = capacity
        self.memory = []

    def push(self, t):
        """Saves a transition."""

        trajectory = []
        for i in range(len(t)):
            trajectory.append(dagger_transition(t[i][0], t[i][1], t[i][2], t[i][3], t[i][4]))
        self.memory.append(trajectory)
        if len(self.memory) > self.capacity:
            remove_id = np.random.randint(self.capacity)
            self.memory = self.memory[:remove_id] + self.memory[remove_id + 1:]

    def sample(self, batch_size):
        how_many = min(batch_size, len(self.memory))
        res = []
        for _ in range(how_many):
            trajectory_id = np.random.randint(len(self.memory))
            trajectory = self.memory[trajectory_id]
            head = np.random.randint(0, len(trajectory))
            res.append(trajectory[head])
        return res

    def sample_sequence(self, batch_size, sample_history_length):
        assert sample_history_length > 1
        how_many = min(batch_size, len(self.memory))
        res = []
        for _ in range(sample_history_length):
            res.append([])

        random_number = np.random.uniform(low=0.0, high=1.0, size=(1,))
        contains_first_step = random_number[0] < 0.05  # hard coded here. So 5% of the sampled batches will have first step

        for _ in range(how_many):
            trajectory_id = np.random.randint(len(self.memory))
            trajectory = self.memory[trajectory_id]
            if len(trajectory) < sample_history_length:
                continue
            if contains_first_step:
                head = 0
            else:
                if 1 >= len(trajectory) - sample_history_length + 1:
                    continue
                head = np.random.randint(1, len(trajectory) - sample_history_length + 1)
            for i in range(sample_history_length):
                res[i].append(trajectory[head + i])
        if len(res) == 0:
            return None, None
        return res, contains_first_step

    def __len__(self):
        return len(self.memory)
