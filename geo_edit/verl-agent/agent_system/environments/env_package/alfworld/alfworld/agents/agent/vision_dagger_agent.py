import copy

import numpy as np
try:
    import torch
    import torch.nn.functional as F
except ImportError:
    raise ImportError("torch not found. Please install them via `pip install alfworld[full]`.")

import alfworld.agents
import alfworld.agents.modules.memory as memory
from alfworld.agents.agent import TextDAggerAgent
from alfworld.agents.modules.generic import to_np, to_pt, _words_to_ids, pad_sequences, preproc, max_len, ez_gather_dim_1, LinearSchedule
from alfworld.agents.modules.layers import NegativeLogLoss, masked_mean, compute_mask
from alfworld.agents.detector.mrcnn import load_pretrained_model

import torchvision.transforms as T
from torchvision import models
from torchvision.ops import boxes as box_ops


class VisionDAggerAgent(TextDAggerAgent):
    '''
    Vision Agent trained with DAgger
    '''
    def __init__(self, config):
        super().__init__(config)

        assert self.action_space == "generation"

        self.use_gpu = config['general']['use_cuda']
        self.transform = T.Compose([T.ToTensor()])

        # choose vision model
        self.vision_model_type = config['vision_dagger']['model_type']
        self.use_exploration_frame_feats = config['vision_dagger']['use_exploration_frame_feats']
        self.sequence_aggregation_method = config['vision_dagger']['sequence_aggregation_method']

        # initialize model
        if self.vision_model_type in {'resnet'}:
            self.detector = models.resnet18(pretrained=True)
            self.detector.eval()
            if self.use_gpu:
                self.detector.cuda()
        elif self.vision_model_type in {'maskrcnn', 'maskrcnn_whole'}:
            pretrained_model_path = config['mask_rcnn']['pretrained_model_path']
            self.mask_rcnn_top_k_boxes = self.config['vision_dagger']['maskrcnn_top_k_boxes']
            self.avg2dpool = torch.nn.AvgPool2d((13, 13))
            self.detector = load_pretrained_model(pretrained_model_path)
            self.detector.roi_heads.register_forward_hook(self.box_features_hook)
            self.detection_box_features = []
            self.fpn_pooled_features = []
            self.detector.eval()
            if self.use_gpu:
                self.detector.cuda()
        elif self.vision_model_type in {"no_vision"}:
            print("No Vision Agent")
        else:
            raise NotImplementedError()

    def box_features_hook(self, module, input, output):
        '''
        hook for extracting features from MaskRCNN
        '''

        features, proposals, image_shapes, targets = input

        box_features = module.box_roi_pool(features, proposals, image_shapes)
        box_features = module.box_head(box_features)
        class_logits, box_regression = module.box_predictor(box_features)

        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [len(boxes_in_image) for boxes_in_image in proposals]
        pred_boxes = module.box_coder.decode(box_regression, proposals)

        pred_scores = F.softmax(class_logits, -1)

        # split boxes and scores per image
        pred_boxes = pred_boxes.split(boxes_per_image, 0)
        pred_scores = pred_scores.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        all_keeps = []
        for boxes, scores, image_shape in zip(pred_boxes, pred_scores, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.flatten()
            labels = labels.flatten()

            # remove low scoring boxes
            inds = torch.nonzero(scores > module.score_thresh).squeeze(1)
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, module.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[:self.mask_rcnn_top_k_boxes]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
            all_keeps.append(keep)

        box_features_per_image = []
        for keep in all_keeps:
            box_features_per_image.append(box_features[keep])

        self.detection_box_features = box_features_per_image
        self.fpn_pooled_features = self.avg2dpool(features['pool']).squeeze(-1).squeeze(-1)

    # visual features for state representation
    def extract_visual_features(self, images):
        with torch.no_grad():
            if "resnet" in self.vision_model_type:
                image_tensors = [self.transform(i).cuda() if self.use_gpu else self.transform() for i in images]
                image_tensors = torch.stack(image_tensors, dim=0)
                res_out = self.detector(image_tensors)
                res_out_list = [res_out[i].unsqueeze(0) for i in range(res_out.shape[0])]
                return res_out_list
            elif "maskrcnn" in self.vision_model_type:
                image_tensors = [self.transform(i).cuda() if self.use_gpu else self.transform(i) for i in images]
                self.detector(image_tensors) # hook writes to self.detection_box_features
                if "maskrcnn_whole" in self.vision_model_type:
                    return [i.unsqueeze(0) for i in self.fpn_pooled_features]
                else:
                    return self.detection_box_features
            elif "no_vision" in self.vision_model_type:
                batch_size = len(images)
                zeros = [torch.zeros((1, 1000)) for _ in range(batch_size)]
                if self.use_gpu:
                    zeros = [z.cuda() for z in zeros]
                return zeros
            else:
                raise NotImplementedError()

    # without recurrency
    def train_dagger(self):

        if len(self.dagger_memory) < self.dagger_replay_batch_size:
            return None
        transitions = self.dagger_memory.sample(self.dagger_replay_batch_size)
        if transitions is None:
            return None
        batch = memory.dagger_transition(*zip(*transitions))

        if self.action_space == "generation":
            return self.command_generation_teacher_force(batch.observation_list, batch.task_list, batch.target_list)
        else:
            raise NotImplementedError()

    # with recurrency
    def train_dagger_recurrent(self):

        if len(self.dagger_memory) < self.dagger_replay_batch_size:
            return None
        sequence_of_transitions, contains_first_step = self.dagger_memory.sample_sequence(self.dagger_replay_batch_size, self.dagger_replay_sample_history_length)
        if sequence_of_transitions is None:
            return None

        batches = []
        for transitions in sequence_of_transitions:
            batch = memory.dagger_transition(*zip(*transitions))
            batches.append(batch)

        if self.action_space == "generation":
            return self.command_generation_recurrent_teacher_force([batch.observation_list for batch in batches], [batch.task_list for batch in batches], [batch.target_list for batch in batches], contains_first_step)
        else:
            raise NotImplementedError()

    def command_generation_teacher_force(self, observation_feats, task_desc_strings, target_strings):
        input_target_strings = [" ".join(["[CLS]"] + item.split()) for item in target_strings]
        output_target_strings = [" ".join(item.split() + ["[SEP]"]) for item in target_strings]
        batch_size = len(observation_feats)

        aggregated_obs_feat = self.aggregate_feats_seq(observation_feats)
        h_obs = self.online_net.vision_fc(aggregated_obs_feat)
        h_td, td_mask = self.encode(task_desc_strings, use_model="online")
        h_td_mean = self.online_net.masked_mean(h_td, td_mask).unsqueeze(1)
        h_obs = h_obs.to(h_td_mean.device)
        vision_td = torch.cat((h_obs, h_td_mean), dim=1) # batch x k boxes x hi
        vision_td_mask = torch.ones((batch_size, h_obs.shape[1]+h_td_mean.shape[1])).to(h_td_mean.device)

        input_target = self.get_word_input(input_target_strings)
        ground_truth = self.get_word_input(output_target_strings)  # batch x target_length
        target_mask = compute_mask(input_target)  # mask of ground truth should be the same
        pred = self.online_net.vision_decode(input_target, target_mask, vision_td, vision_td_mask, None)  # batch x target_length x vocab

        batch_loss = NegativeLogLoss(pred * target_mask.unsqueeze(-1), ground_truth, target_mask, smoothing_eps=self.smoothing_eps)
        loss = torch.mean(batch_loss)

        if loss is None:
            return None, None
        # Backpropagate
        self.online_net.zero_grad()
        self.optimizer.zero_grad()
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.clip_grad_norm)
        self.optimizer.step()  # apply gradients
        return to_np(pred), to_np(loss)

    def command_generation_recurrent_teacher_force(self, seq_observation_feats, seq_task_desc_strings, seq_target_strings, contains_first_step=False):
        loss_list = []
        previous_dynamics = None
        batch_size = len(seq_observation_feats[0])
        h_td, td_mask = self.encode(seq_task_desc_strings[0], use_model="online")
        h_td_mean = self.online_net.masked_mean(h_td, td_mask).unsqueeze(1)
        for step_no in range(self.dagger_replay_sample_history_length):
            input_target_strings = [" ".join(["[CLS]"] + item.split()) for item in seq_target_strings[step_no]]
            output_target_strings = [" ".join(item.split() + ["[SEP]"]) for item in seq_target_strings[step_no]]

            obs = [o.to(h_td.device) for o in seq_observation_feats[step_no]]
            aggregated_obs_feat = self.aggregate_feats_seq(obs)
            h_obs = self.online_net.vision_fc(aggregated_obs_feat)
            vision_td = torch.cat((h_obs, h_td_mean), dim=1) # batch x k boxes x hid
            vision_td_mask = torch.ones((batch_size, h_obs.shape[1]+h_td_mean.shape[1])).to(h_td_mean.device)

            averaged_vision_td_representation = self.online_net.masked_mean(vision_td, vision_td_mask)
            current_dynamics = self.online_net.rnncell(averaged_vision_td_representation, previous_dynamics) if previous_dynamics is not None else self.online_net.rnncell(averaged_vision_td_representation)

            input_target = self.get_word_input(input_target_strings)
            ground_truth = self.get_word_input(output_target_strings)  # batch x target_length
            target_mask = compute_mask(input_target)  # mask of ground truth should be the same
            pred = self.online_net.vision_decode(input_target, target_mask, vision_td, vision_td_mask, current_dynamics)  # batch x target_length x vocab

            previous_dynamics = current_dynamics
            if (not contains_first_step) and step_no < self.dagger_replay_sample_update_from:
                previous_dynamics = previous_dynamics.detach()
                continue

            batch_loss = NegativeLogLoss(pred * target_mask.unsqueeze(-1), ground_truth, target_mask, smoothing_eps=self.smoothing_eps)
            loss = torch.mean(batch_loss)
            loss_list.append(loss)

        loss = torch.stack(loss_list).mean()
        if loss is None:
            return None
        # Backpropagate
        self.online_net.zero_grad()
        self.optimizer.zero_grad()
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.clip_grad_norm)
        self.optimizer.step()  # apply gradients
        return to_np(loss)

    def command_generation_greedy_generation(self, observation_feats, task_desc_strings, previous_dynamics):
        with torch.no_grad():
            batch_size = len(observation_feats)

            aggregated_obs_feat = self.aggregate_feats_seq(observation_feats)
            h_obs = self.online_net.vision_fc(aggregated_obs_feat)
            h_td, td_mask = self.encode(task_desc_strings, use_model="online")
            h_td_mean = self.online_net.masked_mean(h_td, td_mask).unsqueeze(1)
            h_obs = h_obs.to(h_td_mean.device)
            vision_td = torch.cat((h_obs, h_td_mean), dim=1) # batch x k boxes x hid
            vision_td_mask = torch.ones((batch_size, h_obs.shape[1]+h_td_mean.shape[1])).to(h_td_mean.device)

            if self.recurrent:
                averaged_vision_td_representation = self.online_net.masked_mean(vision_td, vision_td_mask)
                current_dynamics = self.online_net.rnncell(averaged_vision_td_representation, previous_dynamics) if previous_dynamics is not None else self.online_net.rnncell(averaged_vision_td_representation)
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
                pred = self.online_net.vision_decode(input_target, target_mask, vision_td, vision_td_mask, current_dynamics)  # batch x target_length x vocab
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
            res = [self.tokenizer.decode(item) for item in input_target_list]
            res = [item.replace("[CLS]", "").replace("[SEP]", "").strip() for item in res]
            res = [item.replace(" in / on ", " in/on " ) for item in res]
            return res, current_dynamics

    def get_vision_feat_mask(self, observation_feats):
        batch_size = len(observation_feats)
        num_vision_feats = [of.shape[0] for of in observation_feats]
        max_feat_len = max(num_vision_feats)
        mask = torch.zeros((batch_size, max_feat_len))
        for b, num_vision_feat in enumerate(num_vision_feats):
            mask[b,:num_vision_feat] = 1
        return mask

    def extract_exploration_frame_feats(self, exploration_frames):
        exploration_frame_feats = []
        for batch in exploration_frames:
            ef_feats = []
            for image in batch:
                ef_feats.append(self.extract_visual_features([image])[0])
            # cat_feats = torch.cat(ef_feats, dim=0)
            max_feat_len = max([f.shape[0] for f in ef_feats])
            stacked_feats = self.online_net.vision_fc.pad_and_stack(ef_feats, max_feat_len=max_feat_len)
            stacked_feats = stacked_feats.view(-1, self.online_net.vision_fc.in_features)
            exploration_frame_feats.append(stacked_feats)
        return exploration_frame_feats

    def aggregate_feats_seq(self, feats):
        if self.sequence_aggregation_method == "sum":
            return [f.sum(0).unsqueeze(0) for f in feats]
        elif self.sequence_aggregation_method == "average":
            return [f.mean(0).unsqueeze(0) for f in feats]
        elif self.sequence_aggregation_method == "rnn":
            max_feat_len = max([f.shape[0] for f in feats])
            feats_stack = self.online_net.vision_fc.pad_and_stack(feats, max_feat_len=max_feat_len)
            feats_h, feats_c = self.online_net.vision_feat_seq_rnn(feats_stack)
            aggregated_feats = feats_h[:,0,:].unsqueeze(1)
            return [b for b in aggregated_feats]
        else:
            raise ValueError("sequence_aggregation_method must be sum, average or rnn")
