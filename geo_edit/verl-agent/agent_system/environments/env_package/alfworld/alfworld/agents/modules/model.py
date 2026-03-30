import os
import copy
import logging
import numpy as np

import torch
from alfworld.agents.modules.layers import EncoderBlock, DecoderBlock, NoisyLinear, CQAttention, compute_mask, PointerSoftmax, masked_softmax, SelfAttention, BoxFeaturesFC
BERT_EMBEDDING_SIZE = 768

logger = logging.getLogger(__name__)


class Policy(torch.nn.Module):
    model_name = 'policy'

    def __init__(self, config, bert_model, word_vocab_size):
        super(Policy, self).__init__()
        self.config = config
        self.bert_model = bert_model
        self.word_vocab_size = word_vocab_size
        self.read_config()
        self._def_layers()
        # self.print_parameters()

    def print_parameters(self):
        amount = 0
        for p in self.parameters():
            amount += np.prod(p.size())
        print("total number of parameters: %s" % (amount))
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        amount = 0
        for p in parameters:
            amount += np.prod(p.size())
        print("number of trainable parameters: %s" % (amount))

    def read_config(self):
        # model config
        model_config = self.config['general']['model']
        self.encoder_layers = model_config['encoder_layers']
        self.encoder_conv_num = model_config['encoder_conv_num']
        self.block_hidden_dim = model_config['block_hidden_dim']
        self.n_heads = model_config['n_heads']
        self.block_dropout = model_config['block_dropout']
        self.dropout = model_config['dropout']
        self.decoder_layers = model_config['decoder_layers']
        self.recurrent = model_config['recurrent']

        self.noisy_net = self.config['rl']['epsilon_greedy']['noisy_net']

        self.resnet_fc_dim = self.config['vision_dagger']['resnet_fc_dim']
        self.vision_model_type = self.config['vision_dagger']['model_type']
        if self.vision_model_type == "maskrcnn_whole":
            self.vision_input_dim = 256
        elif self.vision_model_type == "maskrcnn":
            self.vision_input_dim = 1024
        else:
            self.vision_input_dim = 1000


    def _def_layers(self):

        self.word_embedding_prj = torch.nn.Linear(BERT_EMBEDDING_SIZE, self.block_hidden_dim, bias=False)
        self.encoder =  torch.nn.ModuleList([EncoderBlock(conv_num=self.encoder_conv_num, ch_num=self.block_hidden_dim, k=5, block_hidden_dim=self.block_hidden_dim, n_head=self.n_heads, dropout=self.block_dropout) for _ in range(self.encoder_layers)])

        self.aggregation_attention = CQAttention(block_hidden_dim=self.block_hidden_dim, dropout=self.dropout)
        self.aggregation_attention_proj = torch.nn.Linear(self.block_hidden_dim * 4, self.block_hidden_dim)

        if self.recurrent:
            self.rnncell = torch.nn.GRUCell(self.block_hidden_dim, self.block_hidden_dim)
            self.dynamics_aggregation = torch.nn.Linear(self.block_hidden_dim * 2, self.block_hidden_dim)
        else:
            self.rnncell, self.dynamics_aggregation = None, None

        self.decoder = torch.nn.ModuleList([DecoderBlock(ch_num=self.block_hidden_dim, k=5, block_hidden_dim=self.block_hidden_dim, n_head=self.n_heads, dropout=self.block_dropout) for _ in range(self.decoder_layers)])
        self.decoding_to_embedding = torch.nn.Linear(self.block_hidden_dim, BERT_EMBEDDING_SIZE)
        self.embedding_to_words = torch.nn.Linear(BERT_EMBEDDING_SIZE, self.word_vocab_size, bias=False)
        self.embedding_to_words.weight = self.bert_model.embeddings.word_embeddings.weight
        self.embedding_to_words.weight.requires_grad = False
        self.pointer_softmax = PointerSoftmax(input_dim=self.block_hidden_dim, hidden_dim=self.block_hidden_dim)

        linear_function = NoisyLinear if self.noisy_net else torch.nn.Linear
        self.action_scorer_linear_1 = linear_function(self.block_hidden_dim * 2, self.block_hidden_dim) # [candidate_rep, obs_att]
        self.action_scorer_linear_2 = linear_function(self.block_hidden_dim, 1)
        self.action_scorer_extra_self_attention = SelfAttention(self.block_hidden_dim, self.n_heads, self.dropout)
        self.action_scorer_extra_linear = linear_function(self.block_hidden_dim, self.block_hidden_dim)

        # vision modules
        self.vision_fc = BoxFeaturesFC(in_features=self.vision_input_dim, out_features=self.resnet_fc_dim)
        self.vision_feat_seq_rnn = torch.nn.GRU(self.vision_input_dim, int(self.vision_input_dim/2), 1, bidirectional=True)

    def get_bert_embeddings(self, _input_words, _input_masks):
        # _input_words: batch x time
        # _input_masks: batch x time
        if _input_words.size(1) > 512:  # exceeds the length limit of pre-trained bert max_position_embeddings
            # hack
            seg_length = 500
            outputs = []
            num_batch = (_input_words.size(1) + seg_length - 1) // seg_length
            for i in range(num_batch):
                batch_input = _input_words[:, i * seg_length: (i + 1) * seg_length]
                batch_mask = _input_masks[:, i * seg_length: (i + 1) * seg_length]
                out = self.get_bert_embeddings(batch_input, batch_mask)
                outputs.append(out)
            return torch.cat(outputs, 1)

        with torch.no_grad():
            res = self.bert_model.embeddings(_input_words)
            res = res * _input_masks.unsqueeze(-1)
        return res

    def embed(self, input_words, input_word_masks):
        word_embeddings = self.get_bert_embeddings(input_words, input_word_masks)  # batch x time x emb
        word_embeddings = word_embeddings * input_word_masks.unsqueeze(-1)  # batch x time x hid
        word_embeddings = self.word_embedding_prj(word_embeddings)
        word_embeddings = word_embeddings * input_word_masks.unsqueeze(-1)  # batch x time x hid
        return word_embeddings

    def encode_text(self, input_word_ids):
        # input_word_ids: batch x seq_len
        # text embedding / encoding
        input_word_masks = compute_mask(input_word_ids)
        embeddings = self.embed(input_word_ids, input_word_masks)  # batch x seq_len x emb
        squared_mask = torch.bmm(input_word_masks.unsqueeze(-1), input_word_masks.unsqueeze(1))  # batch x seq_len x seq_len
        encoding_sequence = embeddings
        for i in range(self.encoder_layers):
            encoding_sequence = self.encoder[i](encoding_sequence, input_word_masks, squared_mask, i * (self.encoder_conv_num + 2) + 1, self.encoder_layers)  # batch x time x enc
        return encoding_sequence, input_word_masks

    def aggretate_information(self, h_obs, obs_mask, h_td, td_mask):
        aggregated_obs_representation = self.aggregation_attention(h_obs, h_td, obs_mask, td_mask)  # batch x obs_length x 4*hid
        aggregated_obs_representation = self.aggregation_attention_proj(aggregated_obs_representation)  # batch x obs_length x hid
        aggregated_obs_representation = torch.tanh(aggregated_obs_representation)
        aggregated_obs_representation = aggregated_obs_representation * obs_mask.unsqueeze(-1)
        return aggregated_obs_representation

    def score_actions(self, candidate_representations, cand_mask, h_obs, obs_mask, current_dynamics, fix_shared_components=False):
        # candidate_representations: batch x num_candidate x hid
        # cand_mask: batch x num_candidate
        batch_size, num_candidate = candidate_representations.size(0), candidate_representations.size(1)
        aggregated_obs_representation = self.masked_mean(h_obs, obs_mask)  # batch x hid
        if self.recurrent:
            aggregated_obs_representation = self.dynamics_aggregation(torch.cat([aggregated_obs_representation, current_dynamics], -1))
            aggregated_obs_representation = torch.relu(aggregated_obs_representation)  # batch x hid

        # if action space is "beam_saerch_choice"
        if fix_shared_components:
            aggregated_obs_representation = aggregated_obs_representation.detach()
            candidate_representations = candidate_representations.detach()

        # only text
        new_h_expanded = torch.stack([aggregated_obs_representation] * num_candidate, 1).view(batch_size, num_candidate, aggregated_obs_representation.size(-1))
        output = self.action_scorer_linear_1(torch.cat([candidate_representations, new_h_expanded], -1))  # batch x num_candidate x hid
        output = torch.relu(output)
        output = output * cand_mask.unsqueeze(-1)

        if fix_shared_components:
            # use some extra trainable parameters to increase the model's capacity
            cand_mask_squared = torch.bmm(cand_mask.unsqueeze(-1), cand_mask.unsqueeze(1))  # batch x num_candidate x num_candidate
            output, _ = self.action_scorer_extra_self_attention(output, cand_mask_squared, output, output)  # batch x num_candidate x hid
            output = self.action_scorer_extra_linear(output)  # batch x num_candidate x hid
            output = torch.relu(output)
            output = output * cand_mask.unsqueeze(-1)

        output = self.action_scorer_linear_2(output).squeeze(-1)  # batch x num_candidate
        output = output * cand_mask

        return output, cand_mask

    def get_subsequent_mask(self, seq):
        ''' For masking out the subsequent info. '''
        _, length = seq.size()
        subsequent_mask = torch.triu(torch.ones((length, length)), diagonal=1).float()
        subsequent_mask = 1.0 - subsequent_mask
        if seq.is_cuda:
            subsequent_mask = subsequent_mask.cuda()
        subsequent_mask = subsequent_mask.unsqueeze(0)  # 1 x time x time
        return subsequent_mask

    def decode(self, input_target_word_ids, input_target_word_masks, h_obs, obs_mask, current_dynamics, input_obs):
        trg_mask = input_target_word_masks
        trg_embeddings = self.embed(input_target_word_ids, input_target_word_masks)  # batch x target_len x emb

        trg_mask_square = torch.bmm(trg_mask.unsqueeze(-1), trg_mask.unsqueeze(1))  # batch x target_len x target_len
        trg_mask_square = trg_mask_square * self.get_subsequent_mask(input_target_word_ids)  # batch x target_len x target_len
        obs_mask_square = torch.bmm(trg_mask.unsqueeze(-1), obs_mask.unsqueeze(1))  # batch x target_len x obs_len

        if self.recurrent:
            current_dynamics_expanded = torch.stack([current_dynamics] * h_obs.size(1), 1)  # batch x source_len x hid
            h_obs = self.dynamics_aggregation(torch.cat([h_obs, current_dynamics_expanded], -1))  # batch x source_len x hid
            h_obs = torch.relu(h_obs)  # batch x source_len x hid

        trg_decoder_output = trg_embeddings
        for i in range(self.decoder_layers):
            trg_decoder_output, target_target_representations, target_source_representations, target_source_attention = self.decoder[i](trg_decoder_output, trg_mask, trg_mask_square, h_obs, obs_mask_square, i * 3 + 1, self.decoder_layers)  # batch x time x hid

        trg_decoder_output = self.decoding_to_embedding(trg_decoder_output)  # batch x time x emb
        trg_decoder_output = torch.tanh(trg_decoder_output)
        trg_decoder_output = self.embedding_to_words(trg_decoder_output)  # batch x time x n_vocab
        trg_decoder_output = masked_softmax(trg_decoder_output, m=trg_mask.unsqueeze(-1), axis=-1)
        output = self.pointer_softmax(target_target_representations, target_source_representations, trg_decoder_output, trg_mask, target_source_attention, obs_mask, input_obs)

        return output

    def vision_decode(self, input_target_word_ids, input_target_word_masks, h_obs, obs_mask, current_dynamics):
        trg_mask = input_target_word_masks
        trg_embeddings = self.embed(input_target_word_ids, input_target_word_masks)  # batch x target_len x emb

        trg_mask_square = torch.bmm(trg_mask.unsqueeze(-1), trg_mask.unsqueeze(1))  # batch x target_len x target_len
        trg_mask_square = trg_mask_square * self.get_subsequent_mask(input_target_word_ids)  # batch x target_len x target_len
        obs_mask_square = torch.bmm(trg_mask.unsqueeze(-1), obs_mask.unsqueeze(1))  # batch x target_len x obs_len

        if self.recurrent:
            current_dynamics_expanded = torch.stack([current_dynamics] * h_obs.size(1), 1)  # batch x source_len x hid
            h_obs = self.dynamics_aggregation(torch.cat([h_obs, current_dynamics_expanded], -1))  # batch x source_len x hid
            h_obs = torch.relu(h_obs)  # batch x source_len x hid

        trg_decoder_output = trg_embeddings
        for i in range(self.decoder_layers):
            trg_decoder_output, target_target_representations, target_source_representations, target_source_attention = self.decoder[i](trg_decoder_output, trg_mask, trg_mask_square, h_obs, obs_mask_square, i * 3 + 1, self.decoder_layers)  # batch x time x hid

        trg_decoder_output = self.decoding_to_embedding(trg_decoder_output)  # batch x time x emb
        trg_decoder_output = torch.tanh(trg_decoder_output)
        trg_decoder_output = self.embedding_to_words(trg_decoder_output)  # batch x time x n_vocab
        trg_decoder_output = masked_softmax(trg_decoder_output, m=trg_mask.unsqueeze(-1), axis=-1)

        return trg_decoder_output

    def masked_mean(self, h_obs, obs_mask):
        # h_obs: batch x obs_length x hid
        # obs_mask: batch x obs_length
        _mask = torch.sum(obs_mask, -1)  # batch
        obs_representations = torch.sum(h_obs, -2)  # batch x hid
        tmp = torch.eq(_mask, 0).float()
        if obs_representations.is_cuda:
            tmp = tmp.cuda()
        _mask = _mask + tmp
        obs_representations = obs_representations / _mask.unsqueeze(-1)  # batch x hid
        return obs_representations

    def reset_noise(self):
        if self.noisy_net:
            self.action_scorer_linear_1.reset_noise()
            self.action_scorer_linear_2.reset_noise()
