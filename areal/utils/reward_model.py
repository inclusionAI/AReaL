import torch
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast


class RewardModel(nn.Module):
    def __init__(
        self, base_model, tokenizer, num_padding_at_beginning=0, compute_fp32_loss=False
    ):
        super().__init__()
        self.config = base_model.config
        self.num_padding_at_beginning = num_padding_at_beginning
        if hasattr(self.config, "word_embed_proj_dim"):
            self.v_head = nn.Linear(
                self.config.word_embed_proj_dim, 1, bias=False, dtype=torch.bfloat16
            )
        else:
            self.config.n_embd = (
                self.config.hidden_size
                if hasattr(self.config, "hidden_size")
                else self.config.n_embd
            )
            print(self.config.n_embd)
            self.v_head = nn.Linear(
                self.config.n_embd, 1, bias=False, dtype=torch.bfloat16
            )
        self.rwtransformer = base_model
        self.PAD_ID = tokenizer.pad_token_id
        self.compute_fp32_loss = compute_fp32_loss
        self._no_split_modules = getattr(
            self.rwtransformer, "_no_split_modules", list()
        )
        self._no_split_modules.append(nn.Linear)

    def gradient_checkpointing_enable(self):
        self.rwtransformer.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.rwtransformer.gradient_checkpointing_disable()

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        use_cache=False,
        **kwargs,
    ):
        if self.config.model_type == "llama":
            kwargs = dict()
        else:
            kwargs = dict(head_mask=head_mask)

        transformer_outputs = self.rwtransformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=True,
            **kwargs,
        )
        hidden_states = transformer_outputs.hidden_states[-1]
        rewards = self.v_head(hidden_states).squeeze(-1)  # (batch_size, seq_len)

        return CausalLMOutputWithPast(
            logits=rewards,
            past_key_values=getattr(transformer_outputs, "past_key_values", None),
            hidden_states=getattr(transformer_outputs, "hidden_states", None),
            attentions=getattr(transformer_outputs, "attentions", None),
        )
