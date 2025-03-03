# coding=utf-8
# copied from bart

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_utils import BaseModel
from transformers.generation_utils import top_k_top_p_filtering
from transformers.models.blenderbot_small import (BlenderbotSmallConfig, BlenderbotSmallForConditionalGeneration, )
from transformers.modeling_outputs import (BaseModelOutput, Seq2SeqModelOutput, Seq2SeqLMOutput, )
from .PARAMS import SAMPLE, TEMPERATURE
from transformers import BartTokenizer, BartForConditionalGeneration


def my_top_k_top_p_filtering(
        logits: torch.Tensor,
        top_k: int = 0,
        top_p: float = 1.0,
        filter_value: float = -float("Inf"),
        min_tokens_to_keep: int = 1,
) -> torch.Tensor:
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


# Copied from transformers.models.bart.modeling_bart.shift_tokens_right
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


class Model(BaseModel, BlenderbotSmallForConditionalGeneration):
    def __init__(self, config: BlenderbotSmallConfig):
        super().__init__(config)
        # verison 1
        # self.fc=nn.Linear(2*self.model.config.d_model,2*self.model.config.d_model)# 通道上进行分割
        # version 2
        self.fc=nn.Linear(self.model.config.d_model,self.model.config.d_model)# token上进行分割
        self.persona_norm = nn.LayerNorm(self.model.config.d_model, elementwise_affine=True)  # 512
        self.context_norm = nn.LayerNorm(self.model.config.d_model, elementwise_affine=True)  # 512
        self.refer_norm = nn.LayerNorm(self.model.config.d_model, elementwise_affine=True)  # 512
        self.context_norm_2 = nn.LayerNorm(self.model.config.d_model, elementwise_affine=True)  # 512
        self.norm = nn.LayerNorm(self.model.config.d_model, elementwise_affine=True)  # 512
        # self.persona_context_w = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.persona_context_w = nn.Parameter(torch.tensor([1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5]))
        self.my_past = None
        self.my_encoder_outputs = None
        self.generation_strategy = None
        # self.strategy_alpha = [0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75]
        # self.strategy_alpha = [0, 0.75, 0, 0, 0.75, 0.75, 0.75, 0.375]
        self.train_alpha = False
        self.strategy_alpha = nn.Parameter(torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]))
        # self.strategy_alpha = [0, 0, 0, 0, 0, 0, 0, 0]
        # "[Question]","[Restatement or Paraphrasing]",
        # "[Reflection of feelings]","[Self-disclosure]",
        # "[Affirmation and Reassurance]","[Providing Suggestions]",
        # "[Information]","[Others]"

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            persona_input_ids=None,
            persona_attention_mask=None,
            refer_input_ids=None,
            refer_attention_mask=None,
            decoder_input_ids=None,
            encoder_outputs=None,
            past_key_values=None,
            labels=None,
            use_cache=None,
            return_dict=None,
            validation=False,
            **kwargs
    ):
        assert self.toker is not None

        encoded_info = kwargs
        assert (self.training or validation) == (labels is not None)
        # if validation:
        #     labels[:, 0] = -100
        #     print(labels[:, 0])
        # TODO: i don't know if need following 5 lines
        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.toker.pad_token_id, self.toker.bos_token_id
                )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if not self.training and not validation:  # inference
            use_cache = True
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        persona_change_attention = True

        if self.train_alpha and labels is not None:
            my_outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                encoder_outputs=encoder_outputs,
                past_key_values=past_key_values,
                use_cache=use_cache,
                return_dict=return_dict,
            )

        if not persona_change_attention or persona_input_ids is None:
            # print(input_ids, attention_mask, decoder_input_ids, encoder_outputs, (past_key_values is None))
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                encoder_outputs=encoder_outputs,
                past_key_values=past_key_values,
                use_cache=use_cache,
                return_dict=return_dict,
            )
        else:
            output_attentions = self.model.config.output_attentions
            output_hidden_states = (
                self.model.config.output_hidden_states
            )
            use_cache = use_cache if use_cache is not None else self.model.config.use_cache
            return_dict = return_dict if return_dict is not None else self.model.config.use_return_dict
            head_mask = None
            inputs_embeds = None
            decoder_inputs_embeds = None
            cross_attn_head_mask = None
            decoder_head_mask = None
            decoder_attention_mask = None

            encoder_outputs = self.model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            persona_encoder_outputs = self.model.encoder(
                input_ids=persona_input_ids,
                attention_mask=persona_attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            # print("encoder_outputs",encoder_outputs[0].size())
            # print("persona_encoder_outputs",persona_encoder_outputs[0].size())
            # version 1
            # concat=torch.cat((encoder_outputs[0],persona_encoder_outputs[0]),dim=-1)
            # version 2
            concat = torch.cat((persona_encoder_outputs[0],encoder_outputs[0]), dim=1)
            # print("concat_outputs",concat.size())
            
            gated=torch.sigmoid(self.fc(concat)) # 也可以softmax互相比，但是要搞清楚shape
            # version 1
            # persona_gated=gated[:, :, :self.config.d_model]
            # context_gated=gated[:, :, :self.config.d_model]
            # version 2
            # concat = torch.cat((persona_encoder_outputs[0],encoder_outputs[0]), dim=1)

            seq_len_context = encoder_outputs[0].size(1)
            seq_len_persona = persona_encoder_outputs[0].size(1)
            persona_gated = gated[:, :seq_len_persona, :]
            context_gated = gated[:, seq_len_persona:, :]

            persona=persona_gated * persona_encoder_outputs[0]
            context=context_gated * encoder_outputs[0]
            # print("persona",persona.size())
            # print("context",context.size())
            # context = torch.stack([torch.matmul(torch.softmax(torch.matmul(j, i.t()), dim=-1), i) for i, j in
            #                        zip(encoder_outputs[0], persona_encoder_outputs[0])])
            # persona = torch.stack([torch.matmul(torch.softmax(torch.matmul(i, j.t()), dim=-1), j) for i, j in
            #                        zip(encoder_outputs[0], persona_encoder_outputs[0])])
            context_persona = self.persona_norm(context+persona_encoder_outputs.last_hidden_state)
            persona = self.context_norm(encoder_outputs.last_hidden_state+persona)

            # Refer
            refer_encoder_outputs = self.model.encoder(
                input_ids=refer_input_ids,
                attention_mask=refer_attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            # comet_output = self.cog_encoder(refer_encoder_outputs[0], comet_attention_mask.unsqueeze(1))
            # cls_tokens = comet_output[:, 0].unsqueeze(1)
            # dim = [-1, encoder_outputs[0].shape[1], -1]
            # cog_concat = torch.cat([encoder_outputs[0], cls_tokens.expand(dim)], dim=-1)
            # cog_concat_enc = self.cog_ref_encoder(cog_concat, attention_mask.unsqueeze(1))
            # cog_contrib = nn.Sigmoid()(cog_concat_enc)
            # cog_ref_ctx = cog_contrib * cog_concat_enc
            # cog_concat_enc = self.cog_lin(cog_ref_ctx)

            context = torch.stack([torch.matmul(torch.softmax(torch.matmul(j, i.t()), dim=-1), i) for i, j in
                                   zip(encoder_outputs[0], refer_encoder_outputs[0])])  # torch.Size([4, 512, 512])
            refer = torch.stack([torch.matmul(torch.softmax(torch.matmul(i, j.t()), dim=-1), j) for i, j in
                                 zip(encoder_outputs[0], refer_encoder_outputs[0])])
            context_refer = self.refer_norm(context + refer_encoder_outputs.last_hidden_state)
            refer = self.context_norm_2(encoder_outputs.last_hidden_state + refer)
            # context_comet = self.comet_norm(context + refer_encoder_outputs[0])
            # comet = self.context_norm_2(encoder_outputs.last_hidden_state + comet)

            # 归一化权重
            w1 = torch.exp(self.persona_context_w[0]) / torch.sum(torch.exp(self.persona_context_w))
            w2 = torch.exp(self.persona_context_w[1]) / torch.sum(torch.exp(self.persona_context_w))
            w3 = torch.exp(self.persona_context_w[2]) / torch.sum(torch.exp(self.persona_context_w))
            w4 = torch.exp(self.persona_context_w[3]) / torch.sum(torch.exp(self.persona_context_w))
            w5 = torch.exp(self.persona_context_w[4]) / torch.sum(torch.exp(self.persona_context_w))
            # encoder_outputs.last_hidden_state = w1 * encoder_outputs.last_hidden_state + w2 * context + w3 * persona + w4 * context_refer + w5 * refer
            encoder_outputs.last_hidden_state = self.norm(w1 * encoder_outputs.last_hidden_state + w2 * context_persona + w3 * persona + w4 * context_refer + w5 * refer)
            decoder_outputs = self.model.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=encoder_outputs[0],
                encoder_attention_mask=attention_mask,
                head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                past_key_values=past_key_values,
                inputs_embeds=decoder_inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

           

            if not return_dict:
                outputs = decoder_outputs + encoder_outputs
            else:
                outputs = Seq2SeqModelOutput(
                    last_hidden_state=decoder_outputs.last_hidden_state,
                    past_key_values=decoder_outputs.past_key_values,
                    decoder_hidden_states=decoder_outputs.hidden_states,
                    decoder_attentions=decoder_outputs.attentions,
                    cross_attentions=decoder_outputs.cross_attentions,
                    encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                    encoder_hidden_states=encoder_outputs.hidden_states,
                    encoder_attentions=encoder_outputs.attentions,
                )

        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias
        # TODO: 考虑搞成两部分loss
        # logits=outputs.logits
        # first_token_logits = logits[:, 0, :]  # shape: [batch_size, vocab_size]
    
        # # 使用softmax将logits转换为概率分布
        # first_token_probs = torch.sum(F.softmax(first_token_logits, dim=-1),dim=-1)
        # loss_single_step = torch.mean(first_token_probs)
        
        # 计算第一个token的损失，直接对词汇表求和
        # target_first_token = labels[:, 0]  # 首个token的真实标签
        # first_token_loss = nn.CrossEntropyLoss()(first_token_logits, target_first_token)

        if self.train_alpha and labels is not None:
            my_logits = self.lm_head(my_outputs[0]) + self.final_logits_bias

            alpha_l = []
            if lm_logits.get_device() == -1:
                device = 'cpu'
            else:
                device = 'cuda'
            lm_size = lm_logits.size()
            for i in labels[:, 0]:
                tmp_alpha = self.strategy_alpha[i.item()+8-len(self.toker)]
                tmp_alpha = tmp_alpha * torch.ones(lm_size[1], lm_size[2], device=device)
                alpha_l.append(tmp_alpha)
            alpha_l = torch.stack(alpha_l)
            lm_logits = (torch.ones_like(lm_logits, device=device) + alpha_l) * lm_logits - alpha_l * my_logits
            # assert False

        masked_lm_loss = None
        if labels is not None:
            # 两部分loss相加
            # logits_strategy = lm_logits[:, :1, -8:].contiguous()
            # lm_logits = lm_logits[:, 1:, :len(self.toker) - 8].contiguous()
            # strategy = labels[:, :1].contiguous()
            # strategy = strategy - (len(self.toker)-8) * torch.ones_like(strategy, device=strategy.get_device())
            # labels = labels[:, 1:].contiguous()
            # print(logits_strategy.size(), strategy.size(), lm_logits.size(), labels.size())

            # print(lm_logits.size(), labels.size())
            loss = F.cross_entropy(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1), reduction='none')
            loss = loss.view(labels.size(0), labels.size(1))
            # loss[:,0]=1.5*loss[:,0]
            label_size = torch.sum(labels.ne(-100), dim=1).type_as(loss)
            # print("not -100: ", torch.sum(labels.ne(-100), dim=1), "equal -100: ", torch.sum(torch.logical_not(labels.ne(-100)), dim=1))
            masked_lm_loss = torch.sum(loss) / torch.sum(label_size)
            ppl_value = torch.exp(torch.mean(torch.sum(loss, dim=1).float() / label_size.float()))
            # print(strategy, logits_strategy)
            # strategy_loss = F.cross_entropy(logits_strategy.view(-1, logits_strategy.size(-1)), strategy.view(-1), reduction='none')
            # strategy_loss = strategy_loss.view(strategy.size(0), strategy.size(1))
            # print(masked_lm_loss, strategy_loss / 4)
            # masked_lm_loss += torch.sum(strategy_loss) / strategy.size(0)


        if not self.training and not validation:  # inference
            # print(input_ids, attention_mask, self.my_past, self.my_encoder_outputs)
            # print(self.strategy_alpha)
            my_outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                encoder_outputs=self.my_encoder_outputs,
                past_key_values=self.my_past,
                use_cache=use_cache,
                return_dict=return_dict,
            )
            self.my_past = my_outputs.past_key_values
            my_logits = self.lm_head(my_outputs[0]) + self.final_logits_bias
            # alpha = 0.75
            # alpha = 0
            alpha_l = []
            # print(lm_logits.get_device())
            if lm_logits.get_device() == -1:
                device = 'cpu'
            else:
                device = 'cuda'
            lm_size = lm_logits.size()
            for i in self.generation_strategy:
                tmp_alpha = self.strategy_alpha[i.item()]
                tmp_alpha = tmp_alpha * torch.ones(lm_size[1], lm_size[2], device=device)
                alpha_l.append(tmp_alpha)
            alpha_l = torch.stack(alpha_l)
            # print(alpha_l.size(), alpha_l)
            # assert 0
            # print(lm_logits.get_device())
            lm_logits = (torch.ones_like(lm_logits, device=device)+alpha_l)*lm_logits - alpha_l*my_logits
            # lm_logits = my_top_k_top_p_filtering(lm_logits, top_k=10)
            # lm_logits = (1+alpha)*lm_logits - alpha*my_logits

            if not return_dict:
                output = (lm_logits,) + outputs[1:]
                return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

            return Seq2SeqLMOutput(
                loss=masked_lm_loss,
                logits=lm_logits,
                past_key_values=outputs.past_key_values,
                decoder_hidden_states=outputs.decoder_hidden_states,
                decoder_attentions=outputs.decoder_attentions,
                cross_attentions=outputs.cross_attentions,
                encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                encoder_hidden_states=outputs.encoder_hidden_states,
                encoder_attentions=outputs.encoder_attentions,
            )

        elif self.training:  # training
            assert not validation
            res = {'all': masked_lm_loss, 'ppl': ppl_value, }
            return res

        else:  # validation
            assert not self.training
            return loss, label_size

    def predict_strategy(self, logits, encoded_info):
        assert not self.training
        strat_id = encoded_info.get('strat_id', None)
        logits = logits[:, 0, -8:]

        if strat_id is not None:
            pred = strat_id
        else:
            if SAMPLE:
                filtered_logits = top_k_top_p_filtering(logits / TEMPERATURE, top_p=0.9)
                pred = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1).squeeze(-1)
            else:
                pred = torch.argmax(logits, dim=-1)

        pred_top1 = torch.topk(logits, k=1, dim=-1)[1]
        pred_top2 = torch.topk(logits, k=2, dim=-1)[1]
        pred_top3 = torch.topk(logits, k=3, dim=-1)[1]
        pred_top4 = torch.topk(logits, k=4, dim=-1)[1]
        pred_top5 = torch.topk(logits, k=5, dim=-1)[1]

        encoded_info.update({
            'pred_strat_id': pred,
            'pred_strat_id_top1': pred_top1,
            'pred_strat_id_top2': pred_top2,
            'pred_strat_id_top3': pred_top3,
            'pred_strat_id_top4': pred_top4,
            'pred_strat_id_top5': pred_top5,
            'pred_strat_id_dist': F.softmax(logits, dim=-1)
        })
        # pred_top1 = torch.topk(logits, k=1, dim=-1)[1]
        # pred_top3 = torch.topk(logits, k=5, dim=-1)[1]

        # encoded_info.update({
        #     'pred_strat_id': pred,
        #     'pred_strat_id_top1': pred_top1,
        #     'pred_strat_id_top3': pred_top3,
        #     'pred_strat_id_dist': F.softmax(logits, dim=-1)
        # })

    @torch.no_grad()
    def generate(
            self,
            input_ids=None,
            attention_mask=None,
            persona_input_ids=None,
            persona_attention_mask=None,
            refer_input_ids=None,
            refer_attention_mask=None,
            decoder_input_ids=None,
            return_dict=None,
            **kwargs
    ):
        assert not self.training
        assert self.toker is not None

        encoded_info = kwargs
        assert decoder_input_ids.size(1) == 1
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict,
        )

        self.my_encoder_outputs = encoder_outputs.copy()
        self.my_past = None

        persona_change_attention = True
        if persona_change_attention:
            persona_encoder_outputs = self.model.encoder(
                input_ids=persona_input_ids,
                attention_mask=persona_attention_mask,
                return_dict=return_dict,
            )
            
            # print("encoder_outputs",encoder_outputs[0].size())
            # print("persona_encoder_outputs",persona_encoder_outputs[0].size())
            # version 1
            # concat=torch.cat((encoder_outputs[0],persona_encoder_outputs[0]),dim=-1)
            # version 2
            concat = torch.cat((persona_encoder_outputs[0],encoder_outputs[0]), dim=1)
            # print("concat_outputs",concat.size())
            gated=torch.sigmoid(self.fc(concat)) # 也可以softmax互相比，但是要搞清楚shape
            # persona_gated=gated[:, :, :self.config.d_model]
            # context_gated=gated[:, :, :self.config.d_model]
            # version 2
            # concat = torch.cat((persona_encoder_outputs[0],encoder_outputs[0]), dim=1)
            seq_len_context = encoder_outputs[0].size(1)
            seq_len_persona = persona_encoder_outputs[0].size(1)
            persona_gated = gated[:, :seq_len_persona, :]
            context_gated = gated[:, seq_len_persona:, :]

            persona=persona_gated * persona_encoder_outputs[0]
            context=context_gated * encoder_outputs[0]
            # print("persona",persona.size())
            # print("context",context.size())
            # context = torch.stack([torch.matmul(torch.softmax(torch.matmul(j, i.t()), dim=-1), i) for i, j in
            #                        zip(encoder_outputs[0], persona_encoder_outputs[0])])
            # persona = torch.stack([torch.matmul(torch.softmax(torch.matmul(i, j.t()), dim=-1), j) for i, j in
            #                        zip(encoder_outputs[0], persona_encoder_outputs[0])])

            context = self.persona_norm(context + persona_encoder_outputs.last_hidden_state)
            persona = self.context_norm(encoder_outputs.last_hidden_state + persona)

            # Refer
            refer_encoder_outputs = self.model.encoder(
                input_ids=refer_input_ids,
                attention_mask=refer_attention_mask,
                return_dict=return_dict,
            )

            context = torch.stack([torch.matmul(torch.softmax(torch.matmul(j, i.t()), dim=-1), i) for i, j in
                                   zip(encoder_outputs[0], refer_encoder_outputs[0])])  # torch.Size([4, 512, 512])
            refer = torch.stack([torch.matmul(torch.softmax(torch.matmul(i, j.t()), dim=-1), j) for i, j in
                                 zip(encoder_outputs[0], refer_encoder_outputs[0])])
            context_refer = self.refer_norm(context + refer_encoder_outputs.last_hidden_state)
            refer = self.context_norm_2(encoder_outputs.last_hidden_state + refer)
            # 归一化权重
            w1 = torch.exp(self.persona_context_w[0]) / torch.sum(torch.exp(self.persona_context_w))
            w2 = torch.exp(self.persona_context_w[1]) / torch.sum(torch.exp(self.persona_context_w))
            w3 = torch.exp(self.persona_context_w[2]) / torch.sum(torch.exp(self.persona_context_w))
            w4 = torch.exp(self.persona_context_w[3]) / torch.sum(torch.exp(self.persona_context_w))
            w5 = torch.exp(self.persona_context_w[4]) / torch.sum(torch.exp(self.persona_context_w))
            # encoder_outputs.last_hidden_state = w1 * encoder_outputs.last_hidden_state + w2 * context + w3 * persona + w4 * context_refer + w5 * refer
            encoder_outputs.last_hidden_state = self.norm(w1 * encoder_outputs.last_hidden_state + w2 * context + w3 * persona + w4 * context_refer + w5 * refer)
    

        decoder_outputs = self.model.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(decoder_outputs.last_hidden_state) + self.final_logits_bias
        my_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=self.my_encoder_outputs,
            past_key_values=self.my_past,
            return_dict=return_dict,
        )
        my_logits = self.lm_head(my_outputs[0]) + self.final_logits_bias
        alpha = 0.075
        # alpha = 0
        lm_logits = (1+alpha)*lm_logits - alpha*my_logits
        self.predict_strategy(lm_logits, encoded_info)
        self.generation_strategy = encoded_info['pred_strat_id']
        decoder_input_ids = torch.cat(
            [decoder_input_ids, encoded_info['pred_strat_id'][..., None] + len(self.toker) - 8], dim=-1)
        # print(encoded_info)
        # print(decoder_input_ids)
        # print(self.toker.batch_decode(decoder_input_ids))
        assert 'max_length' in kwargs
        kwargs['max_length'] = kwargs['max_length'] + decoder_input_ids.size(1)
        kwargs['use_cache'] = True

        if len(self.toker) > self.toker.vocab_size:
            bad_words_ids = [[i] for i in range(self.toker.vocab_size, len(self.toker))]
            kwargs['bad_words_ids'] = bad_words_ids

        generations = super().generate(
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            **kwargs
        )
        # print(encoded_info)
        encoded_info['persona'] = persona_input_ids
        # print(persona_input_ids)
        return encoded_info, generations[:, decoder_input_ids.size(1):]
