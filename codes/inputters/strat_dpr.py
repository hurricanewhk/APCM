# coding=utf-8
import sys
sys.path.append('/data/wanghongkai/ds/PAL-main/codes')
import json
import tqdm
import time
import torch
import pickle

from typing import List
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers import AutoTokenizer,AutoModel
from scipy.spatial.distance import cosine
import numpy as np
import random
from functools import partial
from torch.utils.data import DataLoader, Sampler, Dataset
from torch.nn.utils.rnn import pad_sequence
from math import ceil
from inputters.inputter_utils import _norm, BucketSampler, BucketingDataLoader, DistributedBucketingDataLoader, \
    my_pad_sequence
# from .PARAMS import GOLDEN_TRUTH
GOLDEN_TRUTH=False
from transformers import BartTokenizer, BartForConditionalGeneration
# from .train_bart import LitModel


class Inputter(object):
    def __init__(self, use_all_persona):
        # prepare
        self.convert_data_to_inputs = convert_data_to_inputs
        self.convert_inputs_to_features = convert_inputs_to_features

        # train
        self.train_sampler = BucketSampler
        self.train_dataset = FeatureDataset
        self.train_dataloader = BucketingDataLoader
        self.train_distributed_dataloader = DistributedBucketingDataLoader

        # valid
        self.valid_dataloader = DynamicBatchingLoader

        # infer
        self.prepare_infer_batch = prepare_infer_batch
        self.infer_dataloader = get_infer_batch

        # persona
        self.use_all_persona = use_all_persona


# basic utils
class InputFeatures(object):
    def __init__(
            self,
            input_ids, refer_input_ids,
            decoder_input_ids, labels, persona_input_ids
    ):
        self.input_ids = input_ids
        self.input_length = len(input_ids)

        self.decoder_input_ids = decoder_input_ids
        self.decoder_input_length = len(decoder_input_ids)
        self.labels = labels
        self.persona_input_ids = persona_input_ids
        self.persona_input_length = len(persona_input_ids)
        self.padding_length = max(self.input_length, self.persona_input_length)

        self.refer_input_ids = refer_input_ids[:self.padding_length]
        self.input_len = self.input_length + self.decoder_input_length


def featurize(
        bos, eos, persona, refer,
        context, max_input_length,
        response, max_decoder_input_length, strat_id, encode_context, toker, concat_ctx_persona=False
):
    if concat_ctx_persona:
        # if encode_context is True, we need to encode context here
        if not encode_context:
            # already encoded
            context = [c + [eos] for c in context]
            input_ids = sum(context, [])[:-1]
            input_ids = input_ids[-(max_input_length - len(persona)):]
            input_ids = persona + input_ids
            persona_input_ids = persona
            labels = ([strat_id] + response + [eos])[:max_decoder_input_length + 1]
            decoder_input_ids = [bos] + labels[:-1]
        else:
            # TODO: encode context here
            strat_id = toker.convert_tokens_to_ids(toker.tokenize(strat_id))
            context = '</s> <s>'.join(context)
            context = toker(context)
            persona = toker(persona).input_ids
            persona_input_ids = persona
            input_ids = context.input_ids
            input_ids = input_ids[-(max_input_length - len(persona)):]
            input_ids = persona + input_ids
            response = toker(response).input_ids
            labels = (strat_id + response + [eos])[:max_decoder_input_length + 1]
            decoder_input_ids = [bos] + labels[:-1]
            # print(decoder_input_ids)
    else:
        # if encode_context is True, we need to encode context here
        if not encode_context:
            # already encoded
            context = [c + [eos] for c in context]
            input_ids = sum(context, [])[:-1]
            input_ids = input_ids[-max_input_length:]
            persona_input_ids = persona
            labels = ([strat_id] + response + [eos])[:max_decoder_input_length + 1]
            decoder_input_ids = [bos] + labels[:-1]
        else:
            # TODO: encode context here
            strat_id = toker.convert_tokens_to_ids(toker.tokenize(strat_id))
            context = '</s> <s>'.join(context)
            context = toker(context)
            persona_input_ids = toker(persona).input_ids
            input_ids = context.input_ids
            input_ids = input_ids[-max_input_length:]
            response = toker(response).input_ids
            labels = (strat_id + response + [eos])[:max_decoder_input_length + 1]
            decoder_input_ids = [bos] + labels[:-1]
            refer_input_ids = toker(refer).input_ids[-max_input_length:]

    assert len(decoder_input_ids) == len(labels), decoder_input_ids[1:] == labels[:-1]

    return InputFeatures(
        input_ids, refer_input_ids,
        decoder_input_ids, labels, persona_input_ids
    )


def filter_persona(infer_res):
    infer_res = infer_res.replace("</s>", "")
    infer_res = infer_res.replace("<s>", "")
    infer_res = infer_res.replace("<pad>", "")
    infer_res = infer_res.split("<persona>")
    for j in infer_res:
        if j.lower().count("my favorite color is"):
            infer_res.remove(j)
        elif j.lower().count("my favorite band"):
            infer_res.remove(j)
        elif len(j.split(' ')) < 2 or len(j.split(' ')) > 25:
            infer_res.remove(j)
    return "<persona>".join(infer_res) + "<persona>"


def calculate_similarity(embedding1, embedding2):
    return 1 - cosine(embedding1, embedding2)

def select_relevant_persona(persona_list, context, threshold, tokenizer, model):
    # 编码上下文
    context_inputs = tokenizer(context, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        context_embedding = model(**context_inputs, output_hidden_states=True, return_dict=True).pooler_output
    relevant_persona = []

    for persona in persona_list:
        # 编码个性描述
        persona_inputs = tokenizer(persona, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            persona_embedding = model(**persona_inputs, output_hidden_states=True, return_dict=True).pooler_output
        
        # 计算相似度
        similarity = calculate_similarity(context_embedding[0], persona_embedding[0])
        if similarity > threshold:
            relevant_persona.append(persona)
    
    return  relevant_persona
# def select_relevant_persona(persona_list, post, threshold, tokenizer, model):
#     relevant_persona = []

#     for persona in persona_list:
#         # 将 persona 和 post 组成句子对
#         sentence_pair = (persona, post)
        
#         # 使用 tokenizer 将句子对转换为模型输入
#         inputs = tokenizer(sentence_pair[0], sentence_pair[1], return_tensors='pt')
        
#         # 使用模型进行推理，得到匹配分数
#         with torch.no_grad():
#             outputs = model(**inputs)
#             logits = outputs.logits
#             score = torch.softmax(logits, dim=1)[0][1].item()
        
#         # 如果匹配分数超过阈值，将 persona 添加到输出列表
#         if score > threshold:
#             relevant_persona.append(persona)
    
#     return  relevant_persona

def convert_data_to_inputs(data, toker: PreTrainedTokenizer, use_all_persona, encode_context,**kwargs):# tokenizer=None, model=None,
    # whether to use prompt
    use_prompt = False
    add_speaker = True
    with_persona = True
    # encode_context = True # we don't need to convert context to id
    assert ((encode_context and add_speaker) or (not encode_context and not add_speaker))
    process = lambda x: toker.convert_tokens_to_ids(toker.tokenize(x))

    dialog = data['dialog']
    user_number = 0
    persona = None
    persona_list = data['persona_list']
    # for similarity
    if use_all_persona:
        persona = data['persona']
        if use_prompt:
            persona = persona.replace('<input>', '')
            persona = persona.replace('<persona>', '', 1).strip()
            persona = "<persona>".join(persona.split('<persona>'))
            persona = "Persona Information:\n" + persona + '\n' + "\nDialogue:\n"
        if not encode_context:
            persona = process(persona)
    inputs = []
    context = []

    for i in range(len(dialog)):
        text = _norm(dialog[i]['text'])
        if dialog[i]['speaker'] != 'sys':
            if add_speaker:
                text = "Persona:" + text #为什么是Persona？

        if not encode_context:
            text = process(text)

        if dialog[i]['speaker'] != 'sys':
            user_number += 1

        if dialog[i]['speaker'] == 'sys':
            if not encode_context:
                strat_id = process('[' + dialog[i]['strategy'] + ']')
                assert len(strat_id) == 1
                strat_id = strat_id[0]
            else:
                strat_id = '[' + dialog[i]['strategy'] + ']'
        # print(kwargs['stage'])
        if i > 0 and dialog[i]['speaker'] == 'sys' and (kwargs['stage'] == 'training' or dialog[i - 1]['speaker'] != 'sys'):
            last_text = _norm(dialog[i - 1]['text'])
            if not use_all_persona:
                if use_prompt:
                    if user_number > 2:
                        persona = persona_list[user_number - 3]
                        persona = persona.replace('<input>', '')
                        persona = persona.replace('<persona>', '', 1).strip()
                        persona = "</s> <s>".join([p.strip() for p in persona.split('<persona>')])
                        if not with_persona:
                            persona = ""
                        persona = "Persona Information:\n" + persona + '</s> <s>' + "Dialogue:\n"
                    else:
                        persona = "Persona Information:\n</s> <s>Dialogue:\n"
                else:
                    if user_number > 2:
                        # persona_accum=persona_list[1:user_number-1]
                        # persona=''.join(persona_accum)
                        persona=persona_list[user_number-3]# test
                        # persona=select_relevant_persona(persona_accum,last_text,0.5,tokenizer,model)
                        # persona='. '.join(persona)
                        # if len(persona)==0:
                        #     # persona = "<input>"
                        #     persona = persona_list[user_number - 3]# "<input>"才是更好的？？
                        # else:
                            # persona='. '.join(persona)
                        if not with_persona:
                            persona = "<input>"
                        if len(dialog[i]) > 3:
                            ICL = ""
                            for dpr in dialog[i]['dpr'][:3]:
                                ICL += '<persona> Persona: ' + dpr[0] + ' System: [' + dpr[1] + '] ' + dpr[2] + "<input>"
                                # ICL += 'System: [' + dpr[1] + '] ' + dpr[2] + "<input>"
                            # ICL += persona
                        else:
                            ICL = "<input>"
                    else:
                        # persona=data['situation']
                        ICL = "<input>"
                        persona = "<input>"
                if not encode_context:
                    persona = process(persona)
            history_dialog = context.copy()
            if add_speaker:
                history_dialog += ["System:"]
            # print("his: ", history_dialog)
            # print("res: ", text)
            # print("strat: ", sys_strat)
            # print("persona: ", persona)
            res = {
                'last_text': last_text,
                'context': history_dialog,
                'response': text,
                'strat_id': strat_id,
                'persona': persona,
                'refer':ICL
            }

            inputs.append(res)

        if dialog[i]['speaker'] == 'sys':
            if add_speaker:
                text = "System:" + strat_id + text
            else:
                text = [strat_id] + text
        context = context + [text]
        # print(context)

    return inputs


def convert_inputs_to_features(inputs, toker, encode_context, **kwargs):
    # assert encode_context
    if len(inputs) == 0:
        return []

    assert kwargs.get('max_input_length', None) is not None, 'you should give max_input_length'
    max_input_length = kwargs.get('max_input_length')
    assert kwargs.get('max_decoder_input_length', None) is not None, 'you should give max_decoder_input_length'
    max_decoder_input_length = kwargs.get('max_decoder_input_length')

    pad = toker.pad_token_id
    if pad is None:
        pad = toker.eos_token_id
        assert pad is not None, 'either pad_token_id or eos_token_id should be provided'
    bos = toker.bos_token_id
    if bos is None:
        bos = toker.cls_token_id
        assert bos is not None, 'either bos_token_id or cls_token_id should be provided'
    eos = toker.eos_token_id
    if eos is None:
        eos = toker.sep_token_id
        assert eos is not None, 'either eos_token_id or sep_token_id should be provided'

    features = []
    for i in range(len(inputs)):
        ipt = inputs[i]
        feat = featurize(
            bos, eos, ipt['persona'],ipt['refer'],
            ipt['context'], max_input_length,
            ipt['response'], max_decoder_input_length, ipt['strat_id'], encode_context, toker
        )
        features.append(feat)
    return features


# for training
class FeatureDataset(Dataset):
    def __init__(self, features):
        self.features = features

    def __getitem__(self, i):
        return self.features[i]

    def __len__(self):
        return len(self.features)

    @staticmethod
    def collate(features: List[InputFeatures], toker: PreTrainedTokenizer, infer=False):
        pad = toker.pad_token_id
        if pad is None:
            pad = toker.eos_token_id
            assert pad is not None, 'either pad_token_id or eos_token_id should be provided'
        bos = toker.bos_token_id
        if bos is None:
            bos = toker.cls_token_id
            assert bos is not None, 'either bos_token_id or cls_token_id should be provided'
        eos = toker.eos_token_id
        if eos is None:
            eos = toker.sep_token_id
            assert eos is not None, 'either eos_token_id or sep_token_id should be provided'
        # max_len = max([f.padding_length for f in features])
        max_len=512
        input_ids = my_pad_sequence([torch.tensor(f.input_ids, dtype=torch.long) for f in features],
                                    batch_first=True, max_len=max_len, padding_value=pad)
        attention_mask = my_pad_sequence([torch.tensor([1.] * f.input_length, dtype=torch.float) for f in features],
                                         batch_first=True, max_len=max_len, padding_value=0.)
        input_length = torch.tensor([f.input_length for f in features], dtype=torch.long)

        persona_input_ids = my_pad_sequence([torch.tensor(f.persona_input_ids, dtype=torch.long) for f in features],
                                            batch_first=True, max_len=max_len, padding_value=pad)
        persona_attention_mask = my_pad_sequence(
            [torch.tensor([1.] * f.persona_input_length, dtype=torch.float) for f in features],
            batch_first=True, max_len=max_len, padding_value=0.)
        
        refer_input_ids = my_pad_sequence([torch.tensor(f.refer_input_ids, dtype=torch.long) for f in features],
                                          batch_first=True, max_len=max_len, padding_value=pad)
        refer_attention_mask = my_pad_sequence(
            [torch.tensor([1.] * len(f.refer_input_ids), dtype=torch.float) for f in features],
            batch_first=True, max_len=max_len, padding_value=0.)

        if not infer:
            decoder_input_ids = pad_sequence([torch.tensor(f.decoder_input_ids, dtype=torch.long) for f in features],
                                             batch_first=True, padding_value=pad)
            labels = pad_sequence([torch.tensor(f.labels, dtype=torch.long) for f in features],
                                  batch_first=True, padding_value=-100)
        else:
            decoder_input_ids = torch.tensor([[f.decoder_input_ids[0]] for f in features], dtype=torch.long)
            labels = None

        strat_id = torch.tensor([f.labels[0] for f in features], dtype=torch.long) - len(toker) + 8
        # print(strat_id)
        res = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            # 'input_length': input_length,

            'decoder_input_ids': decoder_input_ids,
            'labels': labels,
            "persona_input_ids": persona_input_ids,
            "persona_attention_mask": persona_attention_mask,
            'strat_id': strat_id,
            'refer_input_ids': refer_input_ids,
            'refer_attention_mask': refer_attention_mask,

        }

        return res


# for validation
class DynamicBatchingLoader(object):
    """ this loader takes raw text file, used for validate perplexity """

    def __init__(self, corpus_file, toker, batch_size, use_all_persona, encode_context, tokenizer=None, model=None, **kwargs):# tokenizer=None, model=None,
        self.corpus = corpus_file
        self.toker = toker
        self.bs = batch_size
        self.num_examples = self.get_len(corpus_file)
        self.kwargs = kwargs
        self.use_all_persona = use_all_persona
        self.encode_context = encode_context
        self.tokenizer=tokenizer
        self.model=model

    def __iter__(self, epoch=1):
        if epoch > 0:
            for epoch in range(epoch):
                yield from self._iter_epoch()
        else:
            while True:
                yield from self._iter_epoch()

    def __len__(self):
        return ceil(self.num_examples / self.bs)

    def _iter_epoch(self):
        try:
            with open(self.corpus, 'r', encoding="utf-8") as f:
                reader = f.readlines()

            features = []
            for line in tqdm.tqdm(reader, total=len(reader), desc=f"validating"):
                data = json.loads(line)
                inputs = convert_data_to_inputs(data, self.toker, self.use_all_persona, self.encode_context,                                              
                                                 **self.kwargs)
                # inputs = convert_data_to_inputs(data, self.toker, self.use_all_persona, self.encode_context,                                              
                #                                 self.tokenizer, self.model, **self.kwargs)
                features.extend(convert_inputs_to_features(inputs, self.toker, self.encode_context, **self.kwargs))
                if len(features) >= self.bs:
                    batch = self._batch_feature(features)
                    yield batch
                    features = []

            if len(features) > 0:
                batch = self._batch_feature(features)
                yield batch

        except StopIteration:
            pass

    def _batch_feature(self, features):
        return FeatureDataset.collate(features, self.toker)

    def get_len(self, corpus):
        with open(corpus, 'r', encoding="utf-8") as file:
            reader = [json.loads(line) for line in file]
        return sum(map(lambda x: len(list(filter(lambda y: y['speaker'] == 'sys', x['dialog'][1:]))), reader))


# for inference
def prepare_infer_batch(features, toker, interact=None):
    res = FeatureDataset.collate(features, toker, True)

    res['batch_size'] = res['input_ids'].size(0)

    other_res = res['other_res'] = {}
    other_res['acc_map'] = {
        'cls_strat_id': 'pred_strat_id',
    }

    if interact is None and GOLDEN_TRUTH:
        other_res['cls_strat_id'] = res.get('strat_id')
    else:
        other_res['cls_strat_id'] = res.pop('strat_id')

    return res

# def get_infer_batch(infer_input_file, toker, use_all_persona, encode_context, **kwargs):
#     assert 'infer_batch_size' in kwargs, 'you should give infer_batch_size'
#     infer_batch_size = kwargs.get('infer_batch_size')

#     # 直接读取已经预处理并保存的数据
#     with open(infer_input_file, 'rb') as f:
#         processed_data = pickle.load(f)
    
#     features = []
#     sample_ids = []
#     posts = []
#     references = []

#     for sample_id, inputs in tqdm.tqdm(enumerate(processed_data), total=len(processed_data), desc=f"inferring"):
#         tmp_features=convert_inputs_to_features(inputs,toker,encode_context,**kwargs)
#         for i in range(len(inputs)):
#             features.append(tmp_features[i])
#             ipt = inputs[i]
#             posts.append(ipt['last_text'])
#             if not encode_context:
#                 references.append(toker.decode(ipt['response']))
#             else:
#                 references.append(ipt['response'])
#             sample_ids.append(sample_id)

#             if len(sample_ids) == infer_batch_size:
#                 # print(sample_ids)
#                 yield prepare_infer_batch(features, toker), posts, references, sample_ids
#                 features = []
#                 sample_ids = []
#                 posts = []
#                 references = []

#     if len(sample_ids) > 0:
#         yield prepare_infer_batch(features, toker), posts, references, sample_ids
def get_infer_batch(infer_input_file, toker, use_all_persona, encode_context, **kwargs):
    assert 'infer_batch_size' in kwargs, 'you should give infer_batch_size'
    infer_batch_size = kwargs.get('infer_batch_size')

    with open(infer_input_file, 'r', encoding="utf-8") as f:
        reader = f.readlines()

    features = []
    sample_ids = []
    posts = []
    references = []
    for sample_id, line in tqdm.tqdm(enumerate(reader), total=len(reader), desc=f"inferring"):
        # print("in inputter, sample_id: {}".format(sample_id))
        # print("in inputter, infer_batch_size: {}".format(infer_batch_size))
        data = json.loads(line)
        inputs = convert_data_to_inputs(data, toker, use_all_persona, encode_context, **kwargs)
        tmp_features = convert_inputs_to_features(inputs, toker, encode_context, **kwargs)
        for i in range(len(inputs)):
            features.append(tmp_features[i])
            ipt = inputs[i]
            posts.append(ipt['last_text'])
            if not encode_context:
                references.append(toker.decode(ipt['response']))
            else:
                references.append(ipt['response'])
            sample_ids.append(sample_id)

            if len(sample_ids) == infer_batch_size:
                # print(sample_ids)
                yield prepare_infer_batch(features, toker), posts, references, sample_ids
                features = []
                sample_ids = []
                posts = []
                references = []

    if len(sample_ids) > 0:
        yield prepare_infer_batch(features, toker), posts, references, sample_ids

if __name__=="__main__":
    import pickle
    
    
    toker=AutoTokenizer.from_pretrained('/data/wanghongkai/ds/PAL-main/codes/Blenderbot_small-90M')
    # model=AutoModel.from_pretrained('/data/wanghongkai/ds/PAL-main/codes/Blenderbot_small-90M')
    tokenizer = AutoTokenizer.from_pretrained("/data/wanghongkai/ds/PAL-main/codes/SimCSE")   
    model = AutoModel.from_pretrained("/data/wanghongkai/ds/PAL-main/codes/SimCSE")
    # sim_model_path="/data/wanghongkai/ds/PAL-main/codes/SimCSE"
    with open(f'/data/wanghongkai/ds/PAL-main/sample.json', 'rb') as f:
     data = json.load(f)
    first_data=data[0]
    print(first_data)
    kwargs={'stage':'training'}
    inputs=convert_data_to_inputs(first_data,toker,False,True,tokenizer, model, **kwargs)
    print("suceess")
    # def get_infer_batch(infer_input_file, toker, use_all_persona, encode_context, **kwargs):
#     assert 'infer_batch_size' in kwargs, 'you should give infer_batch_size'
#     infer_batch_size = kwargs.get('infer_batch_size')

#     with open(infer_input_file, 'r', encoding="utf-8") as f:
#         reader = f.readlines()

#     features = []
#     sample_ids = []
#     posts = []
#     references = []
#     for sample_id, line in tqdm.tqdm(enumerate(reader), total=len(reader), desc=f"inferring"):
#         # print("in inputter, sample_id: {}".format(sample_id))
#         # print("in inputter, infer_batch_size: {}".format(infer_batch_size))
#         data = json.loads(line)
#         inputs = convert_data_to_inputs(data, toker, use_all_persona, encode_context, **kwargs)
#         tmp_features = convert_inputs_to_features(inputs, toker, encode_context, **kwargs)
#         for i in range(len(inputs)):
#             features.append(tmp_features[i])
#             ipt = inputs[i]
#             posts.append(ipt['last_text'])
#             if not encode_context:
#                 references.append(toker.decode(ipt['response']))
#             else:
#                 references.append(ipt['response'])
#             sample_ids.append(sample_id)

#             if len(sample_ids) == infer_batch_size:
#                 # print(sample_ids)
#                 yield prepare_infer_batch(features, toker), posts, references, sample_ids
#                 features = []
#                 sample_ids = []
#                 posts = []
#                 references = []

#     if len(sample_ids) > 0:
#         yield prepare_infer_batch(features, toker), posts, references, sample_ids