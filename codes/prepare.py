# coding=utf-8

import argparse
import json
import shutil
import pickle
import os
import logging
import multiprocessing as mp
from os.path import dirname, exists, join

import torch
import tqdm
from inputters import inputters
from utils.building_utils import build_model, boolean_string

parser = argparse.ArgumentParser()
parser.add_argument('--config_name', type=str, required=True)
parser.add_argument('--inputter_name', type=str, required=True)
parser.add_argument('--train_input_file', type=str, required=True)
parser.add_argument('--max_input_length', type=int, default=150, help='discard data longer than this')
parser.add_argument('--max_decoder_input_length', type=int, default=None, help='discard data longer than this')
parser.add_argument('--max_knowledge_length', type=int, default=None, help='discard data longer than this')
parser.add_argument('--label_num', type=int, default=None)
parser.add_argument('--only_encode', action='store_true', help='only do encoding')
parser.add_argument('--single_processing', action='store_true', help='do not use multiprocessing')

# persona model
parser.add_argument('--use_all_persona', type=boolean_string, help="if need to use all persona or generate",
                    default=True)
parser.add_argument('--encode_context', type=boolean_string, default=True)
args = parser.parse_args()

names = {
    'inputter_name': args.inputter_name,
    'config_name': args.config_name,
}

inputter = inputters[args.inputter_name](args.use_all_persona)
toker = build_model(only_toker=True, **names)

with open(args.train_input_file) as f:
    reader = f.readlines()
    
if not os.path.exists(f'./DATA'):
    os.mkdir(f'./DATA')
save_dir = f'./DATA/{args.inputter_name}.{args.config_name}_persona_attention_final_rebuttal'
if not exists(save_dir):
    os.mkdir(save_dir)


kwargs = {
    'max_input_length': args.max_input_length,
    'max_decoder_input_length': args.max_decoder_input_length,
    'max_knowledge_length': args.max_knowledge_length,
    'label_num': args.label_num,
    'only_encode': args.only_encode,
    'stage': "training"
}


def process_data(line):
    data = json.loads(line)
    inputs = inputter.convert_data_to_inputs(
        data=data,
        toker=toker,
        use_all_persona=inputter.use_all_persona,
        encode_context=args.encode_context,
        **kwargs
    )
    features = inputter.convert_inputs_to_features(
        inputs=inputs,
        toker=toker,
        encode_context=args.encode_context,
        **kwargs,
    )
    return features


processed_data = []
print("single processing: ", args.single_processing)
if args.single_processing:
    for features in map(process_data, tqdm.tqdm(reader, total=len(reader))):
        processed_data.extend(features)
else:
    with mp.Pool(processes=mp.cpu_count()) as pool:
        for features in pool.imap(process_data, tqdm.tqdm(reader, total=len(reader))):
            processed_data.extend(features)
# print(processed_data[0].labels)


# save data
data_path =f'{save_dir}/data_select.pkl'
with open(data_path, 'wb') as file:
    pickle.dump(processed_data, file)
kwargs.update({'n_examples': len(processed_data)})
# save relevant information to reproduce
with open(f'{save_dir}/meta_select.json', 'w') as writer:
    json.dump(kwargs, writer, indent=4)
torch.save(toker, f'{save_dir}/tokenizer_select.pt')
