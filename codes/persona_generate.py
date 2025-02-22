import os
os.chdir('/data/wanghongkai/ds/PAL-main/codes')
from transformer import BartForConditionalGeneration
from transformers import BartTokenizer #,BartForConditionalGeneration
import torch
import json
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm

class DialogueInferenceDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=384):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        conversation = item['conversation_history']
        usr2_personality = item['usr2_personality']

        # 1. 拼接对话文本
        dialogue_text = " <sep> ".join([turn['text'] for turn in conversation])
        
        # 2. 生成角色标签 (假设 usr1 为 0，usr2 为 1)
        roles = []
        for turn in conversation:
            speaker = turn['speaker']
            role = 0 if speaker == 'usr1' else 1
            roles.append(role)

        # 3. Tokenize 对话文本
        inputs = self.tokenizer(dialogue_text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")

        # 4. 处理角色信息：扩展句子级别角色标签到每个token
        tokenized_input = self.tokenizer([turn['text'] for turn in conversation], return_attention_mask=False, add_special_tokens=False)
        input_ids = tokenized_input['input_ids']

        expanded_roles = []
        role_idx = 0
        # for token_list in input_ids:
        #     expanded_roles.extend([roles[role_idx]] * len(token_list))
        #     role_idx += 1
        # # expanded_roles = expanded_roles[:self.max_length]  # 截断到 max_length
        # # 5. 将角色标签截断或填充至与 input_ids 长度相同
        # input_length = inputs['input_ids'].size(1)  # 获取 input_ids 的长度
        # if len(expanded_roles) > input_length:
        #     expanded_roles = expanded_roles[:input_length]  # 截断到 input_length
        # else:
        #     expanded_roles.extend([0] * (input_length - len(expanded_roles)))  # 填充到 input_length
        for i, token_list in enumerate(input_ids):
            expanded_roles.extend([roles[role_idx]] * len(token_list))
            # 如果不是最后一个对话，则添加 <sep> 的角色标签
            if i < len(input_ids) - 1:
                expanded_roles.append(roles[role_idx])  # 将 <sep> 归于当前 speaker
            role_idx += 1

        # 5. 将角色标签截断或填充至与 input_ids 长度相同
        input_length = inputs['input_ids'].size(1)  # 获取 input_ids 的长度
        if len(expanded_roles) > input_length:
            expanded_roles = expanded_roles[:input_length]  # 截断到 input_length
        else:
            expanded_roles.extend([0] * (input_length - len(expanded_roles)))  # 填充到 input_length
        # 将扩展后的角色标签转为 tensor
        role_ids = torch.tensor(expanded_roles).long()
        # # 6. 生成角色的 attention_mask，与 input_ids 保持一致
        # role_attention_mask = inputs['attention_mask'].clone()
        
        # 5. Tokenize 输出（User2 的 personality）
        personality_output = " ".join([f"<usr2_personality> {p}" for p in usr2_personality])
        labels = self.tokenizer(personality_output, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")

        return {
            'input_ids': inputs['input_ids'].squeeze(), # 去掉多余的batch_size维度
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': labels['input_ids'].squeeze(),
            'token_type_ids': role_ids,  # 返回对齐好的角色标签
        }
    
# def predict(row, model, tokenizer):
#     device=model.device
#     max_length=64
#     # 将输入对话转为单一文本
#     input_text = " <sep> ".join(row['input'])
#     # 生成token_type_ids，0表示User1的部分，1表示User2的部分
#     role_ids = []
#     user1_len = len(tokenizer(row['input'][0])['input_ids'])
#     user2_len = len(tokenizer(row['input'][1])['input_ids'])

#     role_ids.extend([0] * user1_len)  # User1的token_type_ids为0
#     role_ids.extend([1] * user2_len)  # User2的token_type_ids为1
#     role_ids += [0] * (max_length - len(role_ids))  # 填充至max_length
#     role_ids = torch.tensor(role_ids).long().unsqueeze(0).to(device)

#     # 对输入进行tokenize
#     inputs = tokenizer(input_text, return_tensors="pt", padding="max_length", max_length=max_length, truncation=True).to(device)
    

#     # 使用模型生成输出
#     outputs = model.generate(inputs['input_ids'], token_type_ids=role_ids, max_length=max_length, num_beams=5, early_stopping=True)
    
#     # 解码模型输出，生成最终文本
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return response

    
# # 定义 predict 函数，使用模型生成推理结果
# def predict(row, model, tokenizer):

#     device=model.device
#     conversation = row['conversation_history']

#     # 1. 拼接对话文本 h78
#     dialogue_text = " <sep> ".join([turn['text'] for turn in conversation])

#     # 2. 生成角色标签 (假设 usr1 为 0，usr2 为 1)
#     roles = []
#     for turn in conversation:
#         speaker = turn['speaker']
#         role = 0 if speaker == 'usr1' else 1
#         roles.append(role)

#     # 3. Tokenize 对话文本
#     inputs = tokenizer(dialogue_text, max_length=384, padding="max_length", truncation=True, return_tensors="pt").to(device)

#     # 4. 处理角色信息：扩展句子级别角色标签到每个token
#     tokenized_input = tokenizer([turn['text'] for turn in conversation], return_attention_mask=False, add_special_tokens=False)
#     input_ids = tokenized_input['input_ids']

#     expanded_roles = []
#     role_idx = 0
#     for i, token_list in enumerate(input_ids):
#         expanded_roles.extend([roles[role_idx]] * len(token_list))
#         # 如果不是最后一个对话，则添加 <sep> 的角色标签
#         if i < len(input_ids) - 1:
#             expanded_roles.append(roles[role_idx])  # 将 <sep> 归于当前 speaker
#         role_idx += 1

#     # 5. 将角色标签截断或填充至与 input_ids 长度相同
#     input_length = inputs['input_ids'].size(1)  # 获取 input_ids 的长度
#     if len(expanded_roles) > input_length:
#         expanded_roles = expanded_roles[:input_length]  # 截断到 input_length
#     else:
#         expanded_roles.extend([0] * (input_length - len(expanded_roles)))  # 填充到 input_length

#     # 将扩展后的角色标签转为 tensor
#     role_ids = torch.tensor(expanded_roles).long().unsqueeze(0).to(device)
#     result= {
#     'input_ids': inputs['input_ids'], # 去掉多余的batch_size维度
#     'attention_mask': inputs['attention_mask'],
#     'token_type_ids': role_ids,  # 返回对齐好的角色标签
#     }
#     output = model.generate(
#     input_ids=result['input_ids'],
#     attention_mask=result['attention_mask'],
#     token_type_ids=result['token_type_ids'],  # 在 generate 里传入 token_type_ids
#     max_length=384,
#     num_beams=10,
#     early_stopping=True
#     )
#     response = tokenizer.decode(output[0], skip_special_tokens=False)
#     return response

# # 定义 predict 函数，使用模型生成推理结果
# def predict(row, model, tokenizer):
#     device=model.device
#     # 将输入对话转为单一文本
#     input_text = " <sep> ".join(row['input'])
    
#     # 对输入进行tokenize
#     inputs = tokenizer(input_text, return_tensors="pt", max_length=64, truncation=True).to(device)
    
#     # 使用模型生成输出
#     outputs = model.generate(inputs['input_ids'], max_length=32, num_beams=5, early_stopping=True)
    
#     # 解码模型输出，生成最终文本
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return response

# # 定义 predict 函数，使用模型生成推理结果
# def predict(row, model, tokenizer):

#     device=model.device
#     conversation = row['conversation_history']

#     # 1. 拼接对话文本
#     dialogue_text = " <sep> ".join([turn['text'] for turn in conversation])

#     inputs = tokenizer(dialogue_text, return_tensors="pt", max_length=384, truncation=True).to(device)
    
#     # 使用模型生成输出
#     outputs = model.generate(inputs['input_ids'], max_length=64, num_beams=5, early_stopping=True)
    
#     # 解码模型输出，生成最终文本
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return response

# def predict(messages, model, tokenizer):
#     device = "cuda"
#     # text = tokenizer.apply_chat_template(
#     #     messages,
#     #     tokenize=False,
#     #     add_generation_prompt=True
#     # )
#     text=f"Instruction: Based on the given conversation, generate User2's personality information.\nInput: {messages['input']}\n",
#     model_inputs = tokenizer(text, return_tensors="pt").to(device)

#     generated_ids = model.generate(
#         model_inputs.input_ids,
#         max_new_tokens=512
#     )
#     generated_ids = [
#         output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
#     ]
    
#     response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
     
#     return response

# 定义 predict 函数，使用模型生成推理结果
def predict(conversation, model, tokenizer):

    device=model.device
    # conversation = row['conversation_history']

    # 1. 拼接对话文本 
    dialogue_text = " <sep> ".join([turn['content'] for turn in conversation])

    # 2. 生成角色标签 (假设 usr1 为 0，usr2 为 1)
    roles = []
    for turn in conversation:
        speaker = turn['speaker']
        role = 0 if speaker == 'supporter' else 1
        roles.append(role)

    # 3. Tokenize 对话文本
    inputs = tokenizer(dialogue_text, max_length=384, padding="max_length", truncation=True, return_tensors="pt").to(device)

    # 4. 处理角色信息：扩展句子级别角色标签到每个token
    tokenized_input = tokenizer([turn['content'] for turn in conversation], return_attention_mask=False, add_special_tokens=False)
    input_ids = tokenized_input['input_ids']

    expanded_roles = []
    role_idx = 0
    for i, token_list in enumerate(input_ids):
        expanded_roles.extend([roles[role_idx]] * len(token_list))
        # 如果不是最后一个对话，则添加 <sep> 的角色标签
        if i < len(input_ids) - 1:
            expanded_roles.append(roles[role_idx])  # 将 <sep> 归于当前 speaker
        role_idx += 1
    # for token_list in input_ids:
    #     expanded_roles.extend([roles[role_idx]] * len(token_list))
    #     role_idx += 1
    # expanded_roles = expanded_roles[:self.max_length]  # 截断到 max_length
    # 5. 将角色标签截断或填充至与 input_ids 长度相同
    input_length = inputs['input_ids'].size(1)  # 获取 input_ids 的长度
    if len(expanded_roles) > input_length:
        expanded_roles = expanded_roles[:input_length]  # 截断到 input_length
    else:
        expanded_roles.extend([0] * (input_length - len(expanded_roles)))  # 填充到 input_length

    # 将扩展后的角色标签转为 tensor
    role_ids = torch.tensor(expanded_roles).long().unsqueeze(0).to(device)
    result= {
    'input_ids': inputs['input_ids'], # 去掉多余的batch_size维度
    'attention_mask': inputs['attention_mask'],
    'token_type_ids': role_ids,  # 返回对齐好的角色标签
    }
    output = model.generate(
    input_ids=result['input_ids'],
    attention_mask=result['attention_mask'],
    token_type_ids=result['token_type_ids'],  # 在 generate 里传入 token_type_ids
    max_length=384,
    num_beams=10,
    early_stopping=True
    )
    response = tokenizer.decode(output[0], skip_special_tokens=False)
    return response

# def load_data(data_type):
#     with open(f'/data/wanghongkai/ds/PAL-main/codes/window_{data_type}_data.json', 'r', encoding='utf-8') as f:# ./ window_{data_type}_data
#         data = json.load(f)
#     return data

model = BartForConditionalGeneration.from_pretrained('/data/wanghongkai/ds/PAL-main/codes/persona_extraction/checkpoint-11800')
tokenizer=BartTokenizer.from_pretrained('/data/wanghongkai/ds/PAL-main/codes/persona_extraction/checkpoint-11800')
model.eval()

with open('/data/wanghongkai/ds/PAL-main/window_ESConv.json', 'r', encoding='utf-8') as f:
    test_data = json.load(f)

# sample_data=data[:10]
# test_data=sample_data
# temp_data = [{'conversation_list': row['dialog']} for row in sample_data]

# test_data = [{"conversation_history": conv} for item in temp_data for conv in item['conversation_list']]
# # 测试数据
# test_data = load_data('test')
test_data=test_data[:10]
test_df = pd.DataFrame(test_data)  

# from transformers import AutoTokenizer, AutoModelForCausalLM
# from peft import PeftModel
# model = AutoModelForCausalLM.from_pretrained("/data/multimodal_LLMs/Llama-2-7b-hf/", device_map="auto", torch_dtype=torch.bfloat16)
# tokenizer = AutoTokenizer.from_pretrained("/data/multimodal_LLMs/Llama-2-7b-hf/", use_fast=False, trust_remote_code=True, use_cache=False,)
# tokenizer.pad_token = tokenizer.unk_token
# # 加载训练好的Lora模型，将下面的checkpointXXX替换为实际的checkpoint文件名名称
# model = PeftModel.from_pretrained(model, model_id="./output/Llama2-7b-full_history/checkpoint-1000")
# test_df = pd.read_json("./instruct_test.jsonl", lines=True)
# test_df=test_df.header()
# test_text_list = []
batch_size = 4
# 获取数据的总行数
total_samples = len(test_df)

# # 使用 tqdm 进度条
# for start in tqdm(range(0, total_samples, batch_size), desc="Processing Batches"):
#     end = min(start + batch_size, total_samples)
#     batch = test_df.iloc[start:end]  # 选取当前批次的数据

#     # 对于当前批次的每一行进行推理
#     for index, row in batch.iterrows():
#         instruction = "Based on the given dialogue, generate User2's personality information."
#         input_value = row['input']
#         label=row['output']

#         # 通过模型生成推理结果
#         persona = predict(row, model, tokenizer)
#         persona_list = [t.strip() for t in persona.split('<usr2_personality>') if t.strip()]
        
#         test_text_list.append({'conversation_history': input_value, 'persona': persona_list, 'label': label})

# with open('./sample_infer.json', 'w') as f:
#     json.dump(test_text_list, f, indent=4)

# # 假设 test_data 是原始数据集
# batch_size = 8
# total_samples = len(test_data)  # 获取数据的总行数

# # 使用 tqdm 进度条
# for start in tqdm(range(0, total_samples, batch_size), desc="Processing Batches"):
#     end = min(start + batch_size, total_samples)
#     batch = test_data[start:end]  # 选取当前批次的数据

#     # 对于当前批次的每一条数据进行处理
#     for row in batch:
#         persona_list = []
#         for conversation in row["dialog"]:
#             # 调用预测函数获取结果
#             persona = predict(conversation, model, tokenizer)
#             cleaned_text = persona.replace('</s>', '').replace('<s>', '').strip()
#             temp_list = [t.strip() for t in cleaned_text.split('<usr2_personality>') if t.strip()]
#             persona_list.extend(temp_list)

#         # 将拼接后的列表赋值为新的字段 "persona_list"
#         row["persona_list"] = persona_list  # 直接修改原始的 test_data 列表

# 先在 test_df 中创建 persona_list 列，初始化为空列表
test_df['persona_list'] = [[] for _ in range(len(test_df))]
# 使用 tqdm 进度条
for start in tqdm(range(0, total_samples, batch_size), desc="Processing Batches"):
    end = min(start + batch_size, total_samples)
    batch = test_df.iloc[start:end]  # 选取当前批次的数据

    # 对于当前批次的每一行进行推理
    for index, row in batch.iterrows():
        persona_list = []
        for conversation in row["window_dialog"]:
            # 调用预测函数获取结果
            persona = predict(conversation, model, tokenizer)
            cleaned_text = persona.replace('</s>', '').replace('<s>', '').strip()
            temp_list = [t.strip() for t in cleaned_text.split('<usr2_personality>') if t.strip()]
            persona_list.extend(temp_list)
        # 更新 test_df 中的对应行
        test_df.at[index, 'persona_list'] = persona_list

test_df = test_df.drop(columns=['window_dialog'])
# 将改动后的 test_df 转换回 test_data 的格式
test_data = test_df.to_dict(orient='records')

with open('./sample_infer.json', 'w') as f:
    json.dump(test_data, f, indent=4)
# 原来的 ！！
# with open('/data/wanghongkai/ds/PAL-main/my_PESConv.json', 'w') as f:
#     json.dump(test_data, f, indent=4)

# # 使用 tqdm 进度条
# for start in tqdm(range(0, total_samples, batch_size), desc="Processing Batches"):
#     end = min(start + batch_size, total_samples)
#     batch = test_df.iloc[start:end]  # 选取当前批次的数据

#     # 对于当前批次的每一行进行推理
#     for index, row in batch.iterrows():
#         sentence_pair = row['input']
#         label = row['label']
        
#         # 通过模型生成推理结果
#         persona = predict(row, model, tokenizer)      
#         test_text_list.append({'sentence_pair': sentence_pair, 'persona': persona, 'label': label})

# with open('./sentence_infer_no_role.json', 'w') as f:
#     json.dump(test_text_list, f, indent=4)

# # 使用 tqdm 进度条
# for start in tqdm(range(0, total_samples, batch_size), desc="Processing Batches"):
#     end = min(start + batch_size, total_samples)
#     batch = test_df.iloc[start:end]  # 选取当前批次的数据

#     # 对于当前批次的每一行进行推理
#     for index, row in batch.iterrows():
#         conversation = row['conversation_history']
#         label = row['usr2_personality']
        
#         # 通过模型生成推理结果
#         persona = predict(row, model, tokenizer)
#         cleaned_text = persona.replace('</s>', '').replace('<s>', '').strip()
#         persona_list = [t.strip() for t in cleaned_text.split('<usr2_personality>') if t.strip()]
        
#         test_text_list.append({'conversation_history': conversation, 'persona': persona_list, 'label': label})

# with open('./sample_infer.json', 'w') as f:
#     json.dump(test_text_list, f, indent=4)

