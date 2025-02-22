from transformers import BartTokenizer, Trainer, TrainingArguments
from transformer import BartForConditionalGeneration
# 自定义的模块导入
import swanlab
from swanlab.integration.huggingface import SwanLabCallback
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import math
import random
import json
import pandas as pd

class DialogueInferenceDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=768):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        conversation = item['conversation_history']
        usr2_personality = item['persona']

        # 1. 拼接对话文本
        dialogue_text = " <sep> ".join([turn['text'] for turn in conversation]) + tokenizer.eos_token
        
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
        personality_output = " ".join([f"<persona> {p}" for p in usr2_personality]) + tokenizer.eos_token
    
        labels = self.tokenizer(personality_output, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")

        return {
            'input_ids': inputs['input_ids'].squeeze(), # 去掉多余的batch_size维度
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': labels['input_ids'].squeeze(),
            'token_type_ids': role_ids,  # 返回对齐好的角色标签
        }

class BartWithRoleEmbedding(BartForConditionalGeneration):
    def __init__(self, config, role_embedding_dim=50):
        super(BartWithRoleEmbedding, self).__init__(config)
        # 角色嵌入，假设角色数为2（User1, User2）
        self.role_embedding = nn.Embedding(2, role_embedding_dim)
        # 原BART词嵌入维度
        self.word_embedding_dim = self.model.encoder.embed_tokens.embedding_dim
        # 线性层将词嵌入与角色嵌入维度结合，映射回原始的词嵌入维度
        self.fc = nn.Linear(self.word_embedding_dim + role_embedding_dim, self.word_embedding_dim)

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, labels=None, roles=None, role_attention_mask=None):
        # 通过BART模型获取词嵌入
        word_embeddings = self.model.encoder.embed_tokens(input_ids)

        # 获取角色嵌入
        role_embeddings = self.role_embedding(roles)

        # 拼接词嵌入和角色嵌入
        combined_embeddings = torch.cat((word_embeddings, role_embeddings), dim=-1)

        # 将拼接后的嵌入通过线性层映射回词嵌入维度
        combined_embeddings = self.fc(combined_embeddings)

        # 用组合后的嵌入进行BART编码器的前向传播
        encoder_outputs = self.model.encoder(
            # input_ids=input_ids,
            inputs_embeds=word_embeddings,
            attention_mask=attention_mask
        )

        # 调用BART原有的解码器继续处理
        outputs = super().forward(
            encoder_outputs=encoder_outputs.last_hidden_state,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels
        )

        return outputs


# 定义 predict 函数，使用模型生成推理结果
def predict(row, model, tokenizer):

    device=model.device
    conversation = row['conversation_history']

    # 1. 拼接对话文本
    dialogue_text = " <sep> ".join([turn['text'] for turn in conversation]) + tokenizer.eos_token

    # 2. 生成角色标签 (假设 usr1 为 0，usr2 为 1)
    roles = []
    for turn in conversation:
        speaker = turn['speaker']
        role = 0 if speaker == 'usr1' else 1
        roles.append(role)

    # 3. Tokenize 对话文本
    inputs = tokenizer(dialogue_text, max_length=768, padding="max_length", truncation=True, return_tensors="pt").to(device)

    # 4. 处理角色信息：扩展句子级别角色标签到每个token
    tokenized_input = tokenizer([turn['text'] for turn in conversation], return_attention_mask=False, add_special_tokens=False)
    input_ids = tokenized_input['input_ids']

    expanded_roles = []
    role_idx = 0
    for i, token_list in enumerate(input_ids):
        expanded_roles.extend([roles[role_idx]] * len(token_list))
        # 如果不是最后一个对话，则添加 <sep> 的角色标签
        if i < len(input_ids) - 1:
            expanded_roles.append(roles[role_idx])  # 将 <sep> 归于当前 speaker
        role_idx += 1
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
    max_length=128,
    num_beams=10,
    early_stopping=True
    )
    response = tokenizer.decode(output[0], skip_special_tokens=False)
    return response

    
def load_data(data_type):
    with open(f'./codes/labeled_{data_type}_data2_temp.json', 'r', encoding='utf-8') as f:# ./ window_{data_type}_data
        data = json.load(f)
    return data

# 加载训练、验证和测试数据
train_data = load_data('train')
valid_data = load_data('valid')
test_data = load_data('test')
# 计算验证集的一半大小
valid_sample_size = math.ceil(len(valid_data) / 2)
# 从验证集中随机采样一半数据
random.seed(42)  # 固定随机种子，保证结果可重复
valid_data = random.sample(valid_data, valid_sample_size)
random.shuffle(train_data)


# 初始化带有角色嵌入的BART模型
model = BartForConditionalGeneration.from_pretrained('./persona_extractor/facebook/bart-large-cnn')
# 加载tokenizer
tokenizer = BartTokenizer.from_pretrained('./persona_extractor/facebook/bart-large-cnn')
tokenizer.add_special_tokens({'additional_special_tokens': ["<persona>","<sep>"]})
model.resize_token_embeddings(len(tokenizer))
# 创建数据集实例
train_dataset = DialogueInferenceDataset(train_data, tokenizer)
valid_dataset = DialogueInferenceDataset(valid_data, tokenizer)
test_dataset = DialogueInferenceDataset(test_data, tokenizer)

swanlab_callback = SwanLabCallback(
    project="Persona",
    experiment_name="bart-large",
    description="history的格式, 添加了special token:['<persona>','<sep>']. 随机初始化",
    config={
        "model": "/facebook/bart-large-cnn",
        "dataset": "/persona_data/history_data",
    }
)

# 初始化训练参数
training_args = TrainingArguments(
    output_dir='./persona_bart-large',
    num_train_epochs=10,
    per_device_train_batch_size=6,
    per_device_eval_batch_size=8,
    warmup_steps=100,
    weight_decay=0.01,
    logging_steps=100,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    save_total_limit=2,
    learning_rate=1e-5,
)

# 定义Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
    callbacks=[swanlab_callback],
)

# 开始训练
trainer.train()

# 在测试集上评估模型
print("Evaluating model on test set...")

# 使用Trainer评估函数
test_results = trainer.evaluate(test_dataset)

# 打印评估结果
print(test_results)

# 使用 pandas 将测试集的前10条数据转换为 DataFrame 格式，便于操作
test_df = pd.DataFrame(test_data[0:10])  # 只取前10条

test_text_list = []

# 遍历测试集前10条数据，进行推理
for index, row in test_df.iterrows():
    
    # 对话输入
    input_value = row['conversation_history']
    
    # 期望的标签/输出
    label = "<persona> "+" <persona> ".join(row['persona'])
    
    # 通过模型生成推理结果
    response = predict(row, model, tokenizer)
    
    # 将结果和输入合并成文本
    result_text = f"{response}\n\n{input_value}"
    
    test_text_list.append(swanlab.Text(result_text, caption=label))

swanlab.log({"Prediction": test_text_list})
swanlab.finish()