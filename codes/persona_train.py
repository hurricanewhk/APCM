from transformers import BartTokenizer, Trainer, TrainingArguments
from transformer import BartForConditionalGeneration

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


        dialogue_text = " <sep> ".join([turn['text'] for turn in conversation]) + tokenizer.eos_token
        

        roles = []
        for turn in conversation:
            speaker = turn['speaker']
            role = 0 if speaker == 'usr1' else 1
            roles.append(role)


        inputs = self.tokenizer(dialogue_text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")


        tokenized_input = self.tokenizer([turn['text'] for turn in conversation], return_attention_mask=False, add_special_tokens=False)
        input_ids = tokenized_input['input_ids']

        expanded_roles = []
        role_idx = 0
        for i, token_list in enumerate(input_ids):
            expanded_roles.extend([roles[role_idx]] * len(token_list))

            if i < len(input_ids) - 1:
                expanded_roles.append(roles[role_idx])  
            role_idx += 1


        input_length = inputs['input_ids'].size(1) 
        if len(expanded_roles) > input_length:
            expanded_roles = expanded_roles[:input_length]
        else:
            expanded_roles.extend([0] * (input_length - len(expanded_roles)))

        role_ids = torch.tensor(expanded_roles).long()
        personality_output = " ".join([f"<persona> {p}" for p in usr2_personality]) + tokenizer.eos_token
    
        labels = self.tokenizer(personality_output, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': labels['input_ids'].squeeze(),
            'token_type_ids': role_ids,
        }

class BartWithRoleEmbedding(BartForConditionalGeneration):
    def __init__(self, config, role_embedding_dim=50):
        super(BartWithRoleEmbedding, self).__init__(config)

        self.role_embedding = nn.Embedding(2, role_embedding_dim)

        self.word_embedding_dim = self.model.encoder.embed_tokens.embedding_dim

        self.fc = nn.Linear(self.word_embedding_dim + role_embedding_dim, self.word_embedding_dim)

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, labels=None, roles=None, role_attention_mask=None):

        word_embeddings = self.model.encoder.embed_tokens(input_ids)


        role_embeddings = self.role_embedding(roles)


        combined_embeddings = torch.cat((word_embeddings, role_embeddings), dim=-1)


        combined_embeddings = self.fc(combined_embeddings)


        encoder_outputs = self.model.encoder(

            inputs_embeds=word_embeddings,
            attention_mask=attention_mask
        )


        outputs = super().forward(
            encoder_outputs=encoder_outputs.last_hidden_state,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels
        )

        return outputs



def predict(row, model, tokenizer):

    device=model.device
    conversation = row['conversation_history']


    dialogue_text = " <sep> ".join([turn['text'] for turn in conversation]) + tokenizer.eos_token


    roles = []
    for turn in conversation:
        speaker = turn['speaker']
        role = 0 if speaker == 'usr1' else 1
        roles.append(role)


    inputs = tokenizer(dialogue_text, max_length=768, padding="max_length", truncation=True, return_tensors="pt").to(device)


    tokenized_input = tokenizer([turn['text'] for turn in conversation], return_attention_mask=False, add_special_tokens=False)
    input_ids = tokenized_input['input_ids']

    expanded_roles = []
    role_idx = 0
    for i, token_list in enumerate(input_ids):
        expanded_roles.extend([roles[role_idx]] * len(token_list))

        if i < len(input_ids) - 1:
            expanded_roles.append(roles[role_idx])
        role_idx += 1
    input_length = inputs['input_ids'].size(1)
    if len(expanded_roles) > input_length:
        expanded_roles = expanded_roles[:input_length]
    else:
        expanded_roles.extend([0] * (input_length - len(expanded_roles)))


    role_ids = torch.tensor(expanded_roles).long().unsqueeze(0).to(device)
    result= {
    'input_ids': inputs['input_ids'],
    'attention_mask': inputs['attention_mask'],
    'token_type_ids': role_ids,
    }
    output = model.generate(
    input_ids=result['input_ids'],
    attention_mask=result['attention_mask'],
    token_type_ids=result['token_type_ids'],
    max_length=128,
    num_beams=10,
    early_stopping=True
    )
    response = tokenizer.decode(output[0], skip_special_tokens=False)
    return response

    
def load_data(data_type):
    with open(f'./codes/labeled_{data_type}_data2_temp.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


train_data = load_data('train')
valid_data = load_data('valid')
test_data = load_data('test')

valid_sample_size = math.ceil(len(valid_data) / 2)

random.seed(42)
valid_data = random.sample(valid_data, valid_sample_size)
random.shuffle(train_data)



model = BartForConditionalGeneration.from_pretrained('./persona_extractor/facebook/bart-large-cnn')

tokenizer = BartTokenizer.from_pretrained('./persona_extractor/facebook/bart-large-cnn')
tokenizer.add_special_tokens({'additional_special_tokens': ["<persona>","<sep>"]})
model.resize_token_embeddings(len(tokenizer))

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


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
    callbacks=[swanlab_callback],
)


trainer.train()


print("Evaluating model on test set...")


test_results = trainer.evaluate(test_dataset)


print(test_results)


test_df = pd.DataFrame(test_data[0:10])

test_text_list = []


for index, row in test_df.iterrows():
    

    input_value = row['conversation_history']
    

    label = "<persona> "+" <persona> ".join(row['persona'])
    

    response = predict(row, model, tokenizer)
    

    result_text = f"{response}\n\n{input_value}"
    
    test_text_list.append(swanlab.Text(result_text, caption=label))

swanlab.log({"Prediction": test_text_list})
swanlab.finish()