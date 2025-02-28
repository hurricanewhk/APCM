import os
from transformer import BartForConditionalGeneration
from transformers import BartTokenizer 
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

        dialogue_text = " <sep> ".join([turn['text'] for turn in conversation])
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


        

        personality_output = " ".join([f"<usr2_personality> {p}" for p in usr2_personality])
        labels = self.tokenizer(personality_output, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")

        return {
            'input_ids': inputs['input_ids'].squeeze(), 
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': labels['input_ids'].squeeze(),
            'token_type_ids': role_ids,  
        }
    

def predict(conversation, model, tokenizer):

    device=model.device

    dialogue_text = " <sep> ".join([turn['content'] for turn in conversation])


    roles = []
    for turn in conversation:
        speaker = turn['speaker']
        role = 0 if speaker == 'supporter' else 1
        roles.append(role)


    inputs = tokenizer(dialogue_text, max_length=384, padding="max_length", truncation=True, return_tensors="pt").to(device)


    tokenized_input = tokenizer([turn['content'] for turn in conversation], return_attention_mask=False, add_special_tokens=False)
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
    max_length=384,
    num_beams=10,
    early_stopping=True
    )
    response = tokenizer.decode(output[0], skip_special_tokens=False)
    return response

def load_data(data_type):
    with open(f'./codes/window_{data_type}_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

model = BartForConditionalGeneration.from_pretrained('./codes/persona_extraction/checkpoint')
tokenizer=BartTokenizer.from_pretrained('./codes/persona_extraction/checkpoint')
model.eval()

test_data = load_data('test')

test_df = pd.DataFrame(test_data)  
test_text_list = []
batch_size = 4
total_samples = len(test_df)


for start in tqdm(range(0, total_samples, batch_size), desc="Processing Batches"):
    end = min(start + batch_size, total_samples)
    batch = test_df.iloc[start:end] 

    for index, row in batch.iterrows():
        conversation = row['conversation_history']
        label = row['usr2_personality']
        
        persona = predict(row, model, tokenizer)
        cleaned_text = persona.replace('</s>', '').replace('<s>', '').strip()
        persona_list = [t.strip() for t in cleaned_text.split('<usr2_personality>') if t.strip()]
        
        test_text_list.append({'conversation_history': conversation, 'persona': persona_list, 'label': label})

with open('./test_infer.json', 'w') as f:
    json.dump(test_text_list, f, indent=4)

# with open('./window_ESConv.json', 'r', encoding='utf-8') as f:
#     test_data = json.load(f)

# test_data=test_data[:10]
# test_df = pd.DataFrame(test_data)  

# batch_size = 4

# total_samples = len(test_df)

# test_df['persona_list'] = [[] for _ in range(len(test_df))]

# for start in tqdm(range(0, total_samples, batch_size), desc="Processing Batches"):
#     end = min(start + batch_size, total_samples)
#     batch = test_df.iloc[start:end]  


#     for index, row in batch.iterrows():
#         persona_list = []
#         for conversation in row["window_dialog"]:

#             persona = predict(conversation, model, tokenizer)
#             cleaned_text = persona.replace('</s>', '').replace('<s>', '').strip()
#             temp_list = [t.strip() for t in cleaned_text.split('<usr2_personality>') if t.strip()]
#             persona_list.extend(temp_list)

#         test_df.at[index, 'persona_list'] = persona_list

# test_df = test_df.drop(columns=['window_dialog'])

# test_data = test_df.to_dict(orient='records')

# with open('./sample_infer.json', 'w') as f:
#     json.dump(test_data, f, indent=4)


        
















        




        





