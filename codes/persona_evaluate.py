import json
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity


tokenizer = AutoTokenizer.from_pretrained("./SimCSE")
model = AutoModel.from_pretrained("./SimCSE")

def get_embeddings(texts):
    if not texts:
        texts=" "
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
    return embeddings.cpu().numpy().tolist() 


def preprocess_data(data, output_dir):
    embeddings_data = []

    for entry in tqdm(data, desc="Processing data"):
        persona_texts = entry['persona']
        label_texts = entry['label']
        persona_embeddings = get_embeddings(persona_texts)
        label_embeddings = get_embeddings(label_texts)
    
        embeddings_data.append({
            "persona": persona_texts,
            "persona_embeddings": persona_embeddings,
            "label": label_texts,
            "label_embeddings": label_embeddings
        })

    with open(output_dir, "w") as f:
        json.dump(embeddings_data, f, ensure_ascii=False, indent=2)

# 假设输入数据data
with open('./test_infer.json', 'r') as f:
    data = json.load(f)

output_dir="test_embeddings_data.json"
# 计算并存储embedding
preprocess_data(data,output_dir)
print(f"{output_dir} saved.")


def evaluate_from_embeddings(file_path):
    with open(file_path, "r") as f:
        embeddings_data = json.load(f)

    TP = 0  # True Positives
    FP = 0  # False Positives
    FN = 0  # False Negatives

    for entry in embeddings_data:
        persona_embeddings = entry['persona_embeddings']
        label_embeddings = entry['label_embeddings']

        matched_labels = set()  

        for pe in persona_embeddings:
            match_found = False
            for le_idx, le in enumerate(label_embeddings):
                similarity = cosine_similarity([pe], [le])[0][0] 
                if similarity > 0.5:  
                    matched_labels.add(le_idx) 
                    match_found = True
                    break  

            if not match_found:
                FP += 1  

     
        FN += len(label_embeddings) - len(matched_labels)

        TP += len(matched_labels)

    return {
        "true_positives": TP,
        "false_positives": FP,
        "false_negatives": FN
    }

# 评估模型
output_dir="test_embeddings_data.json"
results = evaluate_from_embeddings(output_dir)
print(f"True Positives: {results['true_positives']}, False Positives: {results['false_positives']}, False Negatives: {results['false_negatives']}")
