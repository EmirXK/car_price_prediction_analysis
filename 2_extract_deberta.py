import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import os

def main():
    # 1. Setup paths
    input_path = 'data/processed/text_profiles.parquet'
    output_path = 'data/processed/deberta_embeddings.npy'
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Could not find {input_path}. Make sure 1_data_preprocessing.py ran successfully.")

    # 2. Load the text data
    print("Loading text profiles...")
    df_text = pd.read_parquet(input_path)
    texts = df_text['car_text_profile'].tolist()
    
    # 3. Detect hardware acceleration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device execution backend: {device.type.upper()}")

    # 4. Initialize Lightweight Transformer
    print("Initializing distilroberta-base model configurations...")
    model_name = "distilroberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    embeddings = []
    batch_size = 64  # Process 64 cars simultaneously to exploit CPU vectorization
    
    # 5. Fast Batched Extraction Loop
    print(f"Beginning batched tensor generation loop (Batch Size: {batch_size})...")
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing Batches"):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize the entire batch at once, capping token lengths to save RAM
            inputs = tokenizer(
                batch_texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=128
            ).to(device)
            
            outputs = model(**inputs)
            
            # Smart Mean Pooling: Accounts for padding masks correctly across the batch
            attention_mask = inputs['attention_mask'].unsqueeze(-1)
            token_embeddings = outputs.last_hidden_state
            
            input_mask_expanded = attention_mask.expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            
            batch_mean_pool = (sum_embeddings / sum_mask).cpu().numpy()
            embeddings.append(batch_mean_pool)

    # 6. Restructure array list back into a single matrix block and cache it
    embeddings_matrix = np.vstack(embeddings)
    np.save(output_path, embeddings_matrix)
    print(f"\nSuccess! Cached matrix shape: {embeddings_matrix.shape} saved to {output_path}")

if __name__ == "__main__":
    main()