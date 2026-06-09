import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import os

def main():
    # 1. Pipeline Paths
    input_path = 'data/processed/text_profiles.parquet'
    output_path = 'data/processed/llm_features.npy'
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Missing {input_path}. Run Block 2 first.")

    # 2. Load Core Data
    print("Loading data profiles for LLaMA context mapping...")
    df_text = pd.read_parquet(input_path)
    texts = df_text['car_text_profile'].tolist()

    # 3. Target CPU Engine Configuration
    device = torch.device('cpu')
    print("Executing extraction pipeline strictly on CPU engine configuration...")

    # 4. Initialize Lightweight LLM (Algorithm-3)
    model_name = "unsloth/Llama-3.2-1B-Instruct"
    print(f"Loading {model_name} from local cache...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Enforce standard padding token support for causal models
    tokenizer.pad_token = tokenizer.eos_token 
    
    # Load model with local-only and low-memory allocation flags for CPU stability
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        local_files_only=True
    ).to(device)
    model.eval()

    llm_embeddings = []
    batch_size = 64 # Maximized grouping to fully saturate CPU cores across smaller vectors
    
    # 5. Hyper-Optimized Batched Vector Extraction Core
    print(f"\nProcessing LLM Feature Spaces in Batches of {batch_size}...")
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="LLaMA Fast Pass"):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize sequence block (Capped at 20 tokens to drop sequence complexity)
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=20 
            ).to(device)
            
            # Pass inputs through the model, capturing hidden states directly
            outputs = model(**inputs, output_hidden_states=True)
            
            # Extract final layer hidden states (Shape: [Batch, SeqLen, 2048])
            final_hidden_states = outputs.hidden_states[-1]
            
            # Apply text pooling mask to calculate the geometric vector averages
            attention_mask = inputs['attention_mask'].unsqueeze(-1)
            input_mask_expanded = attention_mask.expand(final_hidden_states.size()).float()
            
            sum_embeddings = torch.sum(final_hidden_states * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            
            batch_vectors = (sum_embeddings / sum_mask).cpu().numpy()
            llm_embeddings.append(batch_vectors)

    # 6. Save out binary matrix block
    llm_matrix = np.vstack(llm_embeddings)
    np.save(output_path, llm_matrix)
    print(f"\nSuccess! LLaMA parametric matrix generated: {llm_matrix.shape} -> {output_path}")

if __name__ == "__main__":
    main()