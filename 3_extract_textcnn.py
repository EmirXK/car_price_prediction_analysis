import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import re
import os
from tqdm import tqdm

# --- 1. Custom Vocabulary Tokenizer from Scratch ---
class SimpleTokenizer:
    def __init__(self, max_vocab=5000):
        self.max_vocab = max_vocab
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        
    def clean_text(self, text):
        text = str(text).lower()
        text = re.sub(r'[^a-z0-9\s\.]', '', text)
        return text.split()

    def fit(self, texts):
        all_words = []
        for t in texts:
            all_words.extend(self.clean_text(t))
        
        counts = Counter(all_words)
        most_common = counts.most_common(self.max_vocab - 2)
        
        for word, _ in most_common:
            if word not in self.word2idx:
                self.word2idx[word] = len(self.word2idx)
                
    def transform(self, texts, max_len=64):
        sequences = np.zeros((len(texts), max_len), dtype=np.int64)
        for i, t in enumerate(texts):
            words = self.clean_text(t)[:max_len]
            for j, word in enumerate(words):
                sequences[i, j] = self.word2idx.get(word, 1) # Fallback to <UNK>
        return sequences

# --- 2. PyTorch Dataset Layout ---
class CarTextDataset(Dataset):
    def __init__(self, sequences, prices):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.prices = torch.tensor(prices, dtype=torch.float32).unsqueeze(1)
        
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        return self.sequences[idx], self.prices[idx]

# --- 3. TextCNN Architecture with Feature Bottleneck ---
class TextCNNRegressor(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, num_filters=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Convolutional filters scanning 2, 3, and 4 words at a time
        self.conv1 = nn.Conv1d(embed_dim, num_filters, kernel_size=2)
        self.conv2 = nn.Conv1d(embed_dim, num_filters, kernel_size=3)
        self.conv3 = nn.Conv1d(embed_dim, num_filters, kernel_size=4)
        
        # Latent Feature Bottleneck Layer (Algorithm-4 Extractor)
        self.bottleneck = nn.Linear(num_filters * 3, 64)
        self.regressor = nn.Linear(64, 1)

    def forward(self, x, extract_features=False):
        # Transpose to [Batch, EmbedDim, SeqLen] for Conv1d
        x = self.embedding(x).permute(0, 2, 1)
        
        # Apply Convolutions and Max-Over-Time Pooling
        x1 = F.relu(self.conv1(x)).max(dim=2)[0]
        x2 = F.relu(self.conv2(x)).max(dim=2)[0]
        x3 = F.relu(self.conv3(x)).max(dim=2)[0]
        
        combined = torch.cat((x1, x2, x3), dim=1)
        latent_features = F.relu(self.bottleneck(combined))
        
        if extract_features:
            return latent_features
            
        return self.regressor(latent_features)

# --- 4. Main Execution Core ---
def main():
    input_path = 'data/processed/text_profiles.parquet'
    output_path = 'data/processed/textcnn_features.npy'
    
    print("Loading text and continuous price mappings...")
    df = pd.read_parquet(input_path)
    
    # Scale targets globally just to make optimization stable for PyTorch
    target_mean = df['price'].mean()
    target_std = df['price'].std()
    scaled_prices = (df['price'].values - target_mean) / target_std

    # Prepare sequences
    print("Building Custom Automotive Vocabulary...")
    tokenizer = SimpleTokenizer(max_vocab=3000)
    tokenizer.fit(df['car_text_profile'])
    sequences = tokenizer.transform(df['car_text_profile'], max_len=64)
    
    dataset = CarTextDataset(sequences, scaled_prices)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing TextCNN training loop on: {device.type.upper()}")
    
    model = TextCNNRegressor(vocab_size=len(tokenizer.word2idx)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.MSELoss()
    
    # Train for 5 rapid epochs
    model.train()
    print("Fitting TextCNN on domain sequence features...")
    for epoch in range(5):
        total_loss = 0
        for seqs, targets in dataloader:
            seqs, targets = seqs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(seqs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"   Epoch {epoch+1}/5 | Batch Avg MSE Loss: {total_loss/len(dataloader):.4f}")
        
    # --- Feature Extraction Pipeline ---
    print("\nRunning extraction over bottleneck feature dimensions...")
    model.eval()
    all_features = []
    
    # Run structured dataset sequentially to match tracking IDs
    eval_dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
    with torch.no_grad():
        for seqs, _ in tqdm(eval_dataloader, desc="Extracting TextCNN Layers"):
            seqs = seqs.to(device)
            features = model(seqs, extract_features=True)
            all_features.append(features.cpu().numpy())
            
    textcnn_matrix = np.vstack(all_features)
    np.save(output_path, textcnn_matrix)
    print(f"Success! Cached TextCNN matrix shape: {textcnn_matrix.shape} saved to {output_path}")

if __name__ == "__main__":
    main()