import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ==============================
# 1. Load MovieLens 1M dataset
# ==============================

DATA_PATH = "data/ml-1m/ratings.dat"

ratings = pd.read_csv(
    DATA_PATH,
    sep="::",
    engine="python",
    names=["UserID", "MovieID", "Rating", "Timestamp"]
)

print("Ratings shape:", ratings.shape)

# ==========================================
# 2. Create User × Movie Pivot Matrix
# ==========================================

pivot = ratings.pivot(
    index="UserID",
    columns="MovieID",
    values="Rating"
)

print("Pivot shape (users x movies):", pivot.shape)

# ==================================================
# 3. Mean-center each movie (column-wise average)
#    and fill missing values with 0
# ==================================================

movie_means = pivot.mean(axis=0)

# subtract movie mean from each rating
normalized = pivot.subtract(movie_means, axis=1)

# replace NaN with 0
normalized = normalized.fillna(0)

print("Normalized matrix shape:", normalized.shape)

# Convert to numpy
user_matrix = normalized.values.astype(np.float32)

# ======================================
# 4. Define Fully Connected MLP
# ======================================

class MLP(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(MLP, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.Sigmoid(),
            nn.Linear(512, embedding_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# ======================================
# 5. Generate Embeddings
# ======================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_dim = user_matrix.shape[1]
batch_size = 128

dataset = TensorDataset(torch.from_numpy(user_matrix))
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

embedding_sizes = [64, 128, 256]

os.makedirs("output/embeddings", exist_ok=True)

for emb_dim in embedding_sizes:
    
    print(f"\nGenerating embeddings of size {emb_dim}")
    
    model = MLP(input_dim, emb_dim).to(device)
    model.eval()  # no training, just forward pass
    
    all_embeddings = []

    with torch.no_grad():
        for batch in loader:
            batch_x = batch[0].to(device)
            emb = model(batch_x)
            all_embeddings.append(emb.cpu().numpy())
    
    embeddings = np.vstack(all_embeddings)
    
    # Save to CSV
    df_emb = pd.DataFrame(
        embeddings,
        index=pivot.index  # User IDs
    )
    
    output_path = f"output/embeddings/user_embeddings_{emb_dim}.csv"
    df_emb.to_csv(output_path)
    
    print(f"Saved: {output_path}")

print("\nDone.")