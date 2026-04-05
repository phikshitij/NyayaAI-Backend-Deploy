import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

print("Loading dataset...")
df = pd.read_csv("sections_df.csv")

print("Downloading MiniLM-L6-v2 (approx 90MB)...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("Encoding descriptions...")
descriptions = df["Description"].fillna("").astype(str).tolist()
embeddings = model.encode(descriptions, convert_to_tensor=True)

print("Saving section_embeddings.pt...")
torch.save(embeddings, "section_embeddings.pt")
print("Done! Ready for free-tier deployment.")
