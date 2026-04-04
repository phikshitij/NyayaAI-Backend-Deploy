from fastapi import FastAPI
from pydantic import BaseModel
import torch
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi.middleware.cors import CORSMiddleware

import os

app = FastAPI(title="NyayaAI ML Service")

# Best practice for production deployment: 
# Instead of ["*"], explicitly whitelist your frontend domains
ALLOWED_ORIGINS = [
    "http://localhost:5173",  # Local development frontend
    "http://localhost:4173",  # Local production preview
    os.getenv("FRONTEND_URL", ""), # Dynamic production URL via environment variable
    # Add your actual production domain here manually if preferred, e.g.,
    # "https://my-nyaya-project.vercel.app"
]

# Add this BEFORE your routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin for origin in ALLOWED_ORIGINS if origin],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # Restrict to required methods only
    allow_headers=["*"],
)
# Load everything ONCE at startup
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
section_embeddings = torch.load("section_embeddings.pt")
df = pd.read_csv("sections_df.csv")

class ComplaintRequest(BaseModel):
    complaint: str
    top_k: int = 3

def confidence(score):
    if score >= 0.55:
        return "High"
    elif score >= 0.40:
        return "Medium"
    return "Low"

@app.post("/predict")
def predict_sections(req: ComplaintRequest):
    emb = model.encode(req.complaint, convert_to_tensor=True)

    sims = cosine_similarity(
        emb.cpu().numpy().reshape(1, -1),
        section_embeddings.cpu().numpy()
    )[0]

    top_idx = np.argsort(sims)[-req.top_k:][::-1]

    results = []
    for i in top_idx:
        score = float(sims[i])
        results.append({
            "section": str(df.iloc[i]["Section"]),
            "section_name": df.iloc[i]["Section _name"],
            "similarity": round(score, 4),
            "confidence": confidence(score),
            "description": df.iloc[i]["Description"]
        })

    return results
