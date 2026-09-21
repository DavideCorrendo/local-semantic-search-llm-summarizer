import numpy as np
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_chunks(chunks):
    texts = [c["text"] for c in chunks]
    # normalize_embeddings=True makes euclidean distance on these vectors
    # equivalent (in ranking order) to cosine similarity, since for unit
    # vectors ||a-b||^2 = 2 - 2*(a.b). This matters because hnsw.py only
    # implements euclidean_distance.
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    
    for chunk, vector in zip(chunks, embeddings):
        chunk["embedding"] = vector
    
    return chunks

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
