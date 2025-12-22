# scripts/utils.py
import numpy as np

def load_embeddings_from_images(images):
    X, y = [], []

    for img in images:
        if img.embedding_binary is None:
            continue

        emb = np.frombuffer(img.embedding_binary, dtype=np.float32)
        X.append(emb)
        y.append(img.dog_id)   # ใช้ dog_id เป็น label

    return np.array(X), np.array(y)
