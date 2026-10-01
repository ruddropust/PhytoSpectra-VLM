"""Small in-memory cosine-similarity store for the curated knowledge base.

The project currently contains only a handful of documents, so a NumPy store is
both sufficient and more reproducible than an optional FAISS dependency.
"""

from typing import List

import numpy as np


class VectorStore:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.embeddings = np.empty((0, dimension), dtype=np.float32)
        self.documents: List[str] = []

    def add(self, embeddings, documents) -> None:
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != self.dimension:
            raise ValueError("Knowledge embedding dimension mismatch")
        if matrix.shape[0] != len(documents):
            raise ValueError("Knowledge embeddings and documents must have equal length")
        self.embeddings = np.concatenate([self.embeddings, matrix], axis=0)
        self.documents.extend(documents)

    def search(self, query_embedding, k: int = 3):
        if not self.documents:
            return []
        query = np.asarray(query_embedding, dtype=np.float32)
        if query.ndim != 2 or query.shape[1] != self.dimension:
            raise ValueError("Query embedding dimension mismatch")

        scores = query[0] @ self.embeddings.T
        top_indices = np.argsort(scores)[::-1][: min(k, len(self.documents))]
        return [
            {
                "text": self.documents[int(index)],
                "score": float(scores[int(index)]),
            }
            for index in top_indices
        ]
