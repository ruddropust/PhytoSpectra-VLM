from pathlib import Path
from typing import Optional

from app.retrieval.embedding import EmbeddingModel
from app.retrieval.vector_store import VectorStore


class KnowledgeRetriever:
    def __init__(self, knowledge_dir: Optional[Path] = None):
        self.encoder = EmbeddingModel()
        self.store = None
        if knowledge_dir is None:
            self.knowledge_dir = Path(__file__).resolve().parents[1] / "knowledge"
        else:
            self.knowledge_dir = Path(knowledge_dir)
        self.load()

    def load(self):
        docs = []
        if self.knowledge_dir.exists():
            for file in sorted(self.knowledge_dir.glob("*.md")):
                text = file.read_text(encoding="utf-8").strip()
                if text:
                    docs.append(text)

        if docs:
            embeddings = self.encoder.encode(docs)
            self.store = VectorStore(embeddings.shape[1])
            self.store.add(embeddings, docs)

    def retrieve(self, question: str, top_k: int = 3, min_score: float = 0.35) -> str:
        if self.store is None:
            return ""
        q = self.encoder.encode([question])
        results = self.store.search(q, top_k)
        if not results:
            return ""
        return "\n\n".join(
            res["text"]
            for res in results
            if isinstance(res, dict)
            and "text" in res
            and float(res.get("score", 0.0)) >= min_score
        )
