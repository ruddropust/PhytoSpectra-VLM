from sentence_transformers import SentenceTransformer



class EmbeddingModel:


    def __init__(self):

        self.model = SentenceTransformer(
            "BAAI/bge-small-en-v1.5", local_files_only=True
        )


    def encode(self,texts):

        return self.model.encode(
            texts,
            normalize_embeddings=True
        )
