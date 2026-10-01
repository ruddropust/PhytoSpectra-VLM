"""
Semantic target resolver
"""


from sentence_transformers import SentenceTransformer
import numpy as np



class SemanticEncoder:


    def __init__(self):

        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )


        self.targets = [
            "GrainWeight",
            "Gsw",
            "PhiPS2",
            "Fertilizer",
            "Cultivar"
        ]


        self.embeddings = (
            self.model.encode(
                self.targets,
                normalize_embeddings=True
            )
        )



    def find_target(self, question):


        q_emb = self.model.encode(
            question,
            normalize_embeddings=True
        )


        scores = np.dot(
            self.embeddings,
            q_emb
        )


        idx = np.argmax(scores)


        return {

            "target":
            self.targets[idx],

            "confidence":
            float(scores[idx])

        }