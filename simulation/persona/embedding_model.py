import numpy as np

# Implemented using this: https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    def __init__(self, device) -> None:
        self.model = SentenceTransformer(
            "mixedbread-ai/mxbai-embed-large-v1", device=device
        )

        self.device = device

    def embed(self, text: str) -> np.ndarray:
        vec = self.model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        return vec.squeeze()

    def embed_retrieve(self, text: str) -> np.ndarray:
        return self.embed(
            f"Represent this sentence for searching relevant passages: {text}"
        )
