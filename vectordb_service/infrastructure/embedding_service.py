import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer


class EmbeddingService:
    """Wraps the CLIP model for image and text embedding generation."""

    def __init__(self, model_id: str) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_id).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id)

    # -- public ---------------------------------------------------------

    @staticmethod
    def _to_numpy(output) -> np.ndarray:
        """Handle both raw tensors and BaseModelOutputWithPooling, then L2-normalise."""
        if hasattr(output, "pooler_output"):
            output = output.pooler_output
        arr = output.cpu().detach().numpy()
        norm = np.linalg.norm(arr, axis=-1, keepdims=True)
        return arr / np.where(norm == 0, 1, norm)

    def get_image_embedding(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path)
        pixel_values = self.processor(
            text=None, images=image, return_tensors="pt"
        )["pixel_values"].to(self.device)
        with torch.no_grad():
            embedding = self.model.get_image_features(pixel_values)
        return self._to_numpy(embedding)

    def get_text_embedding(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True
        ).to(self.device)
        with torch.no_grad():
            embedding = self.model.get_text_features(**inputs)
        return self._to_numpy(embedding)

    @property
    def embedding_dim(self) -> int:
        """Single-modality embedding dimension (image or text)."""
        return self.model.config.projection_dim
