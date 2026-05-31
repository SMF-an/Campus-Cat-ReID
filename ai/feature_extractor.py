from io import BytesIO
import numpy as np
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor


class CLIPFeatureExtractor:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None
        self.dim = 512
        self._load()

    def _load(self):
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        # infer dim from model
        self.dim = self.model.visual_projection.out_features

    def extract(self, image: Image.Image) -> np.ndarray:
        """Return a L2-normalized feature vector as numpy array."""
        if self.model is None or self.processor is None:
            vec = np.random.rand(self.dim).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            return vec

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
            # transformers versions may return either a tensor or a model output object.
            if torch.is_tensor(outputs):
                tensor = outputs
            elif hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
                tensor = outputs.image_embeds
            elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                tensor = outputs.pooler_output
            elif hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
                tensor = outputs.last_hidden_state[:, 0, :]
            else:
                raise TypeError(f"Unexpected CLIP output type: {type(outputs)!r}")

            feats = tensor.detach().cpu().numpy()[0]
            feats = feats.astype(np.float32)
            feats = feats / np.linalg.norm(feats)
            return feats


def load_image_bytes(data: bytes):
    return Image.open(BytesIO(data)).convert("RGB")
