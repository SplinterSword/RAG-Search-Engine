from PIL import Image
from sentence_transformers import SentenceTransformer

class MultiModalSearch:
    model = None

    def __init__(self, model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)
    
    def embed_image(self, image_path: str):
        with Image.open(image_path) as image:
            image_embedding = self.model.encode([image])
            return image_embedding


def verify_image_embedding(image_path: str):
    multimodal_search = MultiModalSearch()
    image_embedding = multimodal_search.embed_image(image_path)
    return image_embedding
        