from PIL import Image
from sentence_transformers import SentenceTransformer
from typing import List
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from utils.cli_utils.file_loading import load_movies
from utils.semantic_search_utils.vector_operations import cosine_similarity

class MultiModalSearch:
    model = None
    documents: List[dict] = None
    texts: List[str] = None
    text_embeddings: List[List[float]] = []
    doc_map: dict[int, dict] = None

    def __init__(self, model_name="clip-ViT-B-32", documents: List[dict] = None):
        self.model = SentenceTransformer(model_name)
        self.documents = documents
        if documents:
            self.texts = [f"{doc['title']}: {doc['description']}" for doc in documents]
            self.doc_map = {i: doc for i, doc in enumerate(documents)}
            self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)
    
    def embed_image(self, image_path: str):
        with Image.open(image_path) as image:
            image_embedding = self.model.encode([image])
            return image_embedding

    def search_with_image(self, image_path: str):
        image_embedding = self.embed_image(image_path)
        similarity_scores = []
        for i, text_embedding in enumerate(self.text_embeddings):
            similarity = cosine_similarity(image_embedding, text_embedding)
            similarity_scores.append((similarity, i))

        similarity_scores.sort(reverse=True)
        sorted_results = [
            {
                "id": self.doc_map[i]["id"],
                "title": self.doc_map[i]["title"],
                "description": self.doc_map[i]["description"],
                "score": score
            }
            for score, i in similarity_scores[:5]
        ]
        
        return sorted_results



def verify_image_embedding(image_path: str):
    multimodal_search = MultiModalSearch()
    image_embedding = multimodal_search.embed_image(image_path)
    return image_embedding

def image_search_command(image_path: str):
    movies = load_movies()
    multimodal_search = MultiModalSearch(documents=movies)
    results = multimodal_search.search_with_image(image_path)
    return results