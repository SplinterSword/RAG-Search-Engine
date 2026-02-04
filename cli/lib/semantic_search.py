from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
import json
import sys
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))
from utils.semantic_search_utils.text_preprocessing import text_preprocessing

class SemanticSearch:
    modal = None
    embeddings = None
    documents: list[dict] = None
    document_map: dict[int, dict] = {}
    
    def __init__(self):
        self.modal = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text: str):
        if text == "" or text is None or len(text.strip()) == 0:
            raise ValueError("Text cannot be empty or None or only contains whitespace")

        text = text_preprocessing(text)
        
        embeddings = self.modal.encode([text])
        return embeddings[0]

    def build_embeddings(self, documents: list[dict]):
        BASE_DIR = Path(__file__).resolve().parent.parent.parent
        cache_dir = BASE_DIR / 'cache'
        cache_dir.mkdir(exist_ok=True)

        self.documents = documents
        raw_document_data = []
        for doc in documents:
            self.document_map[doc["id"]] = doc
            raw_document_data.append(f"{doc['title']}: {doc['description']}")
        
        self.embeddings = self.modal.encode(raw_document_data, show_progress_bar=True)
        
        # Save embeddings to cache
        with open(cache_dir / 'movie_embeddings.npy', 'wb') as f:
            np.save(f, self.embeddings)
        
        return self.embeddings

    def load_or_create_embeddings(self, documents: list[dict]):
        BASE_DIR = Path(__file__).resolve().parent.parent.parent
        cache_dir = BASE_DIR / 'cache'
        cache_dir.mkdir(exist_ok=True)

        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc
        
        if cache_dir.exists() and (cache_dir / 'movie_embeddings.npy').exists():
            with open(cache_dir / 'movie_embeddings.npy', 'rb') as f:
                self.embeddings = np.load(f)

            if self.embeddings.shape[0] != len(documents):
                print("Embeddings shape does not match documents length, rebuilding embeddings")
                return self.build_embeddings(documents)
            return self.embeddings
        else:
            return self.build_embeddings(documents)


def verify_modal():
    try:
        sematic_search = SemanticSearch()
        print(f"Model loaded: {sematic_search.modal}")
        print(f"Max sequence length: {sematic_search.modal.max_seq_length}")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def embed_query_text(query: str):
    sematic_search = SemanticSearch()
    embedding = sematic_search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")
    return embedding

def verify_embeddings():
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    data_dir = BASE_DIR / 'data'
    data_file = data_dir / 'movies.json'

    with open(data_file, 'r') as f:
        data = json.load(f)

    documents = data['movies']
    
    sematic_search = SemanticSearch()
    embeddings = sematic_search.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")
    return True if len(documents) == embeddings.shape[0] else False
