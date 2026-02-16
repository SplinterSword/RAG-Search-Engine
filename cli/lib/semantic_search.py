from importlib import metadata
from unittest import result
from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
import json
import sys

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from utils.semantic_search_utils.text_preprocessing import text_preprocessing
from utils.semantic_search_utils.vector_operations import cosine_similarity
from utils.semantic_search_utils.semantic_chunk import semantic_chunk

class SemanticSearch:
    modal = None
    embeddings = None
    documents: list[dict] = None
    document_map: dict[int, dict] = {}
    
    def __init__(self, model_name = "all-MiniLM-L6-v2"):
        self.modal = SentenceTransformer(model_name)
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
                print(
                    "Embeddings shape does not match documents length, rebuilding embeddings",
                    file=sys.stderr,
                )
                return self.build_embeddings(documents)
            return self.embeddings
        else:
            return self.build_embeddings(documents)
    
    def search(self, query: str, limit: int) -> list[dict]:
        BASE_DIR = Path(__file__).resolve().parent.parent.parent
        cache_dir = BASE_DIR / 'cache'
        cache_dir.mkdir(exist_ok=True)
        embedding_file = cache_dir / 'movie_embeddings.npy'

        if not embedding_file.exists():
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        
        with open(embedding_file, 'rb') as f:
            self.embeddings = np.load(f)
        
        query_embedding = self.generate_embedding(query)

        similarity_scores = []

        for i, embedding in enumerate(self.embeddings):
            similarity = cosine_similarity(query_embedding, embedding)
            similarity_scores.append((similarity, i+1))
            
        similarity_scores.sort(reverse=True)

        results = []
        
        for i in range(limit):
            score, doc_id = similarity_scores[i]
            title = self.document_map[doc_id]["title"]
            description = self.document_map[doc_id]["description"]
            results.append({"score": score, "title": title, "description": description})
        
        return results

class ChunkedSemanticSearch(SemanticSearch):
    chunk_embeddings: np.ndarray = None
    chunk_metadata: dict = None
    documents: list[dict] = None
    document_map: dict[int, dict] = {}
    
    def __init__(self, model_name = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents: list[dict]):
        BASE_DIR = Path(__file__).resolve().parent.parent.parent
        cache_dir = BASE_DIR / 'cache'
        cache_dir.mkdir(exist_ok=True)
        chunk_embedding_file = cache_dir / 'chunk_embeddings.npy'
        chunk_metadata_file = cache_dir / 'chunk_metadata.json'

        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc
        
        chunks = []
        chunk_meta = {}
        
        for doc in documents:
            if doc["description"] == "":
                continue
            curr_chunks = semantic_chunk(doc["description"], 4, 1)
            chunks.extend(curr_chunks)
            for i, _ in enumerate(curr_chunks):
                movie_idx = doc['id']
                chunk_idx = i
                total_curr_chunks = len(curr_chunks)
                chunk_meta[len(chunks) - total_curr_chunks + i] = {
                    "movie_idx": movie_idx,
                    "chunk_idx": chunk_idx,
                    "total_chunks": total_curr_chunks
                }
        
        self.chunk_embeddings = self.modal.encode(chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_meta

        with open(chunk_embedding_file, 'wb') as f:
            np.save(f, self.chunk_embeddings)
        
        with open(chunk_metadata_file, 'w') as f:
            json.dump({"chunks": chunk_meta, "total_chunks": len(chunks)}, f, indent=2)
        
        return chunks
    

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        BASE_DIR = Path(__file__).resolve().parent.parent.parent
        cache_dir = BASE_DIR / 'cache'
        cache_dir.mkdir(exist_ok=True)
        chunk_embedding_file = cache_dir / 'chunk_embeddings.npy'
        chunk_metadata_file = cache_dir / 'chunk_metadata.json'

        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc

        if cache_dir.exists() and chunk_embedding_file.exists() and chunk_metadata_file.exists():
            with open(chunk_embedding_file, 'rb') as f:
                self.chunk_embeddings = np.load(f)
            
            with open(chunk_metadata_file, 'r') as f:
                metadata = json.load(f)
                self.chunk_metadata = metadata
            
            if self.chunk_embeddings.shape[0] != len(self.chunk_metadata["chunks"]):
                print(
                    "Embeddings shape does not match documents length, rebuilding embeddings",
                    file=sys.stderr,
                )
                return self.build_chunk_embeddings(documents)
            return self.chunk_embeddings
        else:
            return self.build_chunk_embeddings(documents)
    
    def search_chunk(self, query: str, limit: int = 10) -> list[dict]:
        query_embedding = self.generate_embedding(query)

        chuck_scores: list[dict] = []
        chunk_metadata = self.chunk_metadata["chunks"]

        for i, embedding in enumerate(self.chunk_embeddings):
            score = cosine_similarity(query_embedding, embedding)
            curr_chunk_metadata = chunk_metadata[str(i)]
            chunk_idx = curr_chunk_metadata["chunk_idx"]
            movie_idx = curr_chunk_metadata["movie_idx"]
            chuck_scores.append({"score": score, "chunk_idx": chunk_idx, "movie_idx": movie_idx})
        
        movie_score: dict[int, float] = {}
        
        for score in chuck_scores:
            movie_idx = score["movie_idx"]
            if movie_idx not in movie_score:
                movie_score[movie_idx] = 0
            if movie_score[movie_idx] < score["score"]:
                movie_score[movie_idx] = score["score"]
        
        movie_score = sorted(movie_score.items(), key=lambda x: x[1], reverse=True)
        
        result: list[dict] = []

        for i in range(limit):
            movie_idx, score = movie_score[i]
            movie = self.document_map[movie_idx]
            result.append({"id": movie["id"], "title": movie["title"], "document": movie["description"][:100], "score": round(score, 4)})
        
        return result
        

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
