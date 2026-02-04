
from typing import Counter
from utils.text_preprocessing import text_preprocessing
from pathlib import Path
import pickle
import math
from utils.search_utils import BM25_K1, BM25_B

class InvertedIndex:
    """
    Inverted index class for storing token-document mappings.
    """
    index: dict[str, list[int]] = {}
    docmap: dict[int, str] = {}
    term_frequency: dict[int, Counter] = {}
    doc_length: dict[int, int] = {}

    def __init__(self):
        self.doc_length = {}
    
    def __add_document(self, doc_id: int, text: str):
        tokens = text_preprocessing(text)
        self.doc_length[doc_id] = len(tokens)
        for token in tokens:
            if token not in self.index:
                self.index[token] = []
            self.index[token].append(doc_id)
            
            if doc_id not in self.term_frequency:
                self.term_frequency[doc_id] = Counter()
            self.term_frequency[doc_id][token] += 1

    def __get_avg_doc_length(self) -> float:
        if len(self.doc_length) == 0:
            return 0.0
        return sum(self.doc_length.values()) / len(self.doc_length)
    
    def get_documents(self, term: str) -> list[int]:
        term = term.lower()
        doc_ids = self.index.get(term, [])
        return sorted(set(doc_ids))
    
    def get_tf(self, doc_id: int, term: str) -> int:
        tokens = text_preprocessing(term)
        if len(tokens) > 1:
            raise ValueError("Term must be a single token")

        token = tokens[0]
        return self.term_frequency.get(doc_id, Counter()).get(token, 0)
    
    def get_bm25_tf(self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
        tf = self.get_tf(doc_id, term)
        doc_length = self.doc_length.get(doc_id, 0)
        avg_doc_length = self.__get_avg_doc_length()
        length_norm = 1 - b + b * (doc_length / avg_doc_length)

        saturated_tf = (tf * (k1 + 1)) / (tf + k1 * length_norm)
        return saturated_tf
    
    def get_idf(self, term: str) -> float:
        term_tokens = text_preprocessing(term)
        if len(term_tokens) != 1:
            raise ValueError("Term must be a single token")
        
        token = term_tokens[0]
        total_doc_count = len(self.docmap)
        postings = self.index.get(token, [])
        term_match_doc_count = len(set(postings))
        
        idf = math.log((total_doc_count + 1) / (term_match_doc_count + 1))
        return idf
    
    def get_bm25_idf(self, term: str) -> float:
        term_tokens = text_preprocessing(term)
        if len(term_tokens) != 1:
            raise ValueError("Term must be a single token")
        
        token = term_tokens[0]
        total_doc_count = len(self.docmap)
        postings = self.index.get(token, [])
        term_match_doc_count = len(set(postings))
        
        idf = math.log((total_doc_count - term_match_doc_count + 0.5) / (term_match_doc_count + 0.5) + 1)
        return idf
    
    def bm25(self, doc_id: int, term: str) -> float:
        bm25_tf = self.get_bm25_tf(doc_id, term)
        bm25_idf = self.get_bm25_idf(term)
        return bm25_tf * bm25_idf

    def bm25_search(self, query: str, limit: int = 5) -> list[tuple[int, float]]:
        query_tokens = text_preprocessing(query)
        scores = {}
        
        for doc_id in self.docmap.keys():
            scores[doc_id] = 0
            for token in query_tokens:
                scores[doc_id] += self.bm25(doc_id, token)
        
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores[:limit]
    
    def build(self, movies: list[dict]):
        for movie in movies:
            self.__add_document(movie['id'], f"{movie['title']} {movie['description']}")
            self.docmap[movie['id']] = movie
    
    def save(self):
        BASE_DIR = Path(__file__).resolve().parent.parent.parent
        cache_dir = BASE_DIR / 'cache'
        cache_dir.mkdir(exist_ok=True)
        
        with open(cache_dir / 'index.pkl', 'wb') as f:
            pickle.dump(self.index, f)
        with open(cache_dir / 'docmap.pkl', 'wb') as f:
            pickle.dump(self.docmap, f)
        with open(cache_dir / 'term_frequency.pkl', 'wb') as f:
            pickle.dump(self.term_frequency, f)
        with open(cache_dir / 'doc_length.pkl', 'wb') as f:
            pickle.dump(self.doc_length, f)

    def load(self):
        BASE_DIR = Path(__file__).resolve().parent.parent.parent
        cache_dir = BASE_DIR / 'cache'
        
        if not cache_dir.exists():
            raise FileNotFoundError("Cache directory not found. Run 'build' command first.")
        
        if not (cache_dir / 'index.pkl').exists() or not (cache_dir / 'docmap.pkl').exists() or not (cache_dir / 'term_frequency.pkl').exists():
            raise FileNotFoundError("Cache files not found. Run 'build' command first.")

        print("Loading index...")
        with open(cache_dir / 'index.pkl', 'rb') as f:
            self.index = pickle.load(f)
        with open(cache_dir / 'docmap.pkl', 'rb') as f:
            self.docmap = pickle.load(f)
        with open(cache_dir / 'term_frequency.pkl', 'rb') as f:
            self.term_frequency = pickle.load(f)
        with open(cache_dir / 'doc_length.pkl', 'rb') as f:
            self.doc_length = pickle.load(f)
        print("Index loaded.")

