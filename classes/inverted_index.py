
from typing import Counter
from utils.text_preprocessing import text_preprocessing
from pathlib import Path
import pickle

class InvertedIndex:
    """
    Inverted index class for storing token-document mappings.
    """
    index: dict[str, list[int]] = {}
    docmap: dict[int, str] = {}
    term_frequency: dict[int, Counter] = {}
    
    def __add_document(self, doc_id: int, text: str):
        tokens = text_preprocessing(text)
        for token in tokens:
            if token not in self.index:
                self.index[token] = []
            self.index[token].append(doc_id)
            
            if doc_id not in self.term_frequency:
                self.term_frequency[doc_id] = Counter()
            self.term_frequency[doc_id][token] += 1
    
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
    
    def build(self, movies: list[dict]):
        for movie in movies:
            self.__add_document(movie['id'], f"{movie['title']} {movie['description']}")
            self.docmap[movie['id']] = movie
    
    def save(self):
        BASE_DIR = Path(__file__).resolve().parent.parent
        cache_dir = BASE_DIR / 'cache'
        cache_dir.mkdir(exist_ok=True)
        
        with open(cache_dir / 'index.pkl', 'wb') as f:
            pickle.dump(self.index, f)
        with open(cache_dir / 'docmap.pkl', 'wb') as f:
            pickle.dump(self.docmap, f)
        with open(cache_dir / 'term_frequency.pkl', 'wb') as f:
            pickle.dump(self.term_frequency, f)

    def load(self):
        BASE_DIR = Path(__file__).resolve().parent.parent
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
        print("Index loaded.")

