from cli.lib.keyword_search import InvertedIndex
from utils.keyword_seach_utils.search_utils import BM25_K1, BM25_B

def get_bm25_idf(term: str) -> float:
    index = InvertedIndex()
    index.load()
    
    bm25idf = index.get_bm25_idf(term)
    return bm25idf

def bm25_tf_command(doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
    index = InvertedIndex()
    index.load()
    
    bm25tf = index.get_bm25_tf(doc_id, term, k1, b)
    return bm25tf