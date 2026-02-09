import os
from pathlib import Path
from pydoc import doc
import sys

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from utils.hybrid_search_utils.normalize_score import normalize_score


class HybridSearch:
    documents: list[dict] = None
    document_map: list[int, dict] = None
    semantic_search: ChunkedSemanticSearch = None
    idx: InvertedIndex = None

    def __init__(self, documents):
        self.documents = documents
        self.document_map = {}
        for doc in documents:
            self.document_map[doc["id"]] = doc
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        self.idx.load(documents)

    def _bm25_search(self, query, limit):
        return self.idx.bm25_search(query, limit)
    
    def _semantic_search(self, query, limit):
        return self.semantic_search.search_chunk(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        bm25_results = self._bm25_search(query, limit*50)
        semantic_results = self._semantic_search(query, limit*50)

        bm25_scores = [score for _, score in bm25_results]
        semantic_scores = [result["score"] for result in semantic_results]

        normalized_bm25_scores = normalize_score(bm25_scores) if bm25_scores else []
        normalized_semantic_scores = normalize_score(semantic_scores) if semantic_scores else []

        for i, normalized_semantic_score in enumerate(normalized_semantic_scores):
            semantic_results[i]["score"] = normalized_semantic_score

        def hybrid_score(bm25_score, semantic_score):
            return alpha * bm25_score + (1 - alpha) * semantic_score

        document_scores: dict[int, dict] = {}

        for i, (doc_id, _) in enumerate(bm25_results):
            document_scores[doc_id] = {
                "document": self.document_map[doc_id],
                "bm25_score": normalized_bm25_scores[i],
                "semantic_score": 0,
            }

        for i, result in enumerate(semantic_results):
            doc_id = result["id"]
            if doc_id not in document_scores:
                document_scores[doc_id] = {
                    "document": self.document_map[doc_id],
                    "bm25_score": 0,
                    "semantic_score": normalized_semantic_scores[i],
                }
            else:
                document_scores[doc_id]["semantic_score"] = normalized_semantic_scores[i]

        results: list[dict] = []

        for doc_id, data in document_scores.items():
            bm25_score = data["bm25_score"]
            semantic_score = data["semantic_score"]
            combined_score = hybrid_score(bm25_score, semantic_score)
            results.append(
                {
                    "id": doc_id,
                    "title": data["document"]["title"],
                    "hybrid_score": combined_score,
                    "bm25_score": bm25_score,
                    "semantic_score": semantic_score,
                    "document": data["document"]["description"],
                }
            )

        results.sort(key=lambda x: x["hybrid_score"], reverse=True)

        return results[:limit]

    def rrf_search(self, query, k, limit=10):
        # TODO: Implement RRF search logic
        pass
