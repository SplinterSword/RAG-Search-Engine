import argparse
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.hybrid_search_utils.normalize_score import normalize_score
from lib.hybrid_search import HybridSearch
from utils.cli_utils.file_loading import load_movies
from utils.hybrid_search_utils.query_enhancement import enhance_query
from utils.hybrid_search_utils.rerank_methods import rerank

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparser = parser.add_subparsers(dest="command", help="Available commands")
    
    # normalize
    normalize_parser = subparser.add_parser("normalize", help="Normalize text")
    normalize_parser.add_argument("scores", nargs="+", help="List of scores to normalize")

    # weighted_search
    weighted_search_parser = subparser.add_parser("weighted_search", help="Perform weighted search")
    weighted_search_parser.add_argument("query", type=str, help="Query string")
    weighted_search_parser.add_argument("--alpha", type=float, default=0.5, help="Alpha value for weighted search (default: 0.5)")
    weighted_search_parser.add_argument("--limit", type=int, default=5, help="Limit number of results (default: 5)")

    # rrf_search
    rrf_search_parser = subparser.add_parser("rrf_search", help="Perform RRF search")
    rrf_search_parser.add_argument("query", type=str, help="Query string")
    rrf_search_parser.add_argument("-k", type=int, default=60, help="K value for RRF search (default: 60)")
    rrf_search_parser.add_argument("--limit", type=int, default=5, help="Limit number of results (default: 5)")
    rrf_search_parser.add_argument("--enhance", type=str, choices=["spell","rewrite","expand"], help="Query enhancement method")
    rrf_search_parser.add_argument("--rerank-method", type=str, choices=["individual"], help="Rerank method")

    args = parser.parse_args()


    match args.command:
        case "normalize":
            scores = args.scores
            if len(scores) == 0:
                print("No scores provided")
                return
            
            
            # Convert string scores to float
            scores = [float(score) for score in scores]

            normalized_scores = normalize_score(scores)
            
            print("Normalizing scores:", [f"{score:.4f}" for score in normalized_scores])
            return normalized_scores
        
        case "weighted_search":
            query = args.query
            alpha = args.alpha
            limit = args.limit

            documents = load_movies()

            hybrid_search = HybridSearch(documents)
            results = hybrid_search.weighted_search(query, alpha, limit)
            
            for i,result in enumerate(results):
                print(f"{i+1}. {result['title']}\nHybrid Score: {result['hybrid_score']:.4f}\nBM25: {result['bm25_score']:.4f}, Semantic: {result['semantic_score']:.4f}\n{result['document'][:50] + '...'}")

        
        case "rrf_search":
            query = args.query
            k = args.k
            limit = args.limit
            enhance = args.enhance
            rerank_method = args.rerank_method

            if rerank_method is not None:
                limit *= 5

            if enhance:
                query = enhance_query(query, enhance)

            documents = load_movies()

            hybrid_search = HybridSearch(documents)
            results = hybrid_search.rrf_search(query, k, limit)

            if rerank_method:
                limit = limit // 5
                results = rerank(results, rerank_method, query, documents, limit)
            
            for i,result in enumerate(results):
                if rerank_method:
                    print(f"{i+1}. {result['title']}\nRerank Score: {result['rerank_score']:.4f}\nRRF Score: {result['rrf_score']:.4f}\nBM25 Rank: {result['bm25_rank']}, Semantic Rank: {result['semantic_rank']}\n{result['document'][:50] + '...'}")
                else:
                    print(f"{i+1}. {result['title']}\nRRF Score: {result['rrf_score']:.4f}\nBM25 Rank: {result['bm25_rank']}, Semantic Rank: {result['semantic_rank']}\n{result['document'][:50] + '...'}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()