import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.hybrid_search import HybridSearch
from utils.cli_utils.file_loading import load_movies
from utils.hybrid_search_utils.query_enhancement import enhance_query
from utils.hybrid_search_utils.rerank_methods import rerank
from utils.hybrid_search_utils.print import _print_rrf_result


logger = logging.getLogger(__name__)


def _enable_debug_logging() -> None:
    if logger.handlers:
        return
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False


def _summarize_results(results: list[dict], limit: int = 5) -> list[dict]:
    summary = []
    for result in results[:limit]:
        summary.append(
            {
                "id": result.get("id"),
                "title": result.get("title"),
                "rrf_score": result.get("rrf_score"),
                "rerank_score": result.get("rerank_score"),
                "cross_encoder_score": result.get("cross_encoder_score"),
            }
        )
    return summary


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
    rrf_search_parser = subparser.add_parser("rrf_search", aliases=["rrf-search"], help="Perform RRF search")
    rrf_search_parser.add_argument("query", type=str, help="Query string")
    rrf_search_parser.add_argument("-k", "--k", type=int, default=60, help="K value for RRF search (default: 60)")
    rrf_search_parser.add_argument("--limit", type=int, default=5, help="Limit number of results (default: 5)")
    rrf_search_parser.add_argument("--enhance", type=str, choices=["spell","rewrite","expand"], help="Query enhancement method")
    rrf_search_parser.add_argument("--rerank-method", type=str, choices=["individual", "batch", "cross_encoder"], help="Rerank method")
    rrf_search_parser.add_argument("--json", action="store_true", help="Return results as JSON")
    rrf_search_parser.add_argument("--debug", action="store_true", help="Enable debug logging for pipeline stages")

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

        
        case "rrf_search" | "rrf-search":
            query = args.query
            k = args.k
            limit = args.limit
            enhance = args.enhance
            rerank_method = args.rerank_method
            json_output = args.json
            debug = args.debug

            if debug:
                _enable_debug_logging()

            original_query = query
            logger.debug("Original query: %s", original_query)

            if k <= 0:
                parser.error("rrf_search requires k > 0")
            if limit <= 0:
                parser.error("rrf_search requires --limit > 0")

            if enhance:
                query = enhance_query(query, enhance)
            logger.debug("Query after enhancement: %s", query)

            documents = load_movies()
            result_limit = min(limit, len(documents))
            search_limit = min(result_limit * 5, len(documents)) if rerank_method else result_limit

            hybrid_search = HybridSearch(documents)
            results = hybrid_search.rrf_search(query, k, search_limit)
            logger.debug(
                "Results after RRF search (top %s): %s",
                min(5, len(results)),
                _summarize_results(results),
            )

            if rerank_method:
                results = rerank(results, rerank_method, query, documents, result_limit)
                logger.debug(
                    "Final results after rerank (%s, top %s): %s",
                    rerank_method,
                    min(5, len(results)),
                    _summarize_results(results),
                )
            else:
                logger.debug(
                    "Final results (no rerank, top %s): %s",
                    min(5, len(results)),
                    _summarize_results(results),
                )
            
            if not results:
                if json_output:
                    print("[]")
                else:
                    print("No results found.")
                return

            if json_output:
                print(json.dumps(results))
                return results

            for i, result in enumerate(results, start=1):
                _print_rrf_result(i, result, rerank_method)
            
            return results

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
