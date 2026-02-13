import logging
import sys

def _enable_debug_logging(logger: logging.Logger) -> None:
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
