def _print_rrf_result(rank: int, result: dict, rerank_method: str | None) -> None:
    preview = result["document"][:50] + "..."
    base = (
        f"{rank}. {result['title']}\n"
        f"RRF Score: {result['rrf_score']:.4f}\n"
        f"BM25 Rank: {result['bm25_rank']}, Semantic Rank: {result['semantic_rank']}\n"
        f"{preview}"
    )

    if rerank_method == "cross_encoder":
        print(
            f"{rank}. {result['title']}\n"
            f"Cross-Encoder Score: {result['cross_encoder_score']:.4f}\n"
            f"RRF Score: {result['rrf_score']:.4f}\n"
            f"BM25 Rank: {result['bm25_rank']}, Semantic Rank: {result['semantic_rank']}\n"
            f"{preview}"
        )
        return

    if rerank_method in {"individual", "batch"}:
        rerank_score = result.get("rerank_score")
        if isinstance(rerank_score, (int, float)):
            print(
                f"{rank}. {result['title']}\n"
                f"Rerank Score: {float(rerank_score):.4f}\n"
                f"RRF Score: {result['rrf_score']:.4f}\n"
                f"BM25 Rank: {result['bm25_rank']}, Semantic Rank: {result['semantic_rank']}\n"
                f"{preview}"
            )
            return

    print(base)