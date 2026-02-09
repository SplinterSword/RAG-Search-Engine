def normalize_score(scores: list[float]) -> list[float]:
    
    min_score = min(scores)
    max_score = max(scores)
    if min_score == max_score:
        normalized_scores = [1.0] * len(scores)
        print("Normalizing scores:", [f"{score:.4f}" for score in normalized_scores])
        return normalized_scores
    
    # Normalize scores
    normalized_scores = [(score - min_score) / (max_score - min_score) for score in scores]
    return normalized_scores