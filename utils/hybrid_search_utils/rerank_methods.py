import os
from time import sleep
from dotenv import load_dotenv
from google import genai
import json
import re

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)


def _parse_json_list_of_ids(text):
    if not text:
        return None

    raw = text.strip()
    if not raw:
        return None

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Gemini may return markdown fenced JSON or extra prose around the list.
        match = re.search(r"\[[\s\S]*\]", raw)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    if not isinstance(parsed, list):
        return None

    ids = []
    for value in parsed:
        try:
            ids.append(int(value))
        except (TypeError, ValueError):
            continue

    return ids if ids else None


def _parse_float_score(text):
    if not text:
        return 0.0

    raw = text.strip()
    if not raw:
        return 0.0

    try:
        return float(raw)
    except ValueError:
        match = re.search(r"-?\d+(?:\.\d+)?", raw)
        if not match:
            return 0.0
        return float(match.group(0))

def rerank_individual(results, query, documents, limit):
    
    for result in results:
        prompt = f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {result.get("title", "")} - {result.get("document", "")}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Give me ONLY the number in your response, no other text or explanation.

Score:"""
    
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        
        result["rerank_score"] = _parse_float_score(response.text)
        sleep(3)

    results.sort(key=lambda x: x["rerank_score"], reverse=True)

    results = results[:limit]
    
    return results
    
def rerank_batch(results, query, documents, limit):

    prompt = f"""Rank these movies by relevance to the search query.

Query: "{query}"

Movies:
{results}

Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else. For example:

[75, 12, 34, 2, 1]
"""
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    new_ranking = _parse_json_list_of_ids(response.text)
    if not new_ranking:
        return results[:limit]

    ranking_map = {doc_id: rank for rank, doc_id in enumerate(new_ranking)}
    sorted_results = sorted(
        results,
        key=lambda x: ranking_map.get(int(x["id"]), len(ranking_map))
    )

    for i, result in enumerate(sorted_results):
        result["rerank_score"] = i + 1
    
    results = sorted_results[:limit]
    
    return results

def rerank_cross_encoder(results, query, documents, limit):
    from sentence_transformers import CrossEncoder

    pairs = []
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
    for result in results:
        pairs.append([query, f"{result.get('title', '')} - {result.get('document', '')}"])
        result["cross_encoder_score"] = 0
    
    scores = cross_encoder.predict(pairs)

    for i, result in enumerate(results):
        result["cross_encoder_score"] = scores[i]

    results.sort(key=lambda x: x["cross_encoder_score"], reverse=True)

    results = results[:limit]
    
    return results

def rerank(results, rerank_method, query, documents, limit):
    if rerank_method == "individual":
        return rerank_individual(results, query, documents, limit)
    elif rerank_method == "batch":
        return rerank_batch(results, query, documents, limit)
    elif rerank_method == "cross_encoder":
        return rerank_cross_encoder(results, query, documents, limit)
