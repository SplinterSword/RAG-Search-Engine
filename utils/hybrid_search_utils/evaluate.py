import os
import re
import json
from dotenv import load_dotenv
from google import genai


def _parse_json_scores(text: str, expected_count: int) -> list[int] | None:
    if not text:
        return None

    raw = text.strip()
    if not raw:
        return None

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*\]", raw)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    if not isinstance(parsed, list):
        return None
    if len(parsed) != expected_count:
        return None

    scores: list[int] = []
    for value in parsed:
        if not isinstance(value, int):
            return None
        if value < 0 or value > 3:
            return None
        scores.append(value)

    return scores


def _evaluate_results(query: str, results: list[dict]) -> list[int]:
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    client = genai.Client(api_key=api_key)

    formatted_results = [
        f"{i + 1}. {result.get('title', '')}: {result.get('document', '')}"
        for i, result in enumerate(results)
    ]
    prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:

Query: "{query}"

Results:
{chr(10).join(formatted_results)}

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Do NOT give any numbers out than 0, 1, 2, or 3.

Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

[2, 0, 3, 2, 0, 1]"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    scores = _parse_json_scores(response.text, len(results))
    if scores is None:
        raise RuntimeError("Could not parse evaluation scores from LLM response.")

    return scores

