import os

from dotenv import load_dotenv
from google import genai
from google.genai import types


SYSTEM_PROMPT = """Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
- Synthesize visual and textual information
- Focus on movie-specific details (actors, scenes, style, etc.)
- Return only the rewritten query, without any additional commentary"""


load_dotenv()
_api_key = os.environ.get("GEMINI_API_KEY")
_client = genai.Client(api_key=_api_key)


def rewrite_query_from_image(image_bytes: bytes, query: str, mime_type: str) -> types.GenerateContentResponse:
    parts = [
        SYSTEM_PROMPT,
        types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
        query.strip(),
    ]

    return _client.models.generate_content(
        model="gemini-2.5-flash",
        contents=parts,
    )
