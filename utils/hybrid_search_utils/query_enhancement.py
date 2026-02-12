import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)

def spell_check(query: str, method: str) -> str:
    prompt = f"""Fix any spelling errors in this movie search query.

Only correct obvious typos. Don't change correctly spelled words.

Query: "{query}"

If no errors, return the original query.
Corrected:"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    print(f"Enhanced query ({method}): '{query}' -> '{response.text}'\n")
    return response.text


def enhance_query(query: str, method: str) -> str:
    if method == "spell":
        query = spell_check(query, method)
    return query