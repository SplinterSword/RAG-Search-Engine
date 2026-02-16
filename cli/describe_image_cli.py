#!/usr/bin/env python3

import argparse
import mimetypes
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.image_cli_utils.gemini import rewrite_query_from_image


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rewrite a text query using image context with Gemini"
    )
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument("--query", required=True, help="Query to rewrite")
    args = parser.parse_args()

    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"

    with open(args.image, "rb") as image_file:
        image_bytes = image_file.read()

    response = rewrite_query_from_image(image_bytes, args.query, mime)

    print(f"Rewritten query: {response.text.strip()}")
    if response.usage_metadata is not None:
        print(f"Total tokens:    {response.usage_metadata.total_token_count}")


if __name__ == "__main__":
    main()
