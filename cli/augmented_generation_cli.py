import argparse
import subprocess
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.augmented_utils.gemini import generate_response, generate_summary

def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser("rag", help="Perform RAG (search + generate answer)")
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser("summarize", help="Summarize content")
    summarize_parser.add_argument("query", type=str, help="Search query for summary")
    summarize_parser.add_argument("--limit", type=int, default=5, help="limit of documents")

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query

            cmd = [
                "uv",
                "run",
                "cli/hybrid_search_cli.py",
                "rrf_search",
                query,
                "--k=60",
                "--limit=5",
                "--json",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Error: {result.stderr}")
                return

            try:
                search_results = json.loads(result.stdout)
            except json.JSONDecodeError:
                # Fallback for subprocesses that accidentally print logs before JSON.
                lines = [line for line in result.stdout.splitlines() if line.strip()]
                if not lines:
                    print("Error parsing search results: empty output")
                    if result.stderr:
                        print(result.stderr)
                    return
                try:
                    search_results = json.loads(lines[-1])
                except json.JSONDecodeError:
                    print("Error parsing search results:", result.stdout)
                    if result.stderr:
                        print(result.stderr)
                    return

            print("Search Results:")
            for item in search_results:
                title = item.get("title", "Untitled")
                print(f"  - {title}")

            response = generate_response(query, search_results)
            print("\nRAG Response:")
            print(response)

        case "summarize":
            query = args.query
            limit = args.limit

            cmd = [
                "uv",
                "run",
                "cli/hybrid_search_cli.py",
                "rrf_search",
                query,
                "--k=60",
                f"--limit={limit}",
                "--json",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Error: {result.stderr}")
                return

            try:
                search_results = json.loads(result.stdout)
            except json.JSONDecodeError:
                # Fallback for subprocesses that accidentally print logs before JSON.
                lines = [line for line in result.stdout.splitlines() if line.strip()]
                if not lines:
                    print("Error parsing search results: empty output")
                    if result.stderr:
                        print(result.stderr)
                    return
                try:
                    search_results = json.loads(lines[-1])
                except json.JSONDecodeError:
                    print("Error parsing search results:", result.stdout)
                    if result.stderr:
                        print(result.stderr)
                    return

            print("Search Results:")
            for item in search_results:
                title = item.get("title", "Untitled")
                print(f"  - {title}")

            response = generate_summary(query, search_results)
            print("LLM Summary:")
            print(response)

            
                

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
