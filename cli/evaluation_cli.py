import argparse
from pathlib import Path
import json
import subprocess

BASE_DIR = Path(__file__).resolve().parent.parent
data_dir = BASE_DIR / 'data'
golden_response_file = data_dir / 'golden_dataset.json'


def _extract_last_json_list(text: str):
    decoder = json.JSONDecoder()
    parsed = None

    for idx, ch in enumerate(text):
        if ch != "[":
            continue
        try:
            value, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, list):
            parsed = value

    return parsed


def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    # run evaluation logic here
    with open(golden_response_file, 'r') as f:
        golden_dataset = json.load(f)

    for test_case in golden_dataset['test_cases']:
        query = test_case["query"]
        relevant_docs = test_case["relevant_docs"]

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
        completed = subprocess.run(cmd, capture_output=True, text=True)
        output = completed.stdout.strip()

        if completed.returncode != 0:
            err_text = completed.stderr.strip() or completed.stdout.strip() or "Unknown error"
            raise RuntimeError(f"Failed query '{query}': {err_text}")

        if not output:
            raise RuntimeError(f"No output returned for query '{query}'.")

        try:
            rrf_results = json.loads(output)
        except json.JSONDecodeError:
            rrf_results = _extract_last_json_list(output)
            if rrf_results is None:
                raise RuntimeError(f"Invalid JSON output for query '{query}': {output[:200]}")

        relevant_titles = set(relevant_docs)
        relevant_results = [result for result in rrf_results if result["title"] in relevant_titles][:limit]
        
        precision_at_k = len(relevant_results) / limit
        print(f"k={limit}")
        print(f"\n- Query: {query}")
        print(f"  - Precision@{limit}: {precision_at_k:.4f}")
        print(f"  - Retrieved: {', '.join([result['title'] for result in rrf_results][:limit])}")
        print(f"  - Relevant: {', '.join(relevant_docs)}")

    



if __name__ == "__main__":
    main()
