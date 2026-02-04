#!/usr/bin/env python3

import json
import argparse
from pathlib import Path
from lib.semantic_search import SemanticSearch, verify_modal, verify_embeddings, embed_query_text

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Verify Modal
    subparsers.add_parser("verify", help="Verify the semantic search model")
    
    # Embed Text
    embed_text_parser = subparsers.add_parser("embed_text", help="Embed a text")
    embed_text_parser.add_argument("text", type=str, help="Text to embed")

    # Embed Query
    embed_query_parser = subparsers.add_parser("embedquery", help="Embed a query")
    embed_query_parser.add_argument("text", type=str, help="Query to embed")

    # Verify Embeddings
    subparsers.add_parser("verify_embeddings", help="Verify the embeddings")

    # Chunk
    chunk_parser = subparsers.add_parser("chunk", help="Chunk the documents")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, default=10, help="Chunk size")
    chunk_parser.add_argument("--overlap", type=int, default=0, help="Overlap between chunks")

    # search
    search_parser = subparsers.add_parser("search", help="Search for documents")
    search_parser.add_argument("query", type=str, help="Query to search for")
    search_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")
    
    args = parser.parse_args()

    match args.command:
        case "verify":
            verified = verify_modal()

            if verified:
                print("Modal verified successfully")
            else:
                print("Failed to verify modal")
                exit(1)
        
        case "embed_text":
            embedding = embed_query_text(args.text)
            return embedding
        
        case "embedquery":
            embedding = embed_query_text(args.text)
            return embedding

        case "verify_embeddings":
            verified = verify_embeddings()
            if verified:
                print("Embeddings verified successfully")
            else:
                print("Malformed embeddings are produced")
                exit(1)

        case "chunk":
            splited_text = args.text.split()
            total_char = len(splited_text)
            n = args.chunk_size
            overlap = args.overlap

            chunks = []

            for i in range(0, total_char, n - overlap):
                chunks.append(' '.join(splited_text[i:i+n]))

            print(f"Chunking {total_char} characters")
            for i, chunk in enumerate(chunks):
                print(f"{i+1}. {chunk}")
            
            return chunks

        case "search":
            semantic_search = SemanticSearch()

            BASE_DIR = Path(__file__).resolve().parent.parent
            data_dir = BASE_DIR / 'data'
            data_file = data_dir / 'movies.json'

            with open(data_file, 'r') as f:
                data = json.load(f)

            documents = data['movies']

            embeddings = semantic_search.load_or_create_embeddings(documents)
            if embeddings is None:
                print("Failed to load or create embeddings")
                exit(1)

            results = semantic_search.search(args.query, args.limit)
            for i, result in enumerate(results):
                print(f"{i+1}. {result['title']} (score: {result['score']:.4f})")
                print(f"   {result['description']}")
            
            return results

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()