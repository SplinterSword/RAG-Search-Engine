#!/usr/bin/env python3

import argparse
from lib.semantic_search import verify_modal, verify_embeddings, embed_query_text

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

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()