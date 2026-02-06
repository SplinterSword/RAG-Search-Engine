#!/usr/bin/env python3

import re
import argparse
from lib.semantic_search import ChunkedSemanticSearch, SemanticSearch, verify_modal, verify_embeddings, embed_query_text
from utils.cli_utils.file_loading import load_movies

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

    # Semantic Chunk
    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Semantic chunk the documents")
    semantic_chunk_parser.add_argument("text", type=str, help="Text to chunk")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, default=4, help="Max Chunk size")
    semantic_chunk_parser.add_argument("--overlap", type=int, default=0, help="Overlap between chunks")

    # Embed Chunks
    subparsers.add_parser("embed_chunks", help="Embed the chunks")

    # search
    search_parser = subparsers.add_parser("search", help="Search for documents")
    search_parser.add_argument("query", type=str, help="Query to search for")
    search_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")

    # search chunked
    search_chunked_parser = subparsers.add_parser("search_chunked", help="Search for documents using chunked embeddings")
    search_chunked_parser.add_argument("query", type=str, help="Query to search for")
    search_chunked_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")
    
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

        case "semantic_chunk":
            splited_words = args.text.split()
            total_char = len(splited_words)
            splited_text = re.split(r'(?<=[.!?])\s+', args.text)
            max_chunk_size = args.max_chunk_size
            overlap = args.overlap
                    
            chunks = []
                    
            for sentence in splited_text:
                words = sentence.split()
                for i in range(0, len(words), max_chunk_size - overlap):
                    chunks.append(' '.join(words[i:i+max_chunk_size]))

            print(f"Semantically Chuncking {total_char} words")
            for i in range(len(chunks)):
                print(f"{i+1}. {chunks[i]} {chunks[i+1] if i+1 < len(chunks) else ''}")
            
            return chunks

        case "embed_chunks":
            semantic_search = ChunkedSemanticSearch()

            documents = load_movies()

            embeddings = semantic_search.load_or_create_chunk_embeddings(documents)
            if embeddings is None:
                print("Failed to load or create embeddings")
                exit(1)

            print(f"Generated {len(embeddings)} chunked embeddings")
            return embeddings

        case "search":
            semantic_search = SemanticSearch()

            documents = load_movies()

            embeddings = semantic_search.load_or_create_embeddings(documents)
            if embeddings is None:
                print("Failed to load or create embeddings")
                exit(1)

            results = semantic_search.search(args.query, args.limit)
            for i, result in enumerate(results):
                print(f"{i+1}. {result['title']} (score: {result['score']:.4f})")
                print(f"   {result['description']}")
            
            return results
        
        case "search_chunked":
            query = args.query
            limit = args.limit
            chunk_semantic_search = ChunkedSemanticSearch()

            documents = load_movies()
            
            embeddings = chunk_semantic_search.load_or_create_chunk_embeddings(documents)
            if embeddings is None:
                print("Failed to load or create embeddings")
                exit(1)

            results = chunk_semantic_search.search_chunk(query, limit)
            for i, result in enumerate(results):
                print(f"\n{i+1}. {result['title']} (score: {result['score']:.4f})")
                print(f"   {result['document']}...")
            
            return results

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()