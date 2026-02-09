#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.keyword_seach_utils.tfidf_utils import get_bm25_idf, bm25_tf_command
from utils.keyword_seach_utils.text_preprocessing import text_preprocessing
from lib.keyword_search import InvertedIndex
from utils.keyword_seach_utils.search_utils import BM25_K1, BM25_B
from utils.keyword_seach_utils.load_utils import load_movies

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # build index
    subparsers.add_parser("build", help="Build and cache the inverted index")

    # search
    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    # get term frequency
    tf_parser = subparsers.add_parser("tf", help="Get term frequency")
    tf_parser.add_argument("document_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to get frequency for")

    # get inverse document frequency (idf)
    idf_parser = subparsers.add_parser("idf", help="Get inverse document frequency")
    idf_parser.add_argument("term", type=str, help="Term to get inverse document frequency for")

    # get bm25 idf
    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 inverse document frequency")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 inverse document frequency for")

    # get tf-idf
    tf_idf_parser = subparsers.add_parser("tfidf", help="Get TF-IDF")
    tf_idf_parser.add_argument("document_id", type=int, help="Document ID")
    tf_idf_parser.add_argument("term", type=str, help="Term to get TF-IDF for")

    # get bm25 tf
    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given document ID and term")
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 b parameter")

    # bm25 search
    bm25_search_parser = subparsers.add_parser("bm25search", help="Search movies using BM25")
    bm25_search_parser.add_argument("query", type=str, help="Search query")
    bm25_search_parser.add_argument("limit", type=int, nargs='?', default=5, help="Number of results to return")

    args = parser.parse_args()

    match args.command:
        case "build":
            print("Building index...")
            index = InvertedIndex()
            with open("data/movies.json", "r") as f:
                data = json.load(f)
            index.build(data["movies"])
            index.save()

            print("Index built and saved.")


        case "tf":
            print("Getting TF for:", args.term, "in document", args.document_id)

            movies = load_movies()

            index = InvertedIndex()
            index.load(movies)
            
            tf = index.get_tf(args.document_id, args.term)
            print(tf)
            return tf

        case "bm25tf":
            print("Getting BM25 TF for:", args.term, "in document", args.doc_id)

            bm25tf = bm25_tf_command(args.doc_id, args.term, args.k1, args.b)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")
            return bm25tf

        case "idf":
            print("Getting IDF for:", args.term)

            movies = load_movies()

            index = InvertedIndex()
            index.load(movies)
            
            idf = index.get_idf(args.term)

            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
            return idf

        case "bm25idf":
            print("Getting BM25 IDF for:", args.term)

            bm25idf = get_bm25_idf(args.term)
            
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
            return bm25idf

        case "tfidf":
            print("Getting TF-IDF for:", args.term, "in document", args.document_id)

            movies = load_movies()

            index = InvertedIndex()
            index.load(movies)
            
            tf = index.get_tf(args.document_id, args.term)
            idf = index.get_idf(args.term)
            tf_idf = tf * idf
            print(f"TF-IDF score of '{args.term}' in document '{args.document_id}': {tf_idf:.2f}")
            return tf_idf
        
        case "bm25search":
            print("Searching for:", args.query)

            movies = load_movies()

            index = InvertedIndex()
            index.load(movies)
            
            results = index.bm25_search(args.query, args.limit)
            for doc_id, score in results:
                print(f"({index.docmap[doc_id]['id']}) {index.docmap[doc_id]['title']} - Score: {score:.2f}")
            return results

        case "search":
            print("Searching for:", args.query)

            movies = load_movies()

            index = InvertedIndex()
            index.load(movies)
            
            query_tokens = text_preprocessing(args.query)
            seen_doc_ids = []
            for token in query_tokens:
                docs = index.get_documents(token)
                for doc_id in docs:
                    if len(seen_doc_ids) >= 5:
                        break
                    if doc_id not in seen_doc_ids:
                        seen_doc_ids.append(doc_id)

            for doc_id in seen_doc_ids:
                print(index.docmap[doc_id]["id"], index.docmap[doc_id]["title"])            
            return [index.docmap[d] for d in seen_doc_ids]
            
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()