import argparse
from lib.multimodal_search import verify_image_embedding, image_search_command

def main():
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("verify_image_embedding", help="Verify Image Embeddings")
    search_parser.add_argument("image", type=str, help="Path to image file")

    image_search_parser = subparsers.add_parser("image_search", help="Search with image")
    image_search_parser.add_argument("image", type=str, help="Path to image file")

    args = parser.parse_args()
    
    match args.command:

        case "verify_image_embedding":
            image_path = args.image
            print(f"Verifying image embedding for: {image_path}")

            embeddings = verify_image_embedding(image_path)

            embedding_description = f"Embedding shape: {embeddings.shape[0]} dimensions"
            
            print(embedding_description)

        case "image_search":
            image_path = args.image
            print(f"Searching with image: {image_path}")

            results = image_search_command(image_path)

            for result in results:
                print(f"{result['title']} (similarity: {result['score']})")
                print(f"{result['description']}")
                print()

if __name__ == "__main__":
    main()