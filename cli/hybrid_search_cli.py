import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparser = parser.add_subparsers(dest="command", help="Available commands")
    
    # normalize
    normalize_parser = subparser.add_parser("normalize", help="Normalize text")
    normalize_parser.add_argument("scores", nargs="+", help="List of scores to normalize")
    
    args = parser.parse_args()


    match args.command:
        case "normalize":
            scores = args.scores
            if len(scores) == 0:
                print("No scores provided")
                return
            
            print("Original scores:", scores)
            # Convert string scores to float
            scores = [float(score) for score in scores]
            min_score = min(scores)
            max_score = max(scores)
            if min_score == max_score:
                normalized_scores = [1.0] * len(scores)
                print("Normalizing scores:", [f"{score:.4f}" for score in normalized_scores])
                return
            
            # Normalize scores
            normalized_scores = [(score - min_score) / (max_score - min_score) for score in scores]

            print("Normalizing scores:", [f"{score:.4f}" for score in normalized_scores])

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()