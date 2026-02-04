from sentence_transformers import SentenceTransformer

class SemanticSearch:
    modal = None
    
    def __init__(self):
        self.modal = SentenceTransformer('all-MiniLM-L6-v2')

    def generate_embedding(self, text: str):
        if text == "" or text is None or len(text.strip()) == 0:
            raise ValueError("Text cannot be empty or None or only contains whitespace")
        
        embeddings = self.modal.encode([text])
        return embeddings[0]


def verify_modal():
    try:
        sematic_search = SemanticSearch()
        print(f"Model loaded: {sematic_search.modal}")
        print(f"Max sequence length: {sematic_search.modal.max_seq_length}")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def embed_text(text: str):
    sematic_search = SemanticSearch()
    embedding = sematic_search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")
    return embedding
