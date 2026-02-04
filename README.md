# RAG Search Engine — Keyword and Semantic Search

An educational, hands-on repository to learn and compare two core information retrieval approaches:

- Keyword search with inverted indexes and BM25
- Semantic search with dense embeddings and cosine similarity

Use this repo as a quick revision sheet and a runnable playground with small movie metadata.


## What you will learn

- The difference between keyword vs semantic search
- How TF, IDF, TF-IDF, and BM25 work and how to inspect them
- How sentence-transformer embeddings power semantic similarity search
- How to run both systems end-to-end from the command line


## Project structure

- `cli/`
  - `keyword_search_cli.py`: Commands for building/loading an inverted index and running BM25/TF/IDF queries
  - `semantic_search_cli.py`: Commands for generating embeddings and running semantic search
  - `lib/`
    - `keyword_search.py`: Inverted index, TF/IDF, BM25 scoring and search
    - `semantic_search.py`: Embedding generation, caching, and cosine similarity search
- `utils/`
  - `keyword_seach_utils/`: Preprocessing and BM25 constants/utilities
  - `semantic_search_utils/`: Preprocessing and vector operations (cosine similarity)
- `data/movies.json`: Sample dataset used by both CLIs
- `cache/`: Auto-created; stores built indexes and embeddings
- `pyproject.toml`: Dependencies (Python ≥ 3.13)


## Quick concepts: Keyword vs Semantic search

### Keyword search

- Builds an inverted index mapping terms → document IDs
- Uses exact/normalized token matching (after preprocessing)
- Scoring:
  - TF (term frequency): occurrences of a token in a document
  - IDF (inverse document frequency): how rare a token is across documents
    - A common form: `idf(t) = ln( (N + 1) / (df(t) + 1) )`
  - TF-IDF: `tf × idf`
  - BM25: a strong ranking function that saturates TF and normalizes for document length
    - Uses tunable `k1` (term frequency saturation) and `b` (length normalization)

Pros: transparent, fast, and simple. Cons: brittle to paraphrasing and synonyms.

### Semantic search

- Converts text into dense vectors (embeddings) using `sentence-transformers` (here: `all-MiniLM-L6-v2`)
- Measures similarity via cosine similarity in vector space
- Captures meaning beyond exact words (synonyms, paraphrases, context)

Pros: robust to wording; understands semantics. Cons: needs a model and embeddings; less transparent.


## Requirements and setup

- Python 3.13+
- Install dependencies (editable install is fine):

```bash
pip install -e .
```

Notes:
- The first semantic run will download the sentence-transformer model.
- NLTK may download resources on first use depending on your environment.


## Dataset

- Ensure `data/movies.json` exists (included in the repo). Both CLIs read this file.
- Caches are written to `cache/` automatically.


## Using the Keyword Search CLI

Script: `cli/keyword_search_cli.py`

Common commands:

- Build the inverted index (required once before searching):

```bash
python cli/keyword_search_cli.py build
```

- BM25 search top 5 results:

```bash
python cli/keyword_search_cli.py bm25search "space adventure" 5
```

- Simple keyword search (first 5 docs that match any query token):

```bash
python cli/keyword_search_cli.py search "space adventure"
```

- Inspect TF for a term in a document:

```bash
python cli/keyword_search_cli.py tf 12 "space"
```

- Inspect IDF and BM25 IDF of a term:

```bash
python cli/keyword_search_cli.py idf "space"
python cli/keyword_search_cli.py bm25idf "space"
```

- Inspect BM25 TF (with optional k1, b):

```bash
python cli/keyword_search_cli.py bm25tf 12 "space" 1.2 0.75
```

- Compute TF-IDF for a term in a document:

```bash
python cli/keyword_search_cli.py tfidf 12 "space"
```

Where document IDs correspond to the `id` field in `data/movies.json`.


## Using the Semantic Search CLI

Script: `cli/semantic_search_cli.py`

Common commands:

- Verify the model loads:

```bash
python cli/semantic_search_cli.py verify
```

- Create or load embeddings and verify their shape vs documents:

```bash
python cli/semantic_search_cli.py verify_embeddings
```

- Embed a text/query (quick peek at the vector):

```bash
python cli/semantic_search_cli.py embed_text "galactic rescue mission"
python cli/semantic_search_cli.py embedquery "galactic rescue mission"
```

- Run semantic search (top-k with cosine similarity):

```bash
python cli/semantic_search_cli.py search "galactic rescue mission" --limit 5
```

This will print ranked results with their similarity scores and descriptions.


## How it works (implementation overview)

- Keyword search (`cli/lib/keyword_search.py`)
  - Preprocesses text, builds an inverted index, stores term frequencies and document lengths
  - BM25 scoring via `get_bm25_tf` and `get_bm25_idf`; combined in `bm25` and used in `bm25_search`
  - Index and stats cached in `cache/*.pkl`

- Semantic search (`cli/lib/semantic_search.py`)
  - Uses `SentenceTransformer('all-MiniLM-L6-v2')` to embed documents and queries
  - Embeddings cached to `cache/movie_embeddings.npy`
  - Similarities computed with cosine similarity; top-k results returned


## Troubleshooting

- "Cache directory not found" or "Run 'build' command first" (keyword search)
  - Run: `python cli/keyword_search_cli.py build`

- Embeddings shape mismatch (semantic search)
  - The CLI will auto-rebuild if `movie_embeddings.npy` shape doesn't match the dataset length
  - If issues persist, delete `cache/movie_embeddings.npy` and rerun `verify_embeddings`

- Slow first run (semantic search)
  - The model download happens once; subsequent runs will be faster


## Deep dive: core terms and formulas

- **Tokenization & preprocessing**
  - Normalize case, strip punctuation, (optionally) remove stopwords and apply stemming/lemmatization.
  - This repo applies consistent preprocessing for both indexing and querying to align tokens.

- **Term Frequency (TF)**
  - Raw TF in this repo: the count of a token in a document.
  - Variants you may see elsewhere:
    - Boolean TF: 0/1 if the term appears.
    - Log TF: `1 + ln(tf)` to dampen very high counts.

- **Document Frequency (DF)** and **Inverse Document Frequency (IDF)**
  - DF: number of documents containing the term.
  - IDF in this repo (keyword search): `idf(t) = ln( (N + 1) / (df(t) + 1) )`
    - Intuition: rare terms (small `df`) get larger IDF → more discriminative power.

- **TF-IDF**
  - A simple relevance score: `tfidf(d, t) = tf(d, t) × idf(t)`
  - Good for quick ranking, but can overvalue long documents without length normalization.

- **BM25 (Okapi BM25)**
  - Purpose: improve TF-IDF by saturating TF gains and normalizing for document length.
  - In this repo:
    - Length normalization: `length_norm = 1 - b + b × (|d| / avgdl)`
    - Saturated TF: `bm25_tf = (tf × (k1 + 1)) / (tf + k1 × length_norm)`
    - BM25 IDF: `bm25_idf(t) = ln( (N - df + 0.5) / (df + 0.5) + 1 )`
    - Final per-term score: `bm25(d, t) = bm25_tf × bm25_idf(t)`
  - Tunables:
    - `k1` (default here via utils): controls TF saturation (~1.2–2.0 typical)
    - `b` (default here via utils): degree of length normalization (0=no norm, 1=full)

- **Cosine similarity** (semantic search)
  - Measures angle between vectors (scale-invariant):
    - `cos_sim(a, b) = (a · b) / (||a|| × ||b||)`
  - Range is [-1, 1]; in embedding spaces like `all-MiniLM-L6-v2`, relevant pairs typically show higher positive values.
  - Why cosine? It focuses on direction (semantics) rather than magnitude.

- **Embeddings**
  - `SentenceTransformer('all-MiniLM-L6-v2')` encodes text into dense vectors (dimensions ~384).
  - Semantically similar texts map to nearby vectors → higher cosine similarity.
  - This repo caches document embeddings to `cache/movie_embeddings.npy` and computes query embeddings on the fly.

- **Normalization & scaling tips**
  - Preprocessing consistency is critical: index-time and query-time must match.
  - For keyword search, consider stopword handling—removing very common words can sharpen results.
  - For semantic search, you generally don’t need to L2-normalize manually; many models and libraries handle suitable scaling internally, but cosine inherently normalizes by vector length.

- **Caveats**
  - Keyword search can miss paraphrases and synonyms; semantic search can retrieve paraphrases but might be less transparent.
  - Embeddings are model-dependent; domain-shift can hurt quality—consider domain-specific models for best results.


## Revision cheat sheet

- TF: count of a token in a document
- IDF: `ln((N + 1) / (df + 1))` — down-weights frequent terms
- TF-IDF: `tf × idf` — simple relevance signal
- BM25 (intuition):
  - Saturates TF gains, normalizes long docs, boosts rare terms
  - Tunables: `k1` (how fast TF saturates), `b` (how much length normalization)
- Embeddings + cosine similarity:
  - Convert text → vector; similar meanings → close vectors (high cosine)


## License

Educational use. Adapt as needed for your experiments.

