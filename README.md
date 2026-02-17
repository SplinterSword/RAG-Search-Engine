# RAG Search Engine: Full Technical Documentation

This repository implements a movie search system that combines lexical retrieval, dense semantic retrieval, reciprocal-rank fusion, optional reranking, multimodal query support, and retrieval-augmented generation.

The codebase is organized as runnable CLIs over shared retrieval libraries, with cached indexes/embeddings and optional Gemini-based query enhancement and judging.

## 1. System Scope and What Is Actually Implemented

Implemented:
- Keyword retrieval with inverted index + BM25: `cli/lib/keyword_search.py`
- Semantic retrieval with sentence embeddings + cosine similarity: `cli/lib/semantic_search.py`
- Chunk-based semantic retrieval (max-score aggregation per document): `cli/lib/semantic_search.py`
- Hybrid retrieval:
  - Weighted score fusion API: `HybridSearch.weighted_search`
  - Reciprocal Rank Fusion (RRF): `HybridSearch.rrf_search`
  - `cli/hybrid_search_cli.py`
- Query enhancement with Gemini (`spell`, `rewrite`, `expand`): `utils/hybrid_search_utils/query_enhancement.py`
- Reranking:
  - Gemini individual scoring
  - Gemini batch list ranking
  - Local cross-encoder reranking
  - `utils/hybrid_search_utils/rerank_methods.py`
- Evaluation:
  - Golden-set precision/recall/F1@k: `cli/evaluation_cli.py`
  - LLM-as-a-judge 0-3 relevance: `utils/hybrid_search_utils/evaluate.py`
- Multimodal search (image query against text corpus using CLIP encoder): `cli/lib/multimodal_search.py`
- Image-informed query rewrite (Gemini vision): `utils/image_cli_utils/gemini.py`, `cli/describe_image_cli.py`
- Retrieval-Augmented Generation workflows (`rag`, `summarize`, `citations`, `question`): `cli/augmented_generation_cli.py`, `utils/augmented_utils/gemini.py`

Not implemented (important):
- No vector database (FAISS/Chroma/Pinecone/Milvus/Qdrant are not used)
- No ANN index; semantic search is brute-force cosine over all vectors
- No learned hybrid weighting or feedback loop training

## 2. High-Level Architecture

### 2.1 End-to-end query pipeline (RRF + optional rerank + optional generation)

```text
User Query
   |
   +--> Optional Query Enhancement (spell/rewrite/expand via Gemini)
   |
   +--> BM25 Search (Inverted Index)
   |
   +--> Semantic Search (Chunk Embeddings + Cosine)
   |
   +--> Fusion
        |- Weighted Fusion (alpha blend) OR
        \- Reciprocal Rank Fusion (RRF)
   |
   +--> Optional Reranking
        |- Gemini individual score
        |- Gemini batch ordering
        \- Cross-Encoder (ms-marco-TinyBERT-L2-v2)
   |
   +--> Optional LLM Evaluation (0-3 judge)
   |
   +--> Optional Augmented Generation (Gemini answer/summarize/citations)
   |
Final Ranked Results / Generated Response
```

### 2.2 Indexing and cache pipeline

```text
movies.json
   |
   +--> Keyword indexing
   |      - tokenization / stopword removal / stemming
   |      - inverted postings + TF + doc lengths
   |      - cache/index.pkl, docmap.pkl, term_frequency.pkl, doc_length.pkl
   |
   +--> Semantic indexing
          - text normalization
          - bi-encoder embeddings (all-MiniLM-L6-v2)
          - chunk embeddings + metadata
          - cache/movie_embeddings.npy
          - cache/chunk_embeddings.npy
          - cache/chunk_metadata.json
```

### 2.3 Multimodal branch

```text
Input Image
   |
   +--> CLIP image embedding (clip-ViT-B-32)
   |
   +--> Cosine similarity vs CLIP text embeddings of movie title+description
   |
Top-K movies (image-text retrieval)
```

## 3. Data and Runtime Components

- Dataset: `data/movies.json` (5000 movie documents)
- Golden evaluation set: `data/golden_dataset.json` (10 test queries)
- Stopword file: `data/stopwords.txt`
- Cache directory (auto-generated): `cache/`
- Environment variable for Gemini features: `GEMINI_API_KEY`

## 4. Module-by-Module Implementation Details

## 4.1 Keyword search internals

Primary implementation: `cli/lib/keyword_search.py`

Core structures:
- `index: dict[str, list[int]]`: token -> postings list of doc IDs
- `docmap: dict[int, dict]`: doc ID -> full movie record
- `term_frequency: dict[int, Counter]`: per-document term counts
- `doc_length: dict[int, int]`: tokenized doc lengths

Build path:
- `build(movies)` calls private `__add_document(doc_id, text)`
- Indexed text is `"{title} {description}"`
- Preprocessing uses `utils/keyword_seach_utils/text_preprocessing.py`

Search path:
- `bm25_search(query, limit)` scores every document for all query tokens
- Returns top-k `(doc_id, score)` sorted descending

Formulas implemented:
- Raw TF: `tf(d,t)` from `term_frequency`
- IDF:
  `idf(t) = ln((N + 1) / (df(t) + 1))`
- BM25 TF saturation:
  `bm25_tf(d,t) = tf(d,t) * (k1 + 1) / (tf(d,t) + k1 * (1 - b + b * |d|/avgdl))`
- BM25 IDF:
  `bm25_idf(t) = ln(((N - df(t) + 0.5)/(df(t) + 0.5)) + 1)`
- BM25 term score:
  `bm25(d,t) = bm25_tf(d,t) * bm25_idf(t)`

Defaults:
- `k1 = 1.5`, `b = 0.75` from `utils/keyword_seach_utils/search_utils.py`

Caching:
- `save()` writes 4 pickle files into `cache/`
- `load(documents)` loads cache or rebuilds if missing

CLI surface:
- `cli/keyword_search_cli.py`
  - `build`, `search`, `tf`, `idf`, `bm25idf`, `tfidf`, `bm25tf`, `bm25search`

## 4.2 Semantic search internals

Primary implementation: `cli/lib/semantic_search.py`

Bi-encoder model:
- `SentenceTransformer("all-MiniLM-L6-v2")`

Document embedding:
- Uses string form: `"{title}: {description}"`
- Stored in `cache/movie_embeddings.npy`

Query embedding:
- `generate_embedding(text)` with semantic text preprocessing:
  - unicode normalization, lowercase, punctuation strip, whitespace cleanup
  - `utils/semantic_search_utils/text_preprocessing.py`

Similarity:
- `cosine_similarity(query_vec, doc_vec)` in `utils/semantic_search_utils/vector_operations.py`
- Brute-force over all embeddings; sorted descending

Chunked semantic retrieval (`ChunkedSemanticSearch`):
- Splits each description via `semantic_chunk(text, 4, 1)`
- Encodes all chunks, stores:
  - `cache/chunk_embeddings.npy`
  - `cache/chunk_metadata.json` (movie_idx/chunk_idx/total_chunks)
- `search_chunk(query, limit)`:
  - computes query-chunk cosine
  - keeps max chunk score per movie
  - ranks movies by that max score

Why chunking helps here:
- Long descriptions can dilute full-document embeddings
- Max-over-chunks better preserves local relevance signals

## 4.3 Hybrid retrieval internals

Primary implementation: `cli/lib/hybrid_search.py`

HybridSearch initialization:
- Loads chunk semantic index (`ChunkedSemanticSearch`)
- Loads keyword index (`InvertedIndex`)
- Maintains `document_map`

### Weighted fusion

Path: `HybridSearch.weighted_search(query, alpha, limit)`

Behavior:
- Runs BM25 and semantic searches with expanded candidate size `limit*50`
- Min-max normalizes each score list via `utils/hybrid_search_utils/normalize_score.py`
- Combines per doc via weighted sum

Formula (intended):
`hybrid_score = alpha * bm25_norm + (1 - alpha) * semantic_norm`

Helper:
- `utils/hybrid_search_utils/score_utils.py::hybrid_score`

### Reciprocal Rank Fusion (RRF)

Path: `HybridSearch.rrf_search(query, k, limit)`

Behavior:
- Gets BM25 and semantic ranked lists (`limit*50` candidates internally)
- Assigns ranks per list
- Fuses by reciprocal rank contribution from each retriever

Formula:
`RRF(d) = 1/(k + rank_bm25(d)) + 1/(k + rank_semantic(d))`
- Missing rank contributes `0`
- `k` defaults to `60` in CLI

Helper:
- `utils/hybrid_search_utils/score_utils.py::rrf_score`

CLI surface:
- `cli/hybrid_search_cli.py`
  - `weighted_search`
  - `rrf_search` / alias `rrf-search`
  - options: `--enhance`, `--rerank-method`, `--debug`, `--evaluate`, `--json`

## 4.4 Query enhancement

Implementation: `utils/hybrid_search_utils/query_enhancement.py`

Methods:
- `spell`: typo correction
- `rewrite`: concise specific rewrite using movie priors
- `expand`: append related terms/synonyms

All methods call Gemini (`gemini-2.5-flash`) and return text used as modified query.

Design impact:
- Helps lexical recall and may improve semantic focus
- Adds API latency/cost and introduces model variance

## 4.5 Reranking layer

Implementation: `utils/hybrid_search_utils/rerank_methods.py`

Modes:
- `individual`:
  - prompt each result independently for score 0-10
  - parse float via regex fallback
  - sleeps 3 seconds between calls
- `batch`:
  - prompt with whole candidate list
  - ask for JSON ordered IDs
  - parse and reorder by returned ID ranking
- `cross_encoder`:
  - local model `cross-encoder/ms-marco-TinyBERT-L2-v2`
  - score query-document pairs jointly

API entrypoint:
- `rerank(results, rerank_method, query, documents, limit)`

Output:
- Annotates scores (`rerank_score` or `cross_encoder_score`)
- Returns top `limit`

## 4.6 Evaluation layer

### Automatic relevance metrics (golden set)

Implementation: `cli/evaluation_cli.py`

Process:
1. Load `data/golden_dataset.json`
2. For each query, run subprocess:
   - `uv run cli/hybrid_search_cli.py rrf_search <query> --k=60 --limit=<k> --json`
3. Compare retrieved titles to known relevant titles
4. Compute metrics per query

Formulas:
- `Precision@k = (# relevant retrieved in top k) / k`
- `Recall@k = (# relevant retrieved in top k) / (# relevant documents)`
- `F1@k = 2 * P@k * R@k / (P@k + R@k)` (0 if denominator is 0)

### LLM-as-a-judge evaluation

Implementation: `utils/hybrid_search_utils/evaluate.py`

Process:
- Prompts Gemini to rate each result on 0..3 scale
- Requires strict JSON array output length == number of results
- Parsing is robust to fenced/prose output

Scale:
- 3 highly relevant
- 2 relevant
- 1 marginally relevant
- 0 not relevant

## 4.7 Augmented generation layer (RAG)

CLIs and helpers:
- `cli/augmented_generation_cli.py`
- `utils/augmented_utils/rrf_search.py`
- `utils/augmented_utils/gemini.py`

Retrieval path:
- Always retrieves via hybrid RRF (`k=60`) using subprocess call to `cli/hybrid_search_cli.py ... --json`

Generation modes:
- `rag`: general answer generation
- `summarize`: condensed synthesis across retrieved movies
- `citations`: source-citation style answer (`[1]`, `[2]`, ... requested)
- `question`: conversational QA style response

Prompting strategy:
- retrieved docs are injected directly as serialized Python list content
- no token-aware truncation/chunk selection policy implemented
- no explicit groundedness verifier/citation checker implemented

## 4.8 Multimodal retrieval layer

Implementation:
- `cli/lib/multimodal_search.py`
- `cli/multimodal_search_cli.py`

Model:
- `SentenceTransformer("clip-ViT-B-32")`

Indexing:
- Encodes all text documents (`"title: description"`) in CLIP text space

Query:
- Encodes input image via PIL + CLIP image encoder
- Computes cosine similarity vs text embeddings
- Returns top 5 movies

Image-assisted query rewriting:
- `utils/image_cli_utils/gemini.py`
- `cli/describe_image_cli.py`
- Gemini receives image bytes + text query and rewrites search query

## 5. Concept Glossary (Definitions, Formula, Intuition, Usage, Pros/Cons)

Each concept below is tied to repository code.

### Term Frequency (TF)
- Definition: number of times term `t` appears in document `d`.
- Formula: `TF(d,t) = count(d,t)`
- Intuition: repeated words can indicate topical importance.
- Used in: `cli/lib/keyword_search.py::get_tf`
- Pros: simple, interpretable.
- Cons: can over-reward long/repetitive documents.
- Use when: exact lexical matching matters.

### Inverse Document Frequency (IDF)
- Definition: rarity weighting for a term across corpus.
- Formula (implemented): `IDF(t) = ln((N+1)/(df(t)+1))`
- Intuition: rare terms carry more discriminative power.
- Used in: `cli/lib/keyword_search.py::get_idf`
- Pros: downweights ubiquitous terms.
- Cons: unstable on very small corpora.
- Use when: scoring should penalize common words.

### TF-IDF
- Definition: product of term frequency and inverse document frequency.
- Formula: `TFIDF(d,t) = TF(d,t) * IDF(t)`
- Intuition: high if term is frequent in doc and rare globally.
- Used in: `cli/keyword_search_cli.py::tfidf` command.
- Pros: baseline sparse ranking feature.
- Cons: no length saturation/normalization by default.
- Use when: lightweight lexical scoring/debugging.

### BM25
- Definition: probabilistic lexical ranking with TF saturation and length normalization.
- Formulas used:
  - `bm25_tf = tf*(k1+1)/(tf + k1*(1-b + b*|d|/avgdl))`
  - `bm25_idf = ln(((N-df+0.5)/(df+0.5))+1)`
  - `bm25 = bm25_tf * bm25_idf`
- Used in: `cli/lib/keyword_search.py`
- Pros: strong classical baseline, robust lexical retrieval.
- Cons: cannot capture semantic paraphrase.
- Use when: exact tokens and explainable ranking are needed.

### BM25-IDF
- Definition: the BM25-specific IDF term above.
- Used in: `get_bm25_idf` and `bm25idf` CLI command.
- Note: differs from plain IDF smoothing.

### Semantic Search
- Definition: retrieve by vector similarity in embedding space.
- Formula: cosine similarity between query/document vectors.
- Used in: `cli/lib/semantic_search.py`
- Pros: handles synonymy and paraphrase.
- Cons: less transparent, model-dependent.
- Use when: meaning match > exact wording.

### Embeddings
- Definition: dense vectors encoding semantic content.
- Used models:
  - `all-MiniLM-L6-v2` (text)
  - `clip-ViT-B-32` (text-image shared space)
- Used in: `cli/lib/semantic_search.py`, `cli/lib/multimodal_search.py`

### Vector Space Model
- Definition: represent text/image as vectors and compute geometric similarity.
- Used in: semantic and multimodal retrieval modules.

### Cosine Similarity
- Definition: angle-based similarity between vectors.
- Formula: `cos(a,b) = (a·b)/(||a|| ||b||)`
- Used in: `utils/semantic_search_utils/vector_operations.py`
- Pros: scale-invariant.
- Cons: still O(N) scan in this codebase.

### Hybrid Search
- Definition: combine lexical and semantic retrievers.
- Implemented strategies:
  - weighted normalized score fusion
  - RRF rank fusion
- Used in: `cli/lib/hybrid_search.py`

### Weighted Scoring
- Definition: linear blend of normalized retriever scores.
- Formula: `alpha*sparse + (1-alpha)*dense`
- Used in helper: `utils/hybrid_search_utils/score_utils.py::hybrid_score`
- Risk: sensitive to calibration and normalization quality.

### RRF (Reciprocal Rank Fusion)
- Definition: fuse based on rank positions, not raw scores.
- Formula: `sum_i 1/(k + rank_i(d))`
- Used in: `cli/lib/hybrid_search.py::rrf_search`
- Pros: robust across heterogeneous score scales.
- Cons: ignores absolute score margins.

### Cross-Encoder
- Definition: reranker that scores `(query, doc)` jointly in one model forward pass.
- Used in: `rerank_cross_encoder` with `ms-marco-TinyBERT-L2-v2`.
- Pros: usually better precision at top ranks.
- Cons: higher latency than bi-encoder retrieval.

### Bi-Encoder
- Definition: independently encode query and docs, compare vectors.
- Used in: `SemanticSearch` and `ChunkedSemanticSearch`.
- Pros: scalable precompute + fast similarity.
- Cons: less precise than cross-encoder rerank.

### Reranking
- Definition: reorder top candidates after first-pass retrieval.
- Used in: Gemini (`individual`, `batch`) and cross-encoder methods.

### Query Expansion
- Definition: append related terms to broaden lexical match.
- Used in: `enhance_query(..., "expand")`.
- Benefit: recall gains.
- Risk: topic drift.

### Query Enhancement / Rewrite
- Definition: model-driven reformulation for specificity/clarity.
- Used in: `spell`, `rewrite`, `expand`; and image-conditioned rewrite path.

### Precision / Recall / F1
- Precision@k: relevance purity of top-k.
- Recall@k: coverage of relevant set in top-k.
- F1@k: harmonic mean balancing both.
- Used in: `cli/evaluation_cli.py`

### Manual Evaluation
- Definition: human judging relevance quality.
- Status in repo: golden dataset appears manually curated (`data/golden_dataset.json`).

### LLM-based Evaluation
- Definition: LLM judges each result on a rubric.
- Used in: `utils/hybrid_search_utils/evaluate.py`
- Risk: judge bias/variance and prompt sensitivity.

### Multimodal Search
- Definition: retrieval across modalities (image query to text docs in shared embedding space).
- Used in: `cli/lib/multimodal_search.py`

### Retrieval-Augmented Generation (RAG) / Augmented Generation
- Definition: retrieve supporting docs, then generate answer conditioned on them.
- Used in: `cli/augmented_generation_cli.py` + Gemini helpers.
- Difference from retrieval-only: adds synthesis/generation, not just ranking.

### Relevance Scoring
- Implemented forms:
  - BM25 score
  - Cosine similarity
  - RRF score
  - Cross-encoder score
  - Gemini rerank score

### Embedding Indexing
- Definition: precompute/store corpus embeddings for later query-time similarity.
- Used in: `.npy` caches for semantic and chunk embeddings.

### Vector Database
- Status: not used in this project.
- Current approach: in-memory numpy arrays + linear scan.

### ANN (Approximate Nearest Neighbors)
- Status: not implemented.
- Implication: exact scan with O(N) query-time complexity in vector retrieval.

## 6. Design Decisions and Alternatives

### Why not pure keyword search?
- Pure BM25 fails on synonymy/paraphrases and indirect phrasing.
- Semantic branch addresses this by matching meaning, not just tokens.

### Why not pure semantic search?
- Dense retrieval can miss exact rare entities/spellings; lexical precision for named terms can be stronger.
- BM25 branch preserves hard lexical constraints.

### Why hybrid?
- Combines lexical precision and semantic recall.
- More robust across varied query types in movie discovery.

### Why RRF over weighted-only fusion?
- Weighted fusion depends on score calibration and normalization quality.
- RRF uses ranks, typically more stable across retrievers with different score distributions.

### Why cross-encoder reranking?
- First-pass retrieval is recall-oriented.
- Cross-encoder improves top-k precision by modeling fine query-document interactions.

### Why Gemini-based reranking/evaluation?
- Enables semantic judgments without building separate supervised reranker/judge datasets.
- Tradeoff: API latency/cost and non-determinism.

### Why multimodal extension?
- Movie discovery often starts from visual cues (poster/frame/mood).
- CLIP-style shared space enables image-to-text retrieval.

## 7. Comparison Tables

### 7.1 Keyword vs Semantic Search

| Feature | Keyword (BM25) | Semantic (Embeddings) |
|---|---|---|
| Matching basis | token overlap | vector similarity |
| Synonym handling | weak | strong |
| Exact entity matching | strong | moderate |
| Explainability | high | medium/low |
| Latency profile | low (after index) | medium (embedding + scan) |
| Typical failure mode | wording mismatch | semantic drift / missed exact token |
| Best use case | precise known-item queries | natural language intent queries |

### 7.2 Weighted Hybrid vs RRF

| Feature | Weighted Fusion | RRF |
|---|---|---|
| Input | normalized scores | ranks |
| Calibration sensitivity | high | low |
| Interpretability | medium | high (rank-based) |
| Robustness across retrievers | medium | high |
| Parameterization | `alpha` | `k` |
| In this repo | present but currently buggy call path | primary production path |

### 7.3 Cross-Encoder vs Bi-Encoder

| Feature | Bi-Encoder | Cross-Encoder |
|---|---|---|
| Encode docs offline | yes | no |
| Retrieval speed | fast | slower |
| Pairwise interaction modeling | no | yes |
| Top-k precision | moderate | higher |
| Role here | first-pass retrieval | reranker |

### 7.4 Manual vs LLM Evaluation

| Feature | Manual Labels | LLM Judge |
|---|---|---|
| Consistency | depends on rubric/humans | prompt/model dependent |
| Cost at scale | high | lower marginal |
| Transparency | high | medium |
| Bias risk | annotator bias | model bias/prompt bias |
| In this repo | golden set relevance lists | optional 0-3 CLI evaluation |

## 8. Evaluation Methodology in This Repository

## 8.1 Golden-set metric evaluation

- Ground truth source: `data/golden_dataset.json`
- Query set size: 10
- For each query:
  - Run RRF hybrid retrieval
  - Compare top-k titles to list of relevant titles
  - Print precision@k, recall@k, F1@k

Strengths:
- deterministic metric computation
- easy baseline comparison

Limitations:
- small benchmark size
- exact title matching only
- no graded relevance beyond binary relevant/not relevant

## 8.2 LLM-as-a-judge evaluation

- Optional from hybrid CLI: `--evaluate`
- Judge rubric is 0..3 relevance scale
- Parsing enforces strict integer JSON output

Strengths:
- captures nuanced semantic relevance
- faster than full human annotation loop

Risks and bias:
- model may be inconsistent across runs
- prompt wording can alter scores
- possible over-leniency or over-strictness by domain/query type

Recommendation:
- Use LLM judging as supplemental signal, not sole acceptance criterion.

## 9. Multimodal Pipeline Documentation

Implemented multimodal capabilities are twofold:

1. Image-to-movie retrieval (`cli/multimodal_search_cli.py image_search`)
- Build CLIP text embeddings for all movie documents
- Encode query image
- Compute image-text cosine similarity
- Return top matches

2. Image-conditioned query rewriting (`cli/describe_image_cli.py`)
- Send image + text query to Gemini
- Return rewritten query for downstream text retrieval

Difference vs pure text pipeline:
- Text-only path encodes only language with MiniLM/lexical methods.
- Multimodal path uses CLIP shared embedding space and can use visual evidence directly.

## 10. Augmented Generation Documentation

### What augmented generation means here
- Retrieval and generation are separate stages.
- Retrieval finds candidate supporting documents.
- Generation synthesizes final answer from retrieved context.

### Implemented flow
- Retrieval: `rrf_search` JSON results via subprocess
- Generation: Gemini prompt with serialized results
- Modes: answer/summarize/citations/conversational QA

### Prompt construction
- Prompts pass query/question + raw docs list as context.
- The system asks for concise, user-facing output tailored to Hoopla users.

### Context-window considerations
- Current implementation does not explicitly trim by token budget.
- Risk grows with larger `--limit` and long descriptions.

### Hallucination mitigation status
Implemented:
- retrieval grounding by passing documents
- citation-mode prompting

Not implemented:
- hard citation verification
- abstention classifier beyond prompt instructions
- source-faithfulness checker

## 11. Implementation Reference Map (File-Level)

Core retrieval:
- `cli/lib/keyword_search.py`: inverted index, TF/IDF/BM25, cache IO
- `utils/keyword_seach_utils/text_preprocessing.py`: lexical preprocessing (stopwords + Porter stemming)
- `utils/keyword_seach_utils/search_utils.py`: BM25 constants (`k1`, `b`)
- `cli/lib/semantic_search.py`: sentence-transformer retrieval + chunked retrieval
- `utils/semantic_search_utils/semantic_chunk.py`: sentence-window chunking
- `utils/semantic_search_utils/vector_operations.py`: cosine similarity and vector ops
- `utils/semantic_search_utils/text_preprocessing.py`: normalization for embedding inputs

Hybrid and fusion:
- `cli/lib/hybrid_search.py`: orchestrates BM25 + chunk-semantic and fusion
- `utils/hybrid_search_utils/score_utils.py`: weighted + RRF formula helpers
- `utils/hybrid_search_utils/normalize_score.py`: min-max normalization

Rerank and query enhancement:
- `utils/hybrid_search_utils/query_enhancement.py`: spell/rewrite/expand via Gemini
- `utils/hybrid_search_utils/rerank_methods.py`: Gemini rerank and cross-encoder rerank
- `utils/hybrid_search_utils/print.py`: result formatting
- `utils/hybrid_search_utils/debugger.py`: debug logging helpers
- `utils/hybrid_search_utils/evaluate.py`: LLM relevance scoring helper

Evaluation and generation CLIs:
- `cli/evaluation_cli.py`: P/R/F1 evaluation against golden set
- `cli/hybrid_search_cli.py`: main hybrid CLI with enhancement/rerank/eval flags
- `cli/augmented_generation_cli.py`: retrieval + generation workflows
- `utils/augmented_utils/rrf_search.py`: robust subprocess JSON parse wrapper
- `utils/augmented_utils/gemini.py`: generation/summarization/citation prompts

Multimodal:
- `cli/lib/multimodal_search.py`: CLIP image-text retrieval
- `cli/multimodal_search_cli.py`: multimodal CLI entrypoint
- `utils/image_cli_utils/gemini.py`: image-conditioned query rewrite
- `cli/describe_image_cli.py`: command wrapper for image rewrite

Data/loading:
- `utils/cli_utils/file_loading.py`: dataset loading (`movies.json`)
- `data/movies.json`: corpus
- `data/golden_dataset.json`: evaluation cases
- `data/stopwords.txt`: keyword preprocessing stopwords

## 12. Known Limitations and Current Gaps

These are code-level issues visible in the current repository state:

- `cli/keyword_search_cli.py` imports `utils.keyword_seach_utils.load_utils`, but this module is missing.
- `utils/keyword_seach_utils/tfidf_utils.py` calls `InvertedIndex.load()` without required `documents` argument.
- `cli/hybrid_search_cli.py` has `normalize` command path calling `normalize_score` without importing it.
- `cli/lib/hybrid_search.py::weighted_search` calls `hybrid_score` without passing `alpha` argument.
- Semantic retrieval assumes mapping by `i+1` in `SemanticSearch.search`, which can break if document IDs are not contiguous from 1.
- No ANN/vector DB support; dense retrieval is linear scan and may not scale for large corpora.
- No distributed indexing or sharding.
- No online learning from clicks/feedback.
- RAG context assembly is naive (full serialized docs, no token-budget optimizer).
- LLM-dependent modules require network/API key and are non-deterministic.

## 13. Scalability, Latency, and Memory Tradeoffs

- Sparse BM25 index in Python dict/list is memory-heavy but straightforward.
- Dense retrieval stores full embeddings in memory/NumPy arrays; query-time complexity is O(N*d).
- Chunked semantic search improves recall granularity but increases index size and compute.
- Cross-encoder reranking improves precision but increases latency roughly O(k) model inferences.
- Gemini-based enhancement/rerank/eval adds network latency and cost per request.

## 14. Recommended Future Improvements

1. Replace brute-force dense search with ANN (FAISS/HNSW) and benchmark latency/recall tradeoff.
2. Add a vector store abstraction for scalable persistence and filtering.
3. Fix known CLI/module inconsistencies listed above.
4. Add calibrated weighted fusion or learning-to-rank over BM25 + dense + metadata features.
5. Add deterministic reranker evaluation set with NDCG/MRR/Recall@k tracking.
6. Add prompt-injection and hallucination safeguards in generation layer.
7. Add token-aware context builder for RAG (dedupe, compression, citation-linked spans).
8. Add feedback loop from user interactions for continuous relevance tuning.
9. Add domain adaptation options (fine-tuned embeddings or rerankers).
10. Add distributed indexing and asynchronous batch embedding pipeline.

## 15. CLI Entry Points Summary

- Keyword: `cli/keyword_search_cli.py`
- Semantic: `cli/semantic_search_cli.py`
- Hybrid: `cli/hybrid_search_cli.py`
- Evaluation: `cli/evaluation_cli.py`
- Multimodal: `cli/multimodal_search_cli.py`
- Image-assisted rewrite: `cli/describe_image_cli.py`
- Augmented generation: `cli/augmented_generation_cli.py`

## 16. Environment and Dependencies

Defined in `pyproject.toml`:
- `sentence-transformers`
- `numpy`
- `nltk`
- `google-genai`
- `python-dotenv`
- `pillow`

Python requirement:
- `>=3.13`

For Gemini features (`query_enhancement`, rerank with Gemini, LLM evaluation, RAG generation, image rewrite):
- set `GEMINI_API_KEY` in environment or `.env`

---

If you are extending this system, start in `cli/lib/hybrid_search.py` and `utils/hybrid_search_utils/rerank_methods.py`: these are the main orchestration points where retrieval quality/latency tradeoffs are controlled.
