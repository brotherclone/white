"""
Corpus Embedder: RAG infrastructure for gabe_corpus.md

Chunks corpus by markdown headers, generates embeddings with caching,
and retrieves relevant passages for interview simulation.

Supports two modes:
1. Semantic embeddings (requires sentence-transformers + PyTorch)
2. Keyword-based TF-IDF fallback (works everywhere)
"""

import hashlib
import json
import logging
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import List, Tuple, Optional, Dict

logger = logging.getLogger(__name__)

# Check for numpy availability (optional for TF-IDF mode)
try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# Lazy import to avoid loading sentence-transformers until needed
_model = None
_model_name = "all-MiniLM-L6-v2"
_embedding_available = None


def _check_embedding_available() -> bool:
    """Check if sentence-transformers is available."""
    global _embedding_available
    if _embedding_available is None:
        try:
            from sentence_transformers import SentenceTransformer

            _embedding_available = True
            logger.info(
                f"sentence-transformers available: {_model_name}, {SentenceTransformer.__version__}"
            )
        except ImportError:
            _embedding_available = False
            logger.info(
                "sentence-transformers not available, using TF-IDF fallback. "
                "For semantic search, install: pip install sentence-transformers"
            )
    return _embedding_available


def _get_model():
    """Lazy load the sentence transformer model."""
    global _model
    if _model is None:
        if not _check_embedding_available():
            raise ImportError("sentence-transformers not available")
        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading embedding model: {_model_name}")
        _model = SentenceTransformer(_model_name)
    return _model


# =============================================================================
# TF-IDF Fallback Implementation (no external dependencies)
# =============================================================================


def _tokenize(text: str) -> List[str]:
    """Simple tokenization: lowercase, alphanumeric only."""
    return re.findall(r"\b[a-z]+\b", text.lower())


def _compute_tf(tokens: List[str]) -> Dict[str, float]:
    """Compute term frequency."""
    counts = Counter(tokens)
    total = len(tokens)
    return {term: count / total for term, count in counts.items()}


def _compute_idf(documents: List[List[str]]) -> Dict[str, float]:
    """Compute inverse document frequency."""
    n_docs = len(documents)
    doc_freq = Counter()
    for doc in documents:
        doc_freq.update(set(doc))
    return {term: math.log(n_docs / (1 + freq)) for term, freq in doc_freq.items()}


def _tfidf_similarity(
    query_tokens: List[str], doc_tokens: List[str], idf: Dict[str, float]
) -> float:
    """Compute TF-IDF cosine similarity between query and document."""
    query_tf = _compute_tf(query_tokens)
    doc_tf = _compute_tf(doc_tokens)

    # Get all terms
    all_terms = set(query_tf.keys()) | set(doc_tf.keys())

    # Compute TF-IDF vectors
    query_vec = {t: query_tf.get(t, 0) * idf.get(t, 0) for t in all_terms}
    doc_vec = {t: doc_tf.get(t, 0) * idf.get(t, 0) for t in all_terms}

    # Cosine similarity
    dot = sum(query_vec[t] * doc_vec[t] for t in all_terms)
    query_norm = math.sqrt(sum(v**2 for v in query_vec.values()))
    doc_norm = math.sqrt(sum(v**2 for v in doc_vec.values()))

    if query_norm == 0 or doc_norm == 0:
        return 0.0
    return dot / (query_norm * doc_norm)


def chunk_corpus(corpus_text: str) -> List[Tuple[str, str]]:
    """
    Split corpus into chunks by markdown headers.

    Returns list of (header, content) tuples.
    Chunks at ## level, preserving ### subsections within parent chunk.
    """
    chunks = []

    # Split by ## headers (Part headers)
    parts = re.split(r"\n(?=## )", corpus_text)

    for part in parts:
        if not part.strip():
            continue

        # Extract header
        lines = part.strip().split("\n")
        header = lines[0].strip()
        content = "\n".join(lines[1:]).strip()

        if header.startswith("## ") and content:
            chunks.append((header, content))

    return chunks


def chunk_corpus_fine(corpus_text: str) -> List[Tuple[str, str]]:
    """
    Fine-grained chunking - splits at ### level too.

    Use this for more granular retrieval when corpus is large.
    """
    chunks = []

    # First split by ## headers
    parts = re.split(r"\n(?=## )", corpus_text)

    for part in parts:
        if not part.strip():
            continue

        lines = part.strip().split("\n")
        part_header = lines[0].strip()

        if not part_header.startswith("## "):
            continue

        # Check if this part has ### subsections
        subsections = re.split(r"\n(?=### )", part)

        if len(subsections) > 1:
            # Has subsections - chunk each one
            for i, sub in enumerate(subsections):
                sub_lines = sub.strip().split("\n")
                if i == 0:
                    # First chunk is the part header + intro content
                    header = part_header
                    content = "\n".join(sub_lines[1:]).strip()
                else:
                    # Subsection
                    sub_header = sub_lines[0].strip()
                    header = f"{part_header} > {sub_header}"
                    content = "\n".join(sub_lines[1:]).strip()

                if content:
                    chunks.append((header, content))
        else:
            # No subsections - use whole part
            content = "\n".join(lines[1:]).strip()
            if content:
                chunks.append((part_header, content))

    return chunks


def _compute_corpus_hash(corpus_text: str) -> str:
    """Compute hash of corpus for cache invalidation."""
    return hashlib.md5(corpus_text.encode()).hexdigest()[:12]


def _get_cache_path(corpus_path: Path) -> Path:
    """Get path to embeddings cache file."""
    return corpus_path.parent / f"{corpus_path.stem}_embeddings.json"


def _load_cached_embeddings(
    cache_path: Path, corpus_hash: str
) -> Optional[Tuple[List[Tuple[str, str]], np.ndarray]]:
    """Load embeddings from cache if valid."""
    if not cache_path.exists():
        return None

    try:
        with open(cache_path, "r") as f:
            data = json.load(f)

        if data.get("corpus_hash") != corpus_hash:
            logger.info("Corpus changed, cache invalidated")
            return None

        if data.get("model") != _model_name:
            logger.info("Model changed, cache invalidated")
            return None

        chunks = [(c["header"], c["content"]) for c in data["chunks"]]
        embeddings = np.array(data["embeddings"])

        logger.info(f"Loaded {len(chunks)} cached embeddings")
        return chunks, embeddings

    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Cache invalid: {e}")
        return None


def _save_embeddings_cache(
    cache_path: Path,
    corpus_hash: str,
    chunks: List[Tuple[str, str]],
    embeddings: np.ndarray,
):
    """Save embeddings to cache."""
    data = {
        "corpus_hash": corpus_hash,
        "model": _model_name,
        "chunks": [{"header": h, "content": c} for h, c in chunks],
        "embeddings": embeddings.tolist(),
    }

    with open(cache_path, "w") as f:
        json.dump(data, f)

    logger.info(f"Cached {len(chunks)} embeddings to {cache_path}")


class CorpusEmbedder:
    """
    RAG infrastructure for gabe_corpus.md.

    Usage:
        embedder = CorpusEmbedder(corpus_path)
        chunks = embedder.retrieve("question about creative process", top_k=5)

    Supports two retrieval modes:
    - Semantic (sentence-transformers): Better quality, requires PyTorch
    - TF-IDF (built-in): Works everywhere, keyword-based matching
    """

    def __init__(
        self,
        corpus_path: Path,
        fine_grained: bool = False,
        use_cache: bool = True,
        force_tfidf: bool = False,
    ):
        """
        Initialize embedder with corpus.

        Args:
            corpus_path: Path to corpus markdown file
            fine_grained: If True, chunk at ### level too
            use_cache: If True, cache embeddings to disk
            force_tfidf: If True, use TF-IDF even if embeddings available
        """
        self.corpus_path = Path(corpus_path)
        self.fine_grained = fine_grained
        self.use_cache = use_cache
        self.force_tfidf = force_tfidf

        self.chunks: List[Tuple[str, str]] = []
        self.embeddings = None  # numpy array if available
        self._chunk_tokens: List[List[str]] = []  # for TF-IDF
        self._idf: Dict[str, float] = {}  # for TF-IDF
        self._use_embeddings = False
        self._initialized = False

    def _ensure_initialized(self):
        """Lazy initialization - load/generate embeddings on first use."""
        if self._initialized:
            return

        if not self.corpus_path.exists():
            logger.warning(f"Corpus not found: {self.corpus_path}")
            self._initialized = True
            return

        # Load corpus
        with open(self.corpus_path, "r") as f:
            corpus_text = f.read()

        corpus_hash = _compute_corpus_hash(corpus_text)
        cache_path = _get_cache_path(self.corpus_path)

        # Determine if we can use embeddings
        can_use_embeddings = (
            not self.force_tfidf and _check_embedding_available() and HAS_NUMPY
        )

        # Try cache first (only for embedding mode)
        if can_use_embeddings and self.use_cache:
            cached = _load_cached_embeddings(cache_path, corpus_hash)
            if cached:
                self.chunks, self.embeddings = cached
                self._use_embeddings = True
                self._initialized = True
                return

        # Chunk corpus
        if self.fine_grained:
            self.chunks = chunk_corpus_fine(corpus_text)
        else:
            self.chunks = chunk_corpus(corpus_text)

        logger.info(f"Chunked corpus into {len(self.chunks)} segments")

        # Try semantic embeddings first
        if can_use_embeddings:
            try:

                model = _get_model()
                texts = [f"{header}\n{content}" for header, content in self.chunks]
                self.embeddings = model.encode(texts, convert_to_numpy=True)
                self._use_embeddings = True

                # Cache
                if self.use_cache:
                    _save_embeddings_cache(
                        cache_path, corpus_hash, self.chunks, self.embeddings
                    )

                logger.info("Using semantic embeddings for retrieval")
                self._initialized = True
                return

            except Exception as e:
                logger.warning(
                    f"Embedding generation failed: {e}, falling back to TF-IDF"
                )

        # Fallback: TF-IDF
        self._setup_tfidf()
        self._initialized = True

    def _setup_tfidf(self):
        """Initialize TF-IDF structures for keyword-based retrieval."""
        logger.info("Using TF-IDF fallback for retrieval")
        self._chunk_tokens = []
        for header, content in self.chunks:
            text = f"{header} {content}"
            tokens = _tokenize(text)
            self._chunk_tokens.append(tokens)

        self._idf = _compute_idf(self._chunk_tokens)
        self._use_embeddings = False

    def retrieve(
        self, query: str, top_k: int = 5, threshold: float = 0.0
    ) -> List[Tuple[str, str, float]]:
        """
        Retrieve most relevant corpus chunks for a query.

        Args:
            query: The search query (e.g., interview question)
            top_k: Number of chunks to return
            threshold: Minimum similarity score (0-1)

        Returns:
            List of (header, content, similarity_score) tuples
        """
        self._ensure_initialized()

        if not self.chunks:
            logger.warning("No chunks available")
            return []

        if self._use_embeddings:
            return self._retrieve_semantic(query, top_k, threshold)
        else:
            return self._retrieve_tfidf(query, top_k, threshold)

    def _retrieve_semantic(
        self, query: str, top_k: int, threshold: float
    ) -> List[Tuple[str, str, float]]:
        """Retrieve using semantic embeddings."""
        import numpy as np

        model = _get_model()
        query_embedding = model.encode([query], convert_to_numpy=True)[0]

        # Cosine similarity
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score >= threshold:
                header, content = self.chunks[idx]
                results.append((header, content, score))

        return results

    def _retrieve_tfidf(
        self, query: str, top_k: int, threshold: float
    ) -> List[Tuple[str, str, float]]:
        """Retrieve using TF-IDF keyword matching."""
        query_tokens = _tokenize(query)

        if not query_tokens:
            return []

        # Score all chunks
        scores = []
        for idx, doc_tokens in enumerate(self._chunk_tokens):
            score = _tfidf_similarity(query_tokens, doc_tokens, self._idf)
            scores.append((idx, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in scores[:top_k]:
            if score >= threshold:
                header, content = self.chunks[idx]
                results.append((header, content, score))

        return results

    def retrieve_formatted(
        self, query: str, top_k: int = 5, max_tokens: int = 3000
    ) -> str:
        """
        Retrieve and format chunks for prompt injection.

        Args:
            query: The search query
            top_k: Number of chunks to retrieve
            max_tokens: Approximate max characters (rough token estimate)

        Returns:
            Formatted string ready for prompt injection
        """
        results = self.retrieve(query, top_k=top_k)

        if not results:
            return ""

        parts = []
        total_chars = 0

        for header, content, score in results:
            chunk_text = f"[{header}]\n{content}"

            if total_chars + len(chunk_text) > max_tokens * 4:  # ~4 chars per token
                # Truncate this chunk if needed
                remaining = (max_tokens * 4) - total_chars
                if remaining > 200:
                    chunk_text = chunk_text[:remaining] + "..."
                    parts.append(chunk_text)
                break

            parts.append(chunk_text)
            total_chars += len(chunk_text)

        return "\n\n".join(parts)

    def get_all_chunks(self) -> List[Tuple[str, str]]:
        """Return all corpus chunks (for debugging/inspection)."""
        self._ensure_initialized()
        return self.chunks.copy()


# Convenience function for one-off retrieval
def retrieve_from_corpus(
    corpus_path: Path, query: str, top_k: int = 5
) -> List[Tuple[str, str, float]]:
    """
    One-shot retrieval from corpus.

    For repeated queries, use CorpusEmbedder class directly.
    """
    embedder = CorpusEmbedder(corpus_path)
    return embedder.retrieve(query, top_k=top_k)


if __name__ == "__main__":
    # Test the embedder

    corpus_path = Path(
        os.getenv("GABE_CORPUS_FILE", "app/reference/biographical/gabe_corpus.md")
    )

    print(f"Testing corpus embedder with: {corpus_path}")

    embedder = CorpusEmbedder(corpus_path, fine_grained=True)

    test_queries = [
        "What drives your creative process?",
        "Tell me about your relationship with technology",
        "How do you handle self-doubt?",
        "What's your sense of humor like?",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print("=" * 60)

        results = embedder.retrieve(query, top_k=3)
        for header, content, score in results:
            print(f"\n[{score:.3f}] {header}")
            print(content[:200] + "..." if len(content) > 200 else content)
