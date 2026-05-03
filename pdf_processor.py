"""
pdf_processor.py
Handles PDF loading, text chunking, embedding generation,
and FAISS vector-store indexing.
"""

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import logging
import re
from typing import List, Tuple, Optional, Dict
from collections import defaultdict

try:
    from rank_bm25 import BM25Okapi
except Exception:  # pragma: no cover - optional dependency
    BM25Okapi = None  # type: ignore[assignment]

try:
    from sentence_transformers import CrossEncoder
except Exception:  # pragma: no cover - optional dependency
    CrossEncoder = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level singletons — models are loaded once per process
# ---------------------------------------------------------------------------
_EMBEDDER_NAME = "intfloat/multilingual-e5-small"
_RERANKER_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
_embedder: Optional[SentenceTransformer] = None
_reranker = None


def _get_device() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        device = _get_device()
        logger.info("Loading embedding model: %s on %s", _EMBEDDER_NAME, device)
        _embedder = SentenceTransformer(_EMBEDDER_NAME, device=device)
    return _embedder


def _get_reranker():
    global _reranker
    if _reranker is None and CrossEncoder is not None:
        try:
            device = _get_device()
            logger.info("Loading reranker model: %s on %s", _RERANKER_NAME, device)
            _reranker = CrossEncoder(_RERANKER_NAME, device=device)
        except Exception as exc:
            logger.warning("Could not load reranker: %s", exc)
    return _reranker


class PDFProcessor:
    """
    End-to-end pipeline:
      PDF  →  pages  →  chunks  →  embeddings  →  FAISS index
    """

    def __init__(self):
        # Use module-level singletons — no reload on repeated uploads
        self.embedder = _get_embedder()
        self.reranker = _get_reranker()

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""],
        )

        # State populated after chunk_pdf()
        self.chunks: List[Tuple[str, int]] = []   # (text, page_number)
        self.raw_chunk_texts: List[str] = []
        self.chunk_texts: List[str] = []
        self.index: Optional[faiss.Index] = None
        self.bm25 = None
        self.tokenized_corpus: List[List[str]] = []
        self.pdf_name: str = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_pdf(self, pdf_path: str) -> int:
        """
        Load *pdf_path*, split into chunks, embed, and build FAISS index.
        Returns the number of chunks created.
        """
        self.chunks = []
        self.pdf_name = pdf_path

        pages = self._load_pdf(pdf_path)
        for text, page_num in pages:
            if not text or not text.strip():
                continue
            splits = self.text_splitter.split_text(text)
            for split in splits:
                if split.strip():
                    self.chunks.append((split, page_num))

        if not self.chunks:
            raise ValueError("No extractable text found in the PDF.")

        self.raw_chunk_texts = [c[0] for c in self.chunks]

        # E5 models are trained with explicit text prefixes.
        # Using "passage:" for corpus vectors improves retrieval quality.
        self.chunk_texts = [f"passage: {text}" for text in self.raw_chunk_texts]
        logger.info("Embedding %d chunks …", len(self.chunks))
        embeddings = self.embedder.encode(
            self.chunk_texts, show_progress_bar=True, normalize_embeddings=True
        )
        dimension = embeddings.shape[1]
        # Inner-product index (cosine similarity after L2 normalisation)
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings.astype(np.float32))

        # Sparse lexical index for hybrid search.
        self.tokenized_corpus = [self._tokenize(t) for t in self.raw_chunk_texts]
        if BM25Okapi is not None:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        else:
            self.bm25 = None
            logger.warning("rank_bm25 not available; hybrid retrieval will use dense-only.")
        logger.info("FAISS index built with %d vectors (dim=%d)", len(self.chunks), dimension)
        return len(self.chunks)

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        dense_weight: float = 0.75,
        rerank_weight: float = 0.35,
    ) -> List[Dict]:
        """
        Return top-k most relevant chunks for *query*.
        Each result is  {"text": str, "page": int, "score": float}.
        """
        if self.index is None or not self.chunks:
            return []

        variants = self._generate_query_variants(query, max_variants=4)
        candidate_scores: dict[int, dict[str, float]] = defaultdict(
            lambda: {"dense": 0.0, "bm25": 0.0}
        )

        dense_candidates = max(top_k * 4, 12)
        for variant in variants:
            # Dense retrieval
            query_emb = self.embedder.encode([f"query: {variant}"], normalize_embeddings=True)
            dense_scores, dense_indices = self.index.search(
                query_emb.astype(np.float32), dense_candidates
            )
            for score, idx in zip(dense_scores[0], dense_indices[0]):
                if idx == -1:
                    continue
                norm_dense = max(0.0, min(1.0, (float(score) + 1.0) / 2.0))
                candidate_scores[idx]["dense"] = max(candidate_scores[idx]["dense"], norm_dense)

            # Sparse retrieval (BM25)
            if self.bm25 is not None:
                tokens = self._tokenize(variant)
                if tokens:
                    bm25_scores = self.bm25.get_scores(tokens)
                    top_idx = np.argsort(bm25_scores)[-dense_candidates:]
                    max_bm25 = max((bm25_scores[i] for i in top_idx), default=0.0)
                    for idx in top_idx:
                        raw_bm25 = float(bm25_scores[idx])
                        norm_bm25 = (raw_bm25 / max_bm25) if max_bm25 > 0 else 0.0
                        candidate_scores[int(idx)]["bm25"] = max(
                            candidate_scores[int(idx)]["bm25"], norm_bm25
                        )

        if not candidate_scores:
            return []

        results = []
        sparse_weight = 1.0 - dense_weight if self.bm25 is not None else 0.0
        for idx, score_pack in candidate_scores.items():
            text, page = self.chunks[idx]
            combined = (dense_weight * score_pack["dense"]) + (sparse_weight * score_pack["bm25"])
            results.append(
                {
                    "text": text,
                    "page": page,
                    "score": float(combined),
                    "dense_score": float(score_pack["dense"]),
                    "bm25_score": float(score_pack["bm25"]),
                }
            )

        results.sort(key=lambda x: x["score"], reverse=True)
        results = self._rerank(query, results, top_k=top_k, rerank_weight=rerank_weight)
        return results[:top_k]

    def is_ready(self) -> bool:
        return self.index is not None and len(self.chunks) > 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_pdf(self, pdf_path: str) -> List[Tuple[str, int]]:
        """Extract (text, page_number) pairs from the PDF."""
        reader = PdfReader(pdf_path)
        pages = []
        for page_num, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception as exc:
                logger.warning("Could not extract page %d: %s", page_num, exc)
                text = ""
            pages.append((text, page_num))
        logger.info("Loaded PDF '%s': %d pages", pdf_path, len(pages))
        return pages

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[^\W_]+", text.lower())

    def _generate_query_variants(self, query: str, max_variants: int = 4) -> List[str]:
        cleaned = " ".join(query.strip().split())
        if not cleaned:
            return [query]

        variants = [cleaned]
        if "?" in cleaned:
            variants.append(cleaned.replace("?", ""))

        unique: List[str] = []
        seen = set()
        for v in variants:
            norm = v.lower().strip()
            if norm and norm not in seen:
                seen.add(norm)
                unique.append(v)
            if len(unique) >= max_variants:
                break
        return unique

    def _rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int,
        rerank_weight: float,
    ) -> List[Dict]:
        if not candidates or self.reranker is None:
            return candidates

        rerank_pool = candidates[: max(top_k * 3, 10)]
        pairs = [(query, item["text"]) for item in rerank_pool]
        try:
            rerank_scores = self.reranker.predict(pairs)
        except Exception as exc:  # pragma: no cover - inference issue fallback
            logger.warning("Reranker failed, using hybrid ranking only: %s", exc)
            return candidates

        for item, raw_score in zip(rerank_pool, rerank_scores):
            # Normalise unbounded logits to [0, 1]
            rerank_norm = 1.0 / (1.0 + np.exp(-float(raw_score)))
            item["rerank_score"] = float(rerank_norm)
            item["score"] = float((1.0 - rerank_weight) * item["score"] + rerank_weight * rerank_norm)

        rerank_pool.sort(key=lambda x: x["score"], reverse=True)
        remainder = candidates[len(rerank_pool) :]
        return rerank_pool + remainder
