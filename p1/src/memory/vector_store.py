"""Lightweight indexing memory vector store for P1 experience reuse architecture."""

from typing import List, Dict, Any, Optional
import numpy as np


class SimpleVectorStore:
    """Minimal vector store supporting semantic cosine similarity retrieval.

    Supports sentence-transformers if installed, and seamlessly falls back to
    TF-IDF with numpy cosine similarity for zero-dependency operation.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the vector store.

        Args:
            model_name: Name of the sentence-transformers model if available.
        """
        self.model_name = model_name
        self.records: List[Dict[str, Any]] = []
        self._st_model = None
        self._st_embeddings: List[np.ndarray] = []

        # Attempt to load sentence-transformers if installed
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._st_model = SentenceTransformer(model_name)
        except Exception:
            self._st_model = None

    def add_record(
        self,
        memory_id: str,
        source_problem_id: str,
        problem_text: str,
        solution_text: str,
    ) -> None:
        """Vectorize problem text and store the record.

        Args:
            memory_id: Unique identifier for the memory entry.
            source_problem_id: ID of the source problem.
            problem_text: The problem description or prompt text.
            solution_text: The complete reference solution/reasoning steps.
        """
        record = {
            "memory_id": memory_id,
            "source_problem_id": source_problem_id,
            "problem_text": problem_text,
            "solution_text": solution_text,
        }
        self.records.append(record)

        if self._st_model is not None:
            emb = self._st_model.encode(problem_text, convert_to_numpy=True)
            self._st_embeddings.append(np.array(emb, dtype=np.float32))

    def retrieve(self, query_text: str, top_k: int = 1) -> List[Dict[str, Any]]:
        """Retrieve the top-k most similar records using cosine similarity.

        Args:
            query_text: The query problem text.
            top_k: Number of most similar items to return.

        Returns:
            List of dictionaries containing memory records with similarity scores.
        """
        if not self.records or top_k <= 0:
            return []

        if self._st_model is not None and len(self._st_embeddings) == len(self.records):
            similarities = self._retrieve_with_st(query_text)
        else:
            similarities = self._retrieve_with_tfidf(query_text)

        ranked_indices = np.argsort(-similarities)
        k = min(top_k, len(self.records))
        top_indices = ranked_indices[:k]

        results = []
        for idx in top_indices:
            item = dict(self.records[idx])
            item["similarity"] = float(similarities[idx])
            results.append(item)
        return results

    def _retrieve_with_st(self, query_text: str) -> np.ndarray:
        """Compute cosine similarity using sentence-transformers embeddings."""
        query_emb = self._st_model.encode(query_text, convert_to_numpy=True)
        doc_matrix = np.stack(self._st_embeddings, axis=0)
        return self._cosine_similarity(query_emb, doc_matrix)

    def _retrieve_with_tfidf(self, query_text: str) -> np.ndarray:
        """Compute cosine similarity using TF-IDF vectorizer fallback."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            corpus = [r["problem_text"] for r in self.records]
            vectorizer = TfidfVectorizer(
                token_pattern=r"(?u)\b\w+\b",
                stop_words="english",
                ngram_range=(1, 2),
            )
            doc_matrix = vectorizer.fit_transform(corpus).toarray()
            query_vec = vectorizer.transform([query_text]).toarray()[0]
            return self._cosine_similarity(query_vec, doc_matrix)
        except Exception:
            # Mock / fallback when no ML libraries are installed
            return self._mock_similarity(query_text)

    def _mock_similarity(self, query_text: str) -> np.ndarray:
        """Simple word-set Jaccard / token overlap fallback."""
        query_tokens = set(query_text.lower().split())
        scores = []
        for r in self.records:
            doc_tokens = set(r["problem_text"].lower().split())
            if not query_tokens or not doc_tokens:
                scores.append(0.0)
            else:
                sim = len(query_tokens & doc_tokens) / float(len(query_tokens | doc_tokens))
                scores.append(sim)
        return np.array(scores, dtype=np.float32)

    @staticmethod
    def _cosine_similarity(query_vec: np.ndarray, doc_matrix: np.ndarray) -> np.ndarray:
        """Compute cosine similarities between a 1D query vector and 2D doc matrix."""
        q_norm = np.linalg.norm(query_vec)
        d_norms = np.linalg.norm(doc_matrix, axis=1)
        if q_norm == 0:
            return np.zeros(doc_matrix.shape[0], dtype=np.float32)

        denom = np.maximum(d_norms * q_norm, 1e-12)
        dots = np.dot(doc_matrix, query_vec)
        similarities = dots / denom
        return np.clip(similarities, -1.0, 1.0).astype(np.float32)
