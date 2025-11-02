"""
Simple vector store demo with a TF-IDF / cosine similarity fallback.
If scikit-learn is available this uses TfidfVectorizer for embeddings; otherwise
it falls back to a naive substring scoring approach.

This is intended as a small, self-contained example to demonstrate retrieval
for RAG-style prompts. For production use, replace with FAISS/Chroma/Pinecone.
"""
from typing import List, Dict, Any, Optional

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel
except Exception:
    TfidfVectorizer = None
    linear_kernel = None


class SimpleVectorStore:
    def __init__(self):
        self.texts: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        self._vectorizer = None
        self._matrix = None

    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None):
        if metadatas is None:
            metadatas = [{} for _ in texts]
        self.texts.extend(texts)
        self.metadatas.extend(metadatas)
        # rebuild index
        self._build_index()

    def _build_index(self):
        if TfidfVectorizer is not None:
            try:
                self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
                self._matrix = self._vectorizer.fit_transform(self.texts)
            except Exception:
                self._vectorizer = None
                self._matrix = None
        else:
            self._vectorizer = None
            self._matrix = None

    def similarity_search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Return top-k documents similar to query as dicts {text, metadata, score}"""
        results = []
        if self._matrix is not None and self._vectorizer is not None and linear_kernel is not None:
            qv = self._vectorizer.transform([query])
            sims = linear_kernel(qv, self._matrix).flatten()
            idxs = sims.argsort()[::-1][:k]
            for i in idxs:
                results.append({"text": self.texts[i], "metadata": self.metadatas[i], "score": float(sims[i])})
            return results

        # Fallback: simple substring scoring
        query_lower = query.lower()
        scored = []
        for i, t in enumerate(self.texts):
            score = 0.0
            tl = t.lower()
            if query_lower in tl:
                score += 2.0
            # count overlapping token matches
            for token in query_lower.split():
                if token and token in tl:
                    score += 0.1
            scored.append((score, i))
        scored.sort(reverse=True)
        for score, i in scored[:k]:
            results.append({"text": self.texts[i], "metadata": self.metadatas[i], "score": float(score)})
        return results


def build_default_templates() -> List[Dict[str, Any]]:
    """Return some small insurer-specific templates as examples for retrieval.
    In a real system these would be many examples and templates per insurer.
    """
    templates = [
        {
            "text": "Fortis Hospitals ID template: Patient name {name} Policy Number {policy}",
            "metadata": {"insurer": "Fortis", "type": "id_template"}
        },
        {
            "text": "SBI General ID template: Name: {name}\nPolicy Number: {policy}",
            "metadata": {"insurer": "SBI", "type": "id_template"}
        },
        {
            "text": "Generic Insurance ID card often labels 'Patient name' and 'Policy Number' on the top-left.",
            "metadata": {"insurer": "generic", "type": "id_template"}
        }
    ]
    return templates
