"""
RAG (Retrieval-Augmented Generation) Engine for LumosCareer.

Implements a from-scratch TF-IDF vectorizer + cosine similarity retriever
over the local knowledge base. No external vector DB dependency.
"""

import numpy as np
import re
import math
from collections import Counter


class RAGEngine:
    """
    Retrieval-Augmented Generation engine.
    Indexes all knowledge base documents using TF-IDF vectors,
    then retrieves the most relevant documents for any query via cosine similarity.
    """

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if RAGEngine._initialized:
            return
        self.documents = []
        self.doc_keys = []
        self.vocabulary = {}
        self.idf = {}
        self.tfidf_matrix = None
        self._build_index()
        RAGEngine._initialized = True

    def _tokenize(self, text):
        text = text.lower()
        tokens = re.findall(r'\b[a-z][a-z0-9+#.]{1,25}\b', text)
        stop_words = {
            'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but',
            'in', 'with', 'to', 'for', 'of', 'not', 'no', 'can', 'had', 'has',
            'have', 'it', 'its', 'are', 'was', 'were', 'be', 'been', 'being',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may',
            'might', 'shall', 'this', 'that', 'these', 'those', 'am', 'from',
            'as', 'by', 'he', 'she', 'they', 'we', 'you', 'me', 'him', 'her',
            'us', 'them', 'my', 'your', 'his', 'our', 'their', 'what', 'so',
            'if', 'about', 'up', 'out', 'then', 'than', 'too', 'very', 'just',
            'also', 'over', 'such', 'how', 'when', 'where', 'why', 'all',
            'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some',
            'any', 'only', 'own', 'same', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'between', 'under', 'again', 'further',
        }
        return [t for t in tokens if t not in stop_words]

    def _build_index(self):
        from .knowledge_base import KNOWLEDGE_BASE

        self.documents = []
        self.doc_keys = []

        for key, content in KNOWLEDGE_BASE.items():
            if isinstance(content, dict):
                text = f"{key}. "
                for sub_key, sub_val in content.items():
                    if isinstance(sub_val, str):
                        text += f"{sub_key}: {sub_val}. "
                    elif isinstance(sub_val, list):
                        text += f"{sub_key}: {', '.join(str(x) for x in sub_val)}. "
                    elif isinstance(sub_val, dict):
                        for k3, v3 in sub_val.items():
                            text += f"{k3}: {v3}. "
                self.documents.append(text)
                self.doc_keys.append(key)
            elif isinstance(content, str):
                self.documents.append(f"{key}. {content}")
                self.doc_keys.append(key)
            elif isinstance(content, list):
                text = f"{key}. " + " ".join(str(x) for x in content)
                self.documents.append(text)
                self.doc_keys.append(key)

        if not self.documents:
            return

        # Build vocabulary
        doc_tokens = [self._tokenize(doc) for doc in self.documents]
        all_tokens = set()
        for tokens in doc_tokens:
            all_tokens.update(tokens)
        self.vocabulary = {word: idx for idx, word in enumerate(sorted(all_tokens))}

        n_docs = len(self.documents)
        n_vocab = len(self.vocabulary)

        if n_vocab == 0:
            return

        # Compute TF
        tf_matrix = np.zeros((n_docs, n_vocab))
        for i, tokens in enumerate(doc_tokens):
            if not tokens:
                continue
            counts = Counter(tokens)
            for word, count in counts.items():
                if word in self.vocabulary:
                    tf_matrix[i, self.vocabulary[word]] = count / len(tokens)

        # Compute IDF
        df = np.sum(tf_matrix > 0, axis=0)
        idf_vector = np.log((n_docs + 1) / (df + 1)) + 1

        # TF-IDF matrix
        self.tfidf_matrix = tf_matrix * idf_vector
        self.idf_vector = idf_vector

        # Normalize rows for faster cosine similarity
        norms = np.linalg.norm(self.tfidf_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.tfidf_normed = self.tfidf_matrix / norms

    def _vectorize_query(self, query):
        tokens = self._tokenize(query)
        if not tokens or not self.vocabulary:
            return None

        vec = np.zeros(len(self.vocabulary))
        counts = Counter(tokens)
        for word, count in counts.items():
            if word in self.vocabulary:
                vec[self.vocabulary[word]] = (count / len(tokens)) * self.idf_vector[self.vocabulary[word]]

        norm = np.linalg.norm(vec)
        if norm == 0:
            return None
        return vec / norm

    def retrieve(self, query, top_k=5):
        """
        Retrieve the top-k most relevant knowledge base entries for the given query.
        Returns list of (key, text, score) tuples.
        """
        if self.tfidf_matrix is None or len(self.documents) == 0:
            return []

        q_vec = self._vectorize_query(query)
        if q_vec is None:
            return []

        similarities = np.dot(self.tfidf_normed, q_vec)
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score > 0.01:
                results.append((self.doc_keys[idx], self.documents[idx], score))
        return results

    def build_context(self, query, max_chars=3000):
        """
        Retrieve relevant documents and assemble into a context string
        suitable for injection into LLM prompts.
        """
        results = self.retrieve(query, top_k=7)
        if not results:
            return ""

        context_parts = []
        total_chars = 0

        for key, text, score in results:
            snippet = text[:600]
            if total_chars + len(snippet) > max_chars:
                break
            context_parts.append(f"[{key}]: {snippet}")
            total_chars += len(snippet)

        if not context_parts:
            return ""

        return "KNOWLEDGE BASE CONTEXT:\n" + "\n---\n".join(context_parts)


def get_rag_engine():
    """Factory function to get/create the singleton RAG engine."""
    return RAGEngine()
