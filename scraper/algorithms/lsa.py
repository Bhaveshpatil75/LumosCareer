import numpy as np
import re
from ..models import SkillNode, Company

class LSAEngine:
    def __init__(self):
        self.vocabulary = []
        self.tfidf_matrix = None
        self.corpus_docs = []
        self.build_corpus_from_db()

    def build_corpus_from_db(self):
        # Add basic skills
        skills = list(SkillNode.objects.values_list('name', flat=True))
        # Add company tech stacks
        companies = Company.objects.exclude(tech_stack='').values_list('tech_stack', flat=True)
        
        self.corpus_docs = skills + [c for c in companies]

    def preprocess(self, text):
        text = text.lower()
        return re.findall(r'\b\w+\b', text)

    def fit_transform(self, documents):
        # Combined corpus: DB data + new documents (query/resume)
        full_docs = self.corpus_docs + documents
        
        # 1. Build Vocabulary
        unique_words = set()
        processed_docs = [self.preprocess(doc) for doc in full_docs]
        for doc in processed_docs:
            unique_words.update(doc)
        self.vocabulary = sorted(list(unique_words))
        vocab_index = {word: i for i, word in enumerate(self.vocabulary)}

        # 2. TF
        n_docs = len(full_docs)
        n_vocab = len(self.vocabulary)
        
        # Memory check - for huge DBs this is bad, but for <1000 items it's fine
        if n_docs > 5000: # Safe guard
             n_docs = 5000
             processed_docs = processed_docs[:5000]

        tf_matrix = np.zeros((n_docs, n_vocab))

        for i, doc in enumerate(processed_docs):
            doc_len = len(doc)
            if doc_len == 0: continue
            for word in doc:
                if word in vocab_index:
                    tf_matrix[i, vocab_index[word]] += 1
            tf_matrix[i] = tf_matrix[i] / doc_len

        # 3. IDF
        df = np.sum(tf_matrix > 0, axis=0)
        idf = np.log((n_docs + 1) / (df + 1)) + 1
        
        self.tfidf_matrix = tf_matrix * idf

        # 4. SVD
        try:
            U, S, Vt = np.linalg.svd(self.tfidf_matrix, full_matrices=False)
            k = min(n_docs, n_vocab, 50) # Reduced dimensions
            self.U_k = U[:, :k]
            self.S_k = np.diag(S[:k])
            
            # Project
            return np.dot(self.U_k, self.S_k)
        except np.linalg.LinAlgError:
            return self.tfidf_matrix

    def compute_similarity(self, query, documents):
        # We need to project query and documents into the SAME space
        # Re-fitting every time is expensive but simplest for this architecture w/o persistence
        
        all_docs = [query] + documents
        lsa_matrix = self.fit_transform(all_docs)
        
        # The last len(documents)+1 rows correspond to our input (since we appended them to corpus)
        # Actually fit_transform uses self.corpus + documents.
        # So inputs are at the END.
        
        start_idx = len(self.corpus_docs)
        query_vec = lsa_matrix[start_idx]
        doc_vecs = lsa_matrix[start_idx+1 : ]
        
        similarities = []
        norm_query = np.linalg.norm(query_vec)
        
        if norm_query == 0:
            return 0, [], []

        for doc_vec in doc_vecs:
            norm_doc = np.linalg.norm(doc_vec)
            if norm_doc == 0:
                similarities.append(0)
            else:
                sim = np.dot(query_vec, doc_vec) / (norm_query * norm_doc)
                similarities.append(sim)
        
        # Keywords
        query_words = set(self.preprocess(query))
        doc_words = set(self.preprocess(documents[0]))
        matching = list(query_words.intersection(doc_words))
        missing = list(doc_words - query_words)
        
        score = (similarities[0] * 100) if similarities else 0
        return score, matching, missing
