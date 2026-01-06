import numpy as np
import heapq
from collections import defaultdict
import re

class SkillGraph:
    def __init__(self):
        self.graph = defaultdict(list)
        self.nodes = set()

    def add_edge(self, u, v, weight):
        """Adds a directed edge from u to v with a given weight."""
        self.graph[u].append((v, weight))
        self.nodes.add(u)
        self.nodes.add(v)

    def dijkstra(self, start_node, end_node):
        """
        Implements Dijkstra's Algorithm to find the shortest path.
        Returns (path, total_weight).
        """
        if start_node not in self.nodes or end_node not in self.nodes:
            return None, float('inf')

        # Priority queue stores (current_distance, current_node, path)
        pq = [(0, start_node, [start_node])]
        visited = set()

        while pq:
            current_dist, current_node, path = heapq.heappop(pq)

            if current_node in visited:
                continue
            visited.add(current_node)

            if current_node == end_node:
                return path, current_dist

            for neighbor, weight in self.graph[current_node]:
                if neighbor not in visited:
                    heapq.heappush(pq, (current_dist + weight, neighbor, path + [neighbor]))

        return None, float('inf')

    def build_sample_career_graph(self):
        """Builds a sample graph for demonstration."""
        # Web Development Path
        self.add_edge("HTML/CSS", "JavaScript", 2)
        self.add_edge("JavaScript", "React", 3)
        self.add_edge("JavaScript", "Node.js", 3)
        self.add_edge("React", "Frontend Developer", 2)
        self.add_edge("Node.js", "Backend Developer", 2)
        self.add_edge("Frontend Developer", "Full Stack Developer", 4)
        self.add_edge("Backend Developer", "Full Stack Developer", 4)

        # Data Science Path
        self.add_edge("Python", "Pandas", 2)
        self.add_edge("Pandas", "Scikit-Learn", 3)
        self.add_edge("Scikit-Learn", "Deep Learning", 5)
        self.add_edge("Deep Learning", "AI Engineer", 3)
        self.add_edge("Scikit-Learn", "Data Scientist", 3)

        # Cross-domain
        self.add_edge("Python", "Backend Developer", 4) # Django/Flask route

class LSAEngine:
    def __init__(self):
        self.vocabulary = []
        self.tfidf_matrix = None

    def preprocess(self, text):
        """Simple preprocessing: lowercase and remove non-alphanumeric."""
        text = text.lower()
        return re.findall(r'\b\w+\b', text)

    def fit_transform(self, documents):
        """
        Manually implements TF-IDF and SVD.
        """
        # 1. Build Vocabulary
        unique_words = set()
        processed_docs = [self.preprocess(doc) for doc in documents]
        for doc in processed_docs:
            unique_words.update(doc)
        self.vocabulary = sorted(list(unique_words))
        vocab_index = {word: i for i, word in enumerate(self.vocabulary)}

        # 2. Compute TF (Term Frequency)
        n_docs = len(documents)
        n_vocab = len(self.vocabulary)
        tf_matrix = np.zeros((n_docs, n_vocab))

        for i, doc in enumerate(processed_docs):
            doc_len = len(doc)
            if doc_len == 0: continue
            for word in doc:
                if word in vocab_index:
                    tf_matrix[i, vocab_index[word]] += 1
            tf_matrix[i] = tf_matrix[i] / doc_len

        # 3. Compute IDF (Inverse Document Frequency)
        df = np.sum(tf_matrix > 0, axis=0)
        idf = np.log((n_docs + 1) / (df + 1)) + 1
        
        # 4. TF-IDF Matrix
        self.tfidf_matrix = tf_matrix * idf

        # 5. SVD (Singular Value Decomposition)
        # Decompose X into U * S * Vt
        # We use numpy's svd for stability, but the logic is explicit
        try:
            U, S, Vt = np.linalg.svd(self.tfidf_matrix, full_matrices=False)
            
            # Reduce dimensions (LSA) - keep top k components
            k = min(n_docs, n_vocab, 100) # Keep up to 100 dimensions
            self.U_k = U[:, :k]
            self.S_k = np.diag(S[:k])
            self.Vt_k = Vt[:k, :]
            
            # Project documents into latent space
            self.lsa_matrix = np.dot(self.U_k, self.S_k)
            
            return self.lsa_matrix
        except np.linalg.LinAlgError:
            # Fallback if SVD fails (e.g. empty matrix)
            return self.tfidf_matrix

    def compute_similarity(self, query, documents):
        """
        Computes cosine similarity between query and documents in LSA space.
        """
        # Fit on documents + query to ensure shared vocabulary space for this demo
        # In a real prod system, you'd fit on training data and transform query
        all_docs = [query] + documents
        lsa_matrix = self.fit_transform(all_docs)
        
        query_vec = lsa_matrix[0]
        doc_vecs = lsa_matrix[1:]
        
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
        
        # Extract keywords (simple overlap for display, since LSA is abstract)
        query_words = set(self.preprocess(query))
        doc_words = set(self.preprocess(documents[0])) # Assuming 1 doc for now
        matching = list(query_words.intersection(doc_words))
        missing = list(doc_words - query_words)
        
        return (similarities[0] * 100) if similarities else 0, matching, missing

class PersonalityClassifier:
    def __init__(self):
        # Define centroids for 16 MBTI types in 4D space (E/I, S/N, T/F, J/P)
        # Dimensions: 
        # 0: E (+1) vs I (-1)
        # 1: S (+1) vs N (-1)
        # 2: T (+1) vs F (-1)
        # 3: J (+1) vs P (-1)
        self.centroids = {
            'ESTJ': [1, 1, 1, 1], 'ESTP': [1, 1, 1, -1],
            'ESFJ': [1, 1, -1, 1], 'ESFP': [1, 1, -1, -1],
            'ENTJ': [1, -1, 1, 1], 'ENTP': [1, -1, 1, -1],
            'ENFJ': [1, -1, -1, 1], 'ENFP': [1, -1, -1, -1],
            'ISTJ': [-1, 1, 1, 1], 'ISTP': [-1, 1, 1, -1],
            'ISFJ': [-1, 1, -1, 1], 'ISFP': [-1, 1, -1, -1],
            'INTJ': [-1, -1, 1, 1], 'INTP': [-1, -1, 1, -1],
            'INFJ': [-1, -1, -1, 1], 'INFP': [-1, -1, -1, -1],
        }

    def classify(self, user_vector):
        """
        Classifies user based on Euclidean distance to nearest centroid.
        user_vector: [v1, v2, v3, v4] where values are roughly between -1 and 1.
        """
        min_dist = float('inf')
        best_match = None
        
        for p_type, centroid in self.centroids.items():
            # Euclidean Distance: sqrt(sum((p_i - q_i)^2))
            dist = np.sqrt(sum((u - c) ** 2 for u, c in zip(user_vector, centroid)))
            if dist < min_dist:
                min_dist = dist
                best_match = p_type
                
        return best_match

class RecommenderSystem:
    def __init__(self):
        # Mock User-Item Matrix (Rows: Users, Cols: Companies)
        # 1 = Interested/Prepared, 0 = Not
        self.companies = ["Google", "Amazon", "Microsoft", "Meta", "Netflix", "Apple", "Tesla", "SpaceX"]
        self.user_item_matrix = np.array([
            [1, 1, 1, 0, 0, 0, 0, 0], # User A: FAANG-ish
            [1, 0, 1, 1, 0, 1, 0, 0], # User B: Tech Giants
            [0, 1, 0, 1, 1, 0, 0, 0], # User C: Streaming/Commerce
            [0, 0, 0, 0, 0, 0, 1, 1], # User D: Hardware/Engineering
            [1, 1, 1, 1, 1, 1, 0, 0], # User E: All Software
        ])
        
    def get_recommendations(self, target_company):
        """
        Item-Based Collaborative Filtering using Jaccard Similarity.
        Finds companies similar to the target_company based on user patterns.
        """
        if target_company not in self.companies:
            return []
            
        target_idx = self.companies.index(target_company)
        target_vector = self.user_item_matrix[:, target_idx]
        
        similarities = []
        
        for i, company in enumerate(self.companies):
            if i == target_idx:
                continue
                
            other_vector = self.user_item_matrix[:, i]
            
            # Jaccard Similarity: |Intersection| / |Union|
            intersection = np.sum(np.logical_and(target_vector, other_vector))
            union = np.sum(np.logical_or(target_vector, other_vector))
            
            score = intersection / union if union > 0 else 0
            similarities.append((company, score))
            
        # Sort by score desc
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top 3
        return [company for company, score in similarities[:3] if score > 0]
