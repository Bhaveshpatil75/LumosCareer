import numpy as np
import heapq
from collections import defaultdict
import re
import math
import random
from .models import SkillNode, SkillEdge, SkillSignal

class SkillGraph:
    def __init__(self):
        self.graph = defaultdict(list)
        self.nodes = set()
        self.build_graph_from_db()

    def build_graph_from_db(self):
        """
        Constructs the graph from Django Models.
        """
        self.metadata = {}
        
        # Load all Edges
        edges = SkillEdge.objects.select_related('source', 'target').all()
        for edge in edges:
            u, v = edge.source.name, edge.target.name
            weights = {'time': edge.weight_time, 'difficulty': edge.weight_difficulty}
            self.add_edge_multi(u, v, weights)
        
        # Ensure all nodes are present and populate metadata
        all_nodes = SkillNode.objects.all()
        for node in all_nodes:
            self.nodes.add(node.name)
            self.metadata[node.name] = node.category

    def add_edge_multi(self, u, v, weights):
        """Adds a directed edge with multiple weight criteria."""
        self.graph[u].append((v, weights))
        self.nodes.add(u)
        self.nodes.add(v)

    def get_all_paths(self, start, end, max_depth=7):
        """DFS to find all paths up to max_depth."""
        paths = []
        stack = [(start, [start], 0, 0)] # current, path, total_weight, total_diff
        
        while stack:
            u, path, w, d = stack.pop()
            if u == end:
                paths.append({'path': path, 'time': w, 'difficulty': d})
                continue
            
            if len(path) >= max_depth:
                continue
                
            for v, weights in self.graph[u]:
                if v not in path:
                    # Default to 1 if weight missing
                    w_val = weights.get('time', 1)
                    d_val = weights.get('difficulty', 1)
                    stack.append((v, path + [v], w + w_val, d + d_val))
        return paths

    def dijkstra_multi_criteria(self, start_node, end_node, criterion='time'):
        """
        Dijkstra with support for multiple edge weights (time/difficulty).
        """
        if start_node not in self.nodes or end_node not in self.nodes:
            return None

        # Priority queue: (current_cost, current_node, path, secondary_cost)
        pq = [(0, start_node, [start_node], 0)]
        visited = set()
        min_costs = {start_node: 0}

        while pq:
            current_cost, current_node, path, sec_cost = heapq.heappop(pq)

            if current_node == end_node:
                return {
                    'path': path,
                    'primary_cost': current_cost,
                    'secondary_cost': sec_cost
                }

            if current_cost > min_costs.get(current_node, float('inf')):
                continue
            
            visited.add(current_node)

            for neighbor, weights in self.graph[current_node]:
                # Dynamic weight selection based on user criterion
                weight = weights.get(criterion, 1)
                other_criterion = 'difficulty' if criterion == 'time' else 'time'
                other_weight = weights.get(other_criterion, 1)

                new_cost = current_cost + weight
                
                if new_cost < min_costs.get(neighbor, float('inf')):
                    min_costs[neighbor] = new_cost
                    heapq.heappush(pq, (new_cost, neighbor, path + [neighbor], sec_cost + other_weight))

        return None

class PageRank:
    def __init__(self, damping=0.85, max_iter=100):
        self.damping = damping
        self.max_iter = max_iter

    def compute(self, graph_dict, nodes):
        """
        Computes PageRank for the given graph.
        Returns dict: {node: score}
        """
        N = len(nodes)
        if N == 0: return {}
        
        pr = {node: 1.0 / N for node in nodes}
        
        for _ in range(self.max_iter):
            new_pr = {}
            total_diff = 0
            
            for node in nodes:
                rank_sum = 0
                # Find all nodes 'u' that point to 'node'
                # In our adjacency list, we have u -> [v1, v2]
                # We need reverse graph essentially, but let's iterate (slow but fine for <1000 nodes)
                incoming_nodes = []
                for u, neighbors in graph_dict.items():
                    # neighbors is list of (v, weight) or just v
                    # Check structure from SkillGraph
                    for v_tuple in neighbors:
                         # v_tuple is (v, weights)
                        if v_tuple[0] == node:
                            incoming_nodes.append(u)
                
                for u in incoming_nodes:
                    # Outgoing count of u
                    out_links = len(graph_dict[u])
                    if out_links > 0:
                        rank_sum += pr[u] / out_links
                
                new_pr[node] = (1 - self.damping) / N + self.damping * rank_sum
            
            # Check convergence
            diff = sum(abs(new_pr[n] - pr[n]) for n in nodes)
            if diff < 1e-6:
                break
            pr = new_pr
        
        # Normalize to 0-100 for display
        max_score = max(pr.values()) if pr else 1
        return {k: round((v / max_score) * 100, 1) for k, v in pr.items()}


class BayesianPredictor:
    def __init__(self):
        # Fetch signals from DB
        self.signals = {}
        db_signals = SkillSignal.objects.all()
        for signal in db_signals:
            self.signals[signal.skill.name] = {
                'P(E|H)': signal.success_rate,
                'P(E|~H)': signal.failure_rate
            }
        
        # Default Priors
        self.P_H = 0.10  # Base hiring rate (10%)
        self.P_not_H = 0.90

    def calculate_success_probability(self, user_skills):
        """
        Calculates P(Hired | Skills) using Naive Bayes.
        P(H|E1, E2) = P(H) * Prod(P(Ei|H)) / P(E)
        """
        if not user_skills:
            return 10.0, [] # Default low prob

        # Working with log-odds to avoid underflow
        log_odds = np.log(self.P_H / self.P_not_H)
        
        details = []

        for skill in user_skills:
            if skill in self.signals:
                p_e_given_h = self.signals[skill]['P(E|H)']
                p_e_given_not_h = self.signals[skill]['P(E|~H)']
                
                # Avoid division by zero
                if p_e_given_not_h == 0: p_e_given_not_h = 0.001
                
                lift = p_e_given_h / p_e_given_not_h
                log_odds += np.log(lift)
                
                details.append({
                    'skill': skill,
                    'lift': round(lift, 2),
                    'impact': 'High' if lift > 1.5 else 'Neutral'
                })

        # Convert log odds back to probability
        odds = np.exp(log_odds)
        probability = odds / (1 + odds)
        
        return round(probability * 100, 1), details


class AprioriGenerator:
    def __init__(self):
        self.rules = []
        self._mine_rules_from_db()

    def _mine_rules_from_db(self):
        """
        Mining rules from graph structure + signals.
        "If (Django) is High Signal AND Connected to (React) -> Recommend React"
        """
        # Finds high-value adjacent nodes
        high_value_skills = SkillSignal.objects.filter(success_rate__gt=0.8).values_list('skill__name', flat=True)
        high_value_set = set(high_value_skills)
        
        edges = SkillEdge.objects.all()
        for edge in edges:
            u, v = edge.source.name, edge.target.name
            
            # Simple rule: If U -> V, and V is high value, imply U => V
            # Or if U -> V, suggest V if user knows U
            if v in high_value_set:
                self.rules.append({
                    'from': {u},
                    'to': v,
                    'confidence': int(edge.target.market_signal.success_rate * 100) if hasattr(edge.target, 'market_signal') else 85
                })

    def get_recommendations(self, current_skills):
        """
        Returns recommendations based on rules.
        """
        current_set = set(current_skills)
        recs = []
        
        for rule in self.rules:
            if rule['from'].issubset(current_set) and rule['to'] not in current_set:
                recs.append({
                    'to': rule['to'],
                    'confidence': rule['confidence']
                })
        
        # Sort by confidence
        recs.sort(key=lambda x: x['confidence'], reverse=True)
        return recs[:5]

class SimulatedAnnealingScheduler:
    """
    Optimization Algorithm.
    Generates an optimal weekly study schedule minimizing burnout and maximizing retention.
    """
    def __init__(self, subjects, hours_available=20):
        self.subjects = subjects # List of subjects
        self.hours_available = hours_available
        # Constraints
        self.max_daily_hours = 4

    def energy(self, schedule):
        """
        Cost function (Lower is better).
        Penalizes:
        - Exceeding daily limits
        - Studying same subject 3 days in a row (burnout)
        - Uneven distribution
        """
        penalty = 0
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        daily_hours = defaultdict(int)
        subject_counts = defaultdict(int)
        
        last_subject = None
        consecutive_days = 0
        
        for day in days:
            subj = schedule.get(day)
            if subj != 'Rest':
                daily_hours[day] += 2 # Assume 2 hour blocks
                subject_counts[subj] += 1
                
                if subj == last_subject:
                    consecutive_days += 1
                else:
                    consecutive_days = 0
                last_subject = subj
                
                if consecutive_days >= 2: # 3rd day in a row
                    penalty += 50
            else:
                consecutive_days = 0
                last_subject = None

        # Diversity Bonus (We want to cover all subjects)
        unique_subjects = len([s for s in schedule.values() if s != 'Rest'])
        if unique_subjects < len(self.subjects) and len(self.subjects) < 5:
            penalty += 100 * (len(self.subjects) - unique_subjects)
            
        return penalty

    def optimize(self, iterations=1000):
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        # Initial State: Random
        current_schedule = {day: random.choice(self.subjects + ['Rest']) for day in days}
        current_energy = self.energy(current_schedule)
        
        best_schedule = current_schedule.copy()
        best_energy = current_energy
        
        temperature = 100.0
        cooling_rate = 0.95
        
        for i in range(iterations):
            # Neighbor: Change one day randomly
            neighbor = current_schedule.copy()
            day_to_change = random.choice(days)
            neighbor[day_to_change] = random.choice(self.subjects + ['Rest'])
            
            neighbor_energy = self.energy(neighbor)
            
            # Acceptance Probability
            if neighbor_energy < current_energy:
                current_schedule = neighbor
                current_energy = neighbor_energy
            else:
                prob = math.exp((current_energy - neighbor_energy) / temperature)
                if random.random() < prob:
                    current_schedule = neighbor
                    current_energy = neighbor_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_schedule = current_schedule.copy()
                
            temperature *= cooling_rate
            
        return best_schedule, best_energy

class LSAEngine:
    def __init__(self):
        self.vocabulary = []
        self.tfidf_matrix = None

    def preprocess(self, text):
        text = text.lower()
        return re.findall(r'\b\w+\b', text)

    def fit_transform(self, documents):
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

        # 3. Compute IDF
        df = np.sum(tf_matrix > 0, axis=0)
        idf = np.log((n_docs + 1) / (df + 1)) + 1
        
        # 4. TF-IDF Matrix
        self.tfidf_matrix = tf_matrix * idf

        # 5. SVD (Singular Value Decomposition)
        try:
            U, S, Vt = np.linalg.svd(self.tfidf_matrix, full_matrices=False)
            k = min(n_docs, n_vocab, 100)
            self.U_k = U[:, :k]
            self.S_k = np.diag(S[:k])
            self.Vt_k = Vt[:k, :]
            
            # Project
            self.lsa_matrix = np.dot(self.U_k, self.S_k)
            return self.lsa_matrix
        except np.linalg.LinAlgError:
            return self.tfidf_matrix

    def compute_similarity(self, query, documents):
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
        
        query_words = set(self.preprocess(query))
        doc_words = set(self.preprocess(documents[0]))
        matching = list(query_words.intersection(doc_words))
        missing = list(doc_words - query_words)
        
        return (similarities[0] * 100) if similarities else 0, matching, missing

class PersonalityClassifier:
    def __init__(self):
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
        min_dist = float('inf')
        best_match = None
        
        for p_type, centroid in self.centroids.items():
            dist = np.sqrt(sum((u - c) ** 2 for u, c in zip(user_vector, centroid)))
            if dist < min_dist:
                min_dist = dist
                best_match = p_type
                
        return best_match

class RecommenderSystem:
    def __init__(self):
        self.companies = ["Google", "Amazon", "Microsoft", "Meta", "Netflix", "Apple", "Tesla", "SpaceX"]
        # Mock User-Item Matrix
        self.user_item_matrix = np.array([
            [1, 1, 1, 0, 0, 0, 0, 0],
            [1, 0, 1, 1, 0, 1, 0, 0],
            [0, 1, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1],
            [1, 1, 1, 1, 1, 1, 0, 0],
        ])
        
    def get_recommendations(self, target_company):
        if target_company not in self.companies:
            return []
            
        target_idx = self.companies.index(target_company)
        target_vector = self.user_item_matrix[:, target_idx]
        
        similarities = []
        
        for i, company in enumerate(self.companies):
            if i == target_idx:
                continue
            other_vector = self.user_item_matrix[:, i]
            intersection = np.sum(np.logical_and(target_vector, other_vector))
            union = np.sum(np.logical_or(target_vector, other_vector))
            score = intersection / union if union > 0 else 0
            similarities.append((company, score))
            
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [company for company, score in similarities[:3] if score > 0]

