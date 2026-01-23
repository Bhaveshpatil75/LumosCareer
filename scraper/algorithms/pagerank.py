from ..models import SkillNode, SkillEdge

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
            
            for node in nodes:
                rank_sum = 0
                # Find all nodes 'u' that point to 'node'
                incoming_nodes = []
                for u, neighbors in graph_dict.items():
                    for v_tuple in neighbors:
                         # v_tuple is (v, weights)
                        if v_tuple[0] == node:
                            incoming_nodes.append(u)
                
                for u in incoming_nodes:
                    out_links = len(graph_dict[u])
                    if out_links > 0:
                        rank_sum += pr[u] / out_links
                
                new_pr[node] = (1 - self.damping) / N + self.damping * rank_sum
            
            diff = sum(abs(new_pr[n] - pr[n]) for n in nodes)
            if diff < 1e-6:
                break
            pr = new_pr
        
        max_score = max(pr.values()) if pr else 1
        return {k: round((v / max_score) * 100, 1) for k, v in pr.items()}

    def update_db_scores(self):
        """
        Computes PR for the entire SkillGraph in DB and updates SkillNode.importance_score.
        """
        # Circular import avoidance/delayed import
        from .graph import SkillGraph
        sg = SkillGraph()
        # Build simple adjacency list for PageRank logic
        adj = {}
        for u in sg.graph:
            adj[u] = sg.graph[u]
        
        scores = self.compute(adj, list(sg.nodes))
        
        for name, score in scores.items():
            SkillNode.objects.filter(name=name).update(importance_score=score)
