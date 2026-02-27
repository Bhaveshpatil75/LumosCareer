import heapq
from collections import defaultdict
from ..models import SkillNode, SkillEdge

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
