from ..models import SkillSignal, SkillEdge

class AprioriGenerator:
    def __init__(self):
        self.rules = []
        # In a real apriori, we'd mine transaction data. 
        # Here we use the graph + signal structure as a proxy for "frequent itemsets".
    
    def generate_rules(self):
        """
        Mines rules from graph structure + signals.
        Returns list of dicts: {'from': {u}, 'to': v, 'confidence': float}
        """
        self.rules = []
        high_value_skills = SkillSignal.objects.filter(success_rate__gt=0.8).values_list('skill__name', flat=True)
        high_value_set = set(high_value_skills)
        
        edges = SkillEdge.objects.select_related('source', 'target').all()
        for edge in edges:
            u, v = edge.source.name, edge.target.name
            
            # Rule: If you know U, and U -> V is an edge, and V is high value or high demand
            confidence = 85
            if hasattr(edge.target, 'market_signal'):
                confidence = int(edge.target.market_signal.success_rate * 100)
            
            # Simple implication rule
            self.rules.append({
                'from': u, # Simplifying to single item for this demo
                'to': v,
                'confidence': confidence
            })
            
        return self.rules
