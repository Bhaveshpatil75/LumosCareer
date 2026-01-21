import sys
import os
import django

# Context setup
sys.path.append('c:/Users/bhave/comp_research/comp_research')
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "comp_research.settings")
django.setup()

from scraper.algorithms import (
    SkillGraph, LSAEngine, PersonalityClassifier, 
    RecommenderSystem, PageRank, BayesianPredictor, 
    AprioriGenerator, SimulatedAnnealingScheduler
)

def test_skill_graph():
    print("\n--- 1. Testing Advanced Skill Graph (Dijkstra Multi-Criteria from DB) ---")
    graph = SkillGraph() # Now loads from DB automatically
    
    start = "Python"
    end = "System Design" # Changed to a deeper node available in seed data
    
    if start not in graph.nodes or end not in graph.nodes:
        print(f"Skipping graph test: Nodes {start} or {end} not found in DB.")
        return

    # Test Time Criterion
    res_time = graph.dijkstra_multi_criteria(start, end, criterion='time')
    print(f"Fastest Path ({start} -> {end}):")
    if res_time:
        print(f"  Path: {res_time['path']}")
        print(f"  Total Weeks: {res_time['primary_cost']}")
        print(f"  Difficulty Score: {res_time['secondary_cost']}")
    
    # Test Difficulty Criterion
    res_diff = graph.dijkstra_multi_criteria(start, end, criterion='difficulty')
    print(f"Easiest Path ({start} -> {end}):")
    if res_diff:
        print(f"  Path: {res_diff['path']}")
        print(f"  Total Difficulty: {res_diff['primary_cost']}")
        print(f"  Weeks: {res_diff['secondary_cost']}")

def test_pagerank():
    print("\n--- 2. Testing PageRank (Skill Centrality) ---")
    graph = SkillGraph()
    
    pr = PageRank()
    scores = pr.compute(graph.graph, list(graph.nodes))
    
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    print("Top 5 Most Influential Skills (Centrality):")
    for skill, score in sorted_scores[:5]:
        print(f"  {skill}: {score}")

def test_bayesian():
    print("\n--- 3. Testing Bayesian Predictor (Success Probability) ---")
    bp = BayesianPredictor()
    
    skills = ['Python', 'Docker', 'Kubernetes'] # Strong stack
    prob, details = bp.calculate_success_probability(skills)
    
    print(f"Skills: {skills}")
    print(f"Predicted Hiring Probability: {prob}%")
    print("Signal Breakdown:")
    for d in details:
        print(f"  {d['skill']}: Lift {d['lift']}x ({d['impact']})")

def test_apriori():
    print("\n--- 4. Testing Apriori (Skill Association Rules) ---")
    ap = AprioriGenerator()
    recs = ap.get_recommendations(['Python', 'Django'])
    
    print("Recommendations for [Python, Django]:")
    for r in recs:
        print(f"  Recommend: {r['to']} (Conf: {r['confidence']}%)")

def test_annealing():
    print("\n--- 5. Testing Simulated Annealing (Schedule Optimization) ---")
    subjects = ['React', 'TypeScript', 'Node.js', 'System Design']
    sa = SimulatedAnnealingScheduler(subjects)
    
    schedule, energy = sa.optimize()
    print(f"Optimized Schedule (Energy: {energy}):")
    for day in ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']:
        print(f"  {day}: {schedule[day]}")

if __name__ == "__main__":
    test_skill_graph()
    test_pagerank()
    test_bayesian()
    test_apriori()
    test_annealing()
