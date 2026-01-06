import os
import sys
import django
from django.conf import settings

# Setup Django environment (mocking minimal settings if needed, or just testing classes directly)
# Since we are testing the algorithms class which is pure python (mostly), we can import it directly.
# However, to be safe with imports, let's just test the classes from algorithms.py directly.

sys.path.append('c:/Users/bhave/comp_research/comp_research')
from scraper.algorithms import SkillGraph, LSAEngine

def test_skill_graph():
    print("\n--- Testing Skill Graph (Dijkstra) ---")
    graph = SkillGraph()
    graph.build_sample_career_graph()
    
    start = "HTML/CSS"
    end = "Full Stack Developer"
    path, weight = graph.dijkstra(start, end)
    
    print(f"Path from {start} to {end}: {path}")
    print(f"Total Weight: {weight}")
    
    if path == ['HTML/CSS', 'JavaScript', 'React', 'Frontend Developer', 'Full Stack Developer']:
        print("SUCCESS: Path is correct (Frontend route).")
    elif path == ['HTML/CSS', 'JavaScript', 'Node.js', 'Backend Developer', 'Full Stack Developer']:
        print("SUCCESS: Path is correct (Backend route).")
    else:
        print("FAILURE: Unexpected path.")

def test_lsa_engine():
    print("\n--- Testing LSA Engine ---")
    lsa = LSAEngine()
    
    resume = "I am a python developer with experience in django and machine learning."
    job_desc = "Looking for a python developer who knows django and ml."
    
    similarity, matching, missing = lsa.compute_similarity(resume, [job_desc])
    
    print(f"Resume: {resume}")
    print(f"Job: {job_desc}")
    print(f"Similarity Score: {similarity:.2f}")
    print(f"Matching Keywords: {matching}")
    
    if similarity > 0:
        print("SUCCESS: Similarity score calculated.")
    else:
        print("FAILURE: Similarity score is 0.")

def test_personality_classifier():
    print("\n--- Testing Personality Classifier ---")
    from scraper.algorithms import PersonalityClassifier
    classifier = PersonalityClassifier()
    
    # Test case: Introverted, Intuitive, Thinking, Judging (INTJ)
    # E/I = -1, S/N = -1, T/F = 1, J/P = 1
    user_vector = [-0.8, -0.9, 0.8, 0.9] 
    result = classifier.classify(user_vector)
    
    print(f"User Vector: {user_vector}")
    print(f"Classified Type: {result}")
    
    if result == 'INTJ':
        print("SUCCESS: Correctly classified as INTJ.")
    else:
        print(f"FAILURE: Classified as {result} instead of INTJ.")

def test_recommender_system():
    print("\n--- Testing Recommender System ---")
    from scraper.algorithms import RecommenderSystem
    recommender = RecommenderSystem()
    
    target = "Google"
    recommendations = recommender.get_recommendations(target)
    
    print(f"Target Company: {target}")
    print(f"Recommendations: {recommendations}")
    
    if "Microsoft" in recommendations or "Amazon" in recommendations:
        print("SUCCESS: Relevant recommendations found.")
    else:
        print("FAILURE: Unexpected recommendations.")

if __name__ == "__main__":
    test_skill_graph()
    test_lsa_engine()
    test_personality_classifier()
    test_recommender_system()
