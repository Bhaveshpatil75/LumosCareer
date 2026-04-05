"""
MCP Tool implementations for LumosCareer.
Each tool wraps existing algorithm classes and adds RAG context retrieval.
Tools are callable actions that AI agents can invoke through the MCP server.
"""

from .rag_engine import get_rag_engine


class MCPTools:
    """Registry of all MCP-compliant tools available to AI agents."""

    TOOL_DEFINITIONS = {
        "analyze_resume": {
            "name": "analyze_resume",
            "description": "Analyzes a resume against a job description using LSA similarity, keyword extraction, and skill gap identification.",
            "input_schema": {"resume_text": "string", "job_description": "string"},
            "output_schema": {"similarity_score": "float", "matching_keywords": "list", "missing_keywords": "list"},
        },
        "compute_compatibility": {
            "name": "compute_compatibility",
            "description": "Computes candidate-company compatibility score using Bayesian prediction and resume matching.",
            "input_schema": {"resume_text": "string", "company_name": "string", "personality_type": "string"},
            "output_schema": {"compatibility_score": "float", "factors": "list"},
        },
        "find_career_path": {
            "name": "find_career_path",
            "description": "Finds the optimal career path between two skills/roles using multi-criteria Dijkstra on the skill graph.",
            "input_schema": {"start_skill": "string", "target_role": "string", "criterion": "string"},
            "output_schema": {"path": "list", "primary_cost": "int", "secondary_cost": "int"},
        },
        "predict_success": {
            "name": "predict_success",
            "description": "Predicts hiring success probability using Bayesian analysis of skill market signals.",
            "input_schema": {"skills": "list"},
            "output_schema": {"probability": "float", "details": "list"},
        },
        "generate_schedule": {
            "name": "generate_schedule",
            "description": "Generates an optimized weekly study schedule using simulated annealing to minimize burnout.",
            "input_schema": {"subjects": "list", "hours_available": "int"},
            "output_schema": {"schedule": "dict", "energy_score": "float"},
        },
        "retrieve_knowledge": {
            "name": "retrieve_knowledge",
            "description": "Retrieves relevant documents from the RAG knowledge base for a given query.",
            "input_schema": {"query": "string", "top_k": "int"},
            "output_schema": {"documents": "list"},
        },
        "get_market_signals": {
            "name": "get_market_signals",
            "description": "Fetches market demand signals for specified skills from the database.",
            "input_schema": {"skill_names": "list"},
            "output_schema": {"signals": "list"},
        },
        "recommend_companies": {
            "name": "recommend_companies",
            "description": "Recommends companies based on tech stack similarity using content-based filtering.",
            "input_schema": {"company_name": "string"},
            "output_schema": {"recommendations": "list"},
        },
        "assess_personality": {
            "name": "assess_personality",
            "description": "Maps a user's trait vector to the closest MBTI personality type using centroid classification.",
            "input_schema": {"trait_vector": "list"},
            "output_schema": {"personality_type": "string"},
        },
        "get_skill_importance": {
            "name": "get_skill_importance",
            "description": "Returns the PageRank importance score for a skill node in the knowledge graph.",
            "input_schema": {"skill_name": "string"},
            "output_schema": {"importance_score": "float", "category": "string"},
        },
    }

    @staticmethod
    def analyze_resume(resume_text, job_description):
        from .algorithms import LSAEngine
        lsa = LSAEngine()
        score, matching, missing = lsa.compute_similarity(resume_text, [job_description])
        return {
            "similarity_score": round(score, 2),
            "matching_keywords": matching[:15],
            "missing_keywords": missing[:15],
        }

    @staticmethod
    def compute_compatibility(resume_text, company_name, personality_type=""):
        from .algorithms import LSAEngine, BayesianPredictor
        from .models import Company
        lsa = LSAEngine()
        try:
            company = Company.objects.get(name=company_name)
            tech_stack = company.tech_stack
        except Company.DoesNotExist:
            tech_stack = ""

        score = 0
        if tech_stack and resume_text:
            score, _, _ = lsa.compute_similarity(resume_text, [tech_stack])

        bp = BayesianPredictor()
        import re
        skills = re.findall(r'\b[A-Z][a-zA-Z+#.]+\b', resume_text[:3000]) if resume_text else []
        unique_skills = list(set(skills))[:10]
        prob, details = bp.predict_success_probability(unique_skills)

        return {
            "compatibility_score": round(score, 2),
            "success_probability": prob,
            "skill_analysis": details,
        }

    @staticmethod
    def find_career_path(start_skill, target_role, criterion="time"):
        from .algorithms import SkillGraph
        graph = SkillGraph()
        result = graph.dijkstra_multi_criteria(start_skill, target_role, criterion)
        if result:
            return result
        return {"path": [], "primary_cost": 0, "secondary_cost": 0}

    @staticmethod
    def predict_success(skills):
        from .algorithms import BayesianPredictor
        bp = BayesianPredictor()
        prob, details = bp.predict_success_probability(skills)
        return {"probability": prob, "details": details}

    @staticmethod
    def generate_schedule(subjects, hours_available=20):
        from .algorithms import SimulatedAnnealingScheduler
        if len(subjects) < 2:
            subjects = subjects + ["Review"]
        sa = SimulatedAnnealingScheduler(subjects, hours_available)
        schedule, energy = sa.optimize()
        return {"schedule": schedule, "energy_score": energy}

    @staticmethod
    def retrieve_knowledge(query, top_k=5):
        rag = get_rag_engine()
        results = rag.retrieve(query, top_k=top_k)
        return {
            "documents": [
                {"key": key, "text": text[:500], "relevance_score": round(score, 4)}
                for key, text, score in results
            ]
        }

    @staticmethod
    def get_market_signals(skill_names):
        from .models import SkillSignal
        signals = []
        for name in skill_names:
            try:
                sig = SkillSignal.objects.select_related('skill').get(skill__name=name)
                signals.append({
                    "skill": name,
                    "success_rate": sig.success_rate,
                    "failure_rate": sig.failure_rate,
                    "demand_trend": sig.demand_trend,
                    "lift": sig.lift,
                })
            except SkillSignal.DoesNotExist:
                pass
        return {"signals": signals}

    @staticmethod
    def recommend_companies(company_name):
        from .algorithms import RecommenderSystem
        rec = RecommenderSystem()
        recs = rec.get_recommendations(company_name)
        return {"recommendations": recs}

    @staticmethod
    def assess_personality(trait_vector):
        from .algorithms import PersonalityClassifier
        classifier = PersonalityClassifier()
        result = classifier.classify(trait_vector)
        return {"personality_type": result}

    @staticmethod
    def get_skill_importance(skill_name):
        from .models import SkillNode
        try:
            node = SkillNode.objects.get(name__iexact=skill_name)
            return {
                "importance_score": node.importance_score,
                "category": node.category,
                "difficulty": node.difficulty_level,
                "learning_weeks": node.learning_weeks,
            }
        except SkillNode.DoesNotExist:
            return {"importance_score": 0, "category": "Unknown"}
