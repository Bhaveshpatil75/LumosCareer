import re
from ..models import Company

class RecommenderSystem:
    def __init__(self):
        pass
        
    def get_recommendations(self, company_name_or_profile):
        """
        Content-based filtering using Company Tech Stack and User Profile/Input.
        """
        # If input is a string, treat as company name to find similars
        target_tags = set()
        
        # Case 2: Input is Company Name (Item-Item filtering)
        if isinstance(company_name_or_profile, str):
            try:
                target_company = Company.objects.get(name=company_name_or_profile)
                target_tags = set(re.findall(r'\b\w+\b', target_company.tech_stack.lower()))
            except Company.DoesNotExist:
                # Fallback or empty
                return []
        
        all_companies = Company.objects.exclude(name=company_name_or_profile).exclude(tech_stack='')
        scores = []
        
        for comp in all_companies:
            comp_tags = set(re.findall(r'\b\w+\b', comp.tech_stack.lower()))
            if not comp_tags: continue
            
            # Jaccard Similarity
            intersection = len(target_tags.intersection(comp_tags))
            union = len(target_tags.union(comp_tags))
            
            score = intersection / union if union > 0 else 0
            if score > 0:
                scores.append((comp.name, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return [name for name, score in scores[:3]]

    def recommend_for_user(self, resume_text):
        """
        Recommends companies based on resume text match with company tech stacks.
        """
        resume_tokens = set(re.findall(r'\b\w+\b', resume_text.lower()))
        
        companies = Company.objects.exclude(tech_stack='')
        scores = []
        
        for comp in companies:
            comp_tokens = set(re.findall(r'\b\w+\b', comp.tech_stack.lower()))
            intersection = len(resume_tokens.intersection(comp_tokens))
            if intersection > 0:
                scores.append((comp.name, intersection))
                
        scores.sort(key=lambda x: x[1], reverse=True)
        return [name for name, score in scores[:5]]
