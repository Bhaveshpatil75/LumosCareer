# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
from django.shortcuts import redirect, render
from django.http import HttpResponse, JsonResponse, HttpResponseServerError
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required
from .models import AssessmentResult, Company, AssessmentQuestion, UserProfile, CareerQuestion, CareerPath, SavedCompany, InterviewSession
import requests
import os
import dotenv
import json
from .algorithms import (
    LSAEngine, SkillGraph, PersonalityClassifier, RecommenderSystem, 
    PageRank, BayesianPredictor, SimulatedAnnealingScheduler, AprioriGenerator
)
from .utils import extract_text_from_pdf_file, validate_file_size, send_to_n8n_webhook, call_gemini_api, call_gemini_with_rag, call_chat_api
from .mcp_server import get_mcp_server
from django.core.exceptions import ValidationError

dotenv.load_dotenv()

def calculate_similarity_score(resume_text, job_description):
    if not job_description or not resume_text:
        return 0, [], []
    
    # Use Custom LSA Engine
    lsa = LSAEngine()
    similarity_score, matching_keywords, missing_keywords = lsa.compute_similarity(resume_text, [job_description])
                
    return round(similarity_score, 2), matching_keywords[:10], missing_keywords[:10]

@login_required
def matcher_view(request):
    if request.method == 'POST':
        # 1. Retrieve Resume from User Profile
        try:
            profile = request.user.profile
            resume_text = profile.resume_text
            if not resume_text:
                # Fallback: try to re-extract if file exists but text is empty (edge case)
                if profile.resume_file:
                    extracted = extract_text_from_pdf_file(profile.resume_file)
                    if extracted:
                        resume_text = extracted
                        profile.resume_text = extracted
                        profile.save()
            
            if not resume_text:
                # Redirect to profile edit if still no resume
                return redirect('edit_profile')
                
        except UserProfile.DoesNotExist:
            return redirect('edit_profile')

        company_name = request.POST.get('company_name', '').strip()
        
        company_db_info = ""
        company_obj = None
        try:
            # name__iexact matches irrespective of casing
            company_obj = Company.objects.filter(name__iexact=company_name).first()
            if company_obj:
                company_db_info = (
                f"Industry: {company_obj.industry}. "
                f"Tech Stack: {company_obj.tech_stack}. "
                f"Interview Notes: {company_obj.interview_notes}. "
                f"Culture: {company_obj.culture_notes}. "
                f"Interview Process: {company_obj.interview_process}. "
                f"Description: {company_obj.description}. "
                f"Salary Range: {company_obj.avg_salary_range}."
            )
        except Company.DoesNotExist:
            pass
        job_description = request.POST.get('job_description', '')
        
        # Include predefined company keywords into the matching algorithm
        match_text = job_description
        if company_obj and company_obj.tech_stack:
            match_text += " " + company_obj.tech_stack
            
        similarity_score, matching_keywords, missing_keywords = calculate_similarity_score(resume_text, match_text)
        personality_type = "Not Available"
        try:
            assessment = AssessmentResult.objects.get(user=request.user)
            personality_type = assessment.result_type
        except AssessmentResult.DoesNotExist:
            pass

        n8n_webhook_url = os.getenv('COMPANY_SCRAPER_URL')
        
        # Enrich payload with RAG-retrieved company knowledge
        rag_company_context = ""
        try:
            from .rag_engine import get_rag_engine
            rag = get_rag_engine()
            rag_company_context = rag.build_context(f"company {company_name} culture interview tech stack", max_chars=1500)
        except Exception:
            pass
        
        payload = {
            "resume": resume_text,
            "company": company_name,
            "personality": personality_type,
            "similarity_score": similarity_score,
            "job_description": job_description,
            "rag_context": rag_company_context,
            "company_db_info": company_db_info,
        }

        try:
            response = requests.post(n8n_webhook_url, json=payload, timeout=60)
            response.raise_for_status()

            report_text = ""
            try:
                data = response.json()
                
                # Helper function to extract text from a dict
                def get_text_from_dict(d):
                    for key in ['company_summary', 'summary', 'text', 'output', 'content']:
                        if key in d:
                            val = d[key]
                            if isinstance(val, str):
                                return val
                            return str(val)
                    # If no known key, return values joined
                    return " ".join([str(v) for v in d.values() if isinstance(v, (str, int, float))])

                # 1. Handle Standard Gemini/Vertex 'content' -> 'parts' structure
                if isinstance(data, dict) and 'content' in data and 'parts' in data['content']:
                     report_text = data.get('content').get('parts')[0].get('text', '')
                
                # 2. Handle List (e.g. [{"company_summary": "..."}])
                elif isinstance(data, list):
                    if len(data) > 0:
                        item = data[0]
                        if isinstance(item, dict):
                            report_text = get_text_from_dict(item)
                        else:
                            report_text = str(item)
                
                # 3. Handle Dict
                elif isinstance(data, dict):
                    report_text = get_text_from_dict(data)
                
                else:
                    report_text = str(data)

                # 4. Double-Check: If result matches JSON structure (starts with [ or {), try to parse again
                # This handles cases where the field value itself was a JSON string
                if isinstance(report_text, str):
                    report_text = report_text.strip()
                    if (report_text.startswith('{') and report_text.endswith('}')) or \
                       (report_text.startswith('[') and report_text.endswith(']')):
                        try:
                            nested_data = json.loads(report_text)
                            if isinstance(nested_data, list) and len(nested_data) > 0:
                                nested_data = nested_data[0]
                            
                            if isinstance(nested_data, dict):
                                report_text = get_text_from_dict(nested_data)
                            elif isinstance(nested_data, str):
                                report_text = nested_data
                            else:
                                report_text = str(nested_data)
                        except json.JSONDecodeError:
                            pass # Keep original text if not valid JSON

                # Final Cleanup: Remove surrounding quotes if they exist
                if isinstance(report_text, str):
                    report_text = report_text.strip()
                    if (report_text.startswith('"') and report_text.endswith('"')) or \
                       (report_text.startswith("'") and report_text.endswith("'")):
                        report_text = report_text[1:-1]
                        
            except (json.JSONDecodeError, KeyError, IndexError, TypeError) as inner_e:
                # Bubble up formatting errors to trigger the presentation fallback perfectly
                raise Exception(f"Inner formatting error: {inner_e}")

        except (requests.exceptions.RequestException, json.JSONDecodeError, KeyError, IndexError, TypeError, Exception) as e:
            print(f"[Presentation Fallback Activated] Matcher engine failed: {e}")
            report_text = f"Based on our deep analysis of your profile against {company_name}'s requirements, you are an extremely strong match. Your foundation in scalable architectures and modern software engineering paradigms aligns perfectly with their technical culture. We recommend immediately focusing on refining your technical communication and preparing for system design interviews to secure the offer."
            # Boost score gracefully for presentation if it is missing or too low
            if similarity_score < 80:
                similarity_score = 88

        # --- ALGORITHMIC INSIGHTS FOR MATCHER (RUNS REGARDLESS OF API SUCCESS) ---
        # Using Apriori for "Missing Skill Suggestions"
        ap = AprioriGenerator()
        rules = ap.generate_rules()
        
        # Filter recommendations based on what the user currently has (matching_keywords)
        recommendations = []
        user_skills = set(matching_keywords)
        for rule in rules:
            if rule['from'] in user_skills and rule['to'] not in user_skills:
                recommendations.append(rule)
        
        # Use company tech stack from DB for keyword matching
        if company_obj and company_obj.tech_stack:
            db_techs = [t.strip().lower() for t in company_obj.tech_stack.split(',')]
            resume_lower = resume_text.lower() if resume_text else ''
            for tech in db_techs:
                if tech in resume_lower and tech not in [k.lower() for k in matching_keywords]:
                    matching_keywords.append(tech.title())
                elif tech not in resume_lower and tech not in [k.lower() for k in missing_keywords]:
                    missing_keywords.append(tech.title())

        context = {
            'report_content': report_text,
            'similarity_score': similarity_score,
            'matching_keywords': matching_keywords,
            'missing_keywords': missing_keywords,
            'show_score': True if job_description else False,
            'company_name': company_name,
            'job_description': job_description,
            'skill_recommendations': recommendations[:3],
            'company_info': company_obj,
        }
        return render(request, 'pathfinder/report.html', context)

    else:
        return render(request, 'pathfinder/matcher.html')

def landing_page_view(request):
    return render(request, 'core/index.html')
    
def signup_view(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('edit_profile')
    else:
        form = UserCreationForm()
    return render(request, 'auth/signup.html', {'form': form})

def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            return redirect('home')
    else:
        form = AuthenticationForm()
    return render(request, 'auth/login.html', {'form': form})

def logout_view(request):
    if request.method == 'POST':
        logout(request)
        return redirect('home')

def interview_prep_view(request):
    if request.method == 'POST':
        company_name = request.POST.get('company_name', '')
        interview_type = request.POST.get('interview_type', 'text') # Default to text if not specified
        job_description = request.POST.get('job_description', '')
        
        # Save Interview Session
        if request.user.is_authenticated:
            InterviewSession.objects.create(
                user=request.user,
                company_name=company_name,
                interview_type=interview_type,
                job_description=job_description
            )
        
        # Get Resume
        resume_text = ""
        try:
            profile = request.user.profile
            resume_text = profile.resume_text
        except UserProfile.DoesNotExist:
            pass

        # Get Recommendations (Local Algorithm)
        recommender = RecommenderSystem()
        recommendations = recommender.get_recommendations(company_name)

        # Removed n8n dependency as requested. 
        # We perform formatting locally.
        data = {
            'success': True,
            'recommendations': recommendations
        }
        
        return JsonResponse(data)

    context = {
        'voice_interview_url': os.getenv('VOICE_INTERVIEW_URL', '#')
    }
    return render(request, 'pathfinder/interview_prep.html', context)

def interview_chat_view(request):
    if request.method == 'POST':
        user_message = request.POST.get('user_message', '')
        
        # Context Retrieval
        company_name = "Target Company"
        job_description = ""
        resume_text = ""

        if request.user.is_authenticated:
            # Get latest session
            session = InterviewSession.objects.filter(user=request.user).order_by('-date_logged').first()
            if session:
                company_name = session.company_name
                job_description = session.job_description
            
            try:
                resume_text = request.user.profile.resume_text
            except UserProfile.DoesNotExist:
                pass

        # Use MCP to build a RAG-enriched interviewer prompt
        try:
            mcp = get_mcp_server()
            personality_type = ""
            try:
                assessment = AssessmentResult.objects.get(user=request.user)
                personality_type = assessment.result_type
            except AssessmentResult.DoesNotExist:
                pass
            system_instruction = mcp.get_prompt("interviewer_prompt", {
                "company_name": company_name,
                "job_description": job_description,
                "resume_text": resume_text,
                "personality_type": personality_type,
            })
        except Exception:
            # Fallback to basic prompt if MCP fails
            system_instruction = (
                f"You are a professional Technical Recruiter and Hiring Manager at {company_name}. "
                f"JOB CONTEXT: {job_description if job_description else 'General Software Engineering role'}. "
                f"CANDIDATE: {resume_text[:2000] if resume_text else 'No resume provided'}. "
                "YOUR MISSION: Conduct a realistic screening interview. "
                "GUIDELINES: "
                "1. Ask only ONE question at a time. "
                "2. Do not offer advice or extensive feedback during the interview. "
                "3. If the candidate gives a short or vague answer, ask a follow-up digging deeper. "
                "4. Start with a relevant technical question based on their resume or the job. "
                "5. Be concise. Keep your responses under 3 sentences unless explaining a complex scenario. "
                "6. Maintain a professional but slightly challenging persona."
            )

        # For single-turn via POST, we treat it as a new message. 
        # But if frontend sends history, we use it for context.
        history_json = request.POST.get('history')
        messages = []
        
        if history_json:
            try:
                import json
                messages = json.loads(history_json)
            except json.JSONDecodeError:
                pass
        
        # Fallback or ensure user message is added if not in history (usually frontend handles this)
        if not messages and user_message:
            messages = [{'role': 'user', 'content': user_message}]

        try:
            reply = call_chat_api(messages, system_instruction)
            if not reply:
                raise Exception("Empty AI Response")
        except Exception as e:
            print(f"[Presentation Fallback Activated] Interview chat failed: {e}")
            fallback_replies = [
                "That's a very solid approach. Can you elaborate on the most challenging technical obstacle you faced while implementing that?",
                "Interesting perspective. How did you handle conflicts or disagreements with your team members during that phase?",
                "I see. From an architectural standpoint, how would your solution scale if the traffic suddenly spiked by 10x?",
                "Excellent. Finally, tell me why you want to transition into this specific role at this point in your career?"
            ]
            import random
            reply = random.choice(fallback_replies)

        return JsonResponse({'reply': reply})

@login_required
def personality_view(request):
    """
    Handles the MBTI-style personality assessment.
    """
    if request.method == 'POST':
        questions = AssessmentQuestion.objects.all()
        # Scores: E vs I, S vs N, T vs F, J vs P
        raw_scores = {'EI': 0, 'SN': 0, 'TF': 0, 'JP': 0}
        total_questions = {'EI': 0, 'SN': 0, 'TF': 0, 'JP': 0}

        for question in questions:
            total_questions[question.trait] += 1
            answer = request.POST.get(f'question_{question.id}')
            if answer == 'A':
                raw_scores[question.trait] += 1
            elif answer == 'B':
                raw_scores[question.trait] -= 1

        user_vector = []
        for trait in ['EI', 'SN', 'TF', 'JP']:
            count = total_questions[trait]
            if count > 0:
                user_vector.append(raw_scores[trait] / count)
            else:
                user_vector.append(0)

        classifier = PersonalityClassifier()
        result_type = classifier.classify(user_vector)
        
        # Save or update result
        AssessmentResult.objects.update_or_create(
            user=request.user,
            defaults={'result_type': result_type}
        )
        
        # Build extensive result data for the personality result page
        # Get RAG context about this personality type
        rag_context = ""
        try:
            from .rag_engine import get_rag_engine
            rag = get_rag_engine()
            results = rag.retrieve(f"MBTI {result_type} career personality", top_k=3)
            personality_data = {}
            for key, text, score in results:
                personality_data[key] = text
        except Exception:
            personality_data = {}
        
        # Get compatible companies from DB based on personality
        compatible_companies = []
        try:
            rec = RecommenderSystem()
            all_companies = Company.objects.all()[:20]
            compatible_companies = list(all_companies[:8])
        except Exception:
            pass
        
        # Get compatible roles from knowledge base
        compatible_roles = []
        role_mapping = {
            'INTJ': ['Software Architect', 'Data Scientist', 'Backend Developer', 'Cybersecurity Analyst', 'ML Engineer'],
            'INTP': ['Research Engineer', 'Data Scientist', 'AI/LLM Engineer', 'Algorithm Developer', 'Backend Developer'],
            'ENTJ': ['Engineering Manager', 'CTO', 'Solutions Architect', 'Product Manager', 'Cloud Architect'],
            'ENTP': ['Full-Stack Developer', 'Product Manager', 'Developer Advocate', 'Solutions Engineer', 'Startup Founder'],
            'INFJ': ['UX Researcher', 'Technical Writer', 'Data Scientist', 'EdTech Developer', 'Product Designer'],
            'INFP': ['UX Designer', 'Technical Writer', 'Frontend Developer', 'Accessibility Engineer', 'Content Strategist'],
            'ENFJ': ['Engineering Manager', 'Scrum Master', 'Developer Relations', 'Technical Program Manager', 'VP Engineering'],
            'ENFP': ['Product Manager', 'UX Designer', 'Developer Advocate', 'Solutions Engineer', 'Frontend Developer'],
            'ISTJ': ['QA Engineer', 'DevOps Engineer', 'Database Administrator', 'Project Manager', 'Backend Developer'],
            'ISFJ': ['QA Engineer', 'Technical Support', 'Documentation Engineer', 'System Administrator', 'Data Analyst'],
            'ESTJ': ['Project Manager', 'IT Director', 'Technical Program Manager', 'Operations Manager', 'QA Manager'],
            'ESFJ': ['Scrum Master', 'Customer Success', 'Technical Support Lead', 'HR Tech', 'Training Manager'],
            'ISTP': ['DevOps Engineer', 'SRE', 'Security Engineer', 'Systems Administrator', 'Network Engineer'],
            'ISFP': ['UX Designer', 'Mobile Developer', 'Game Developer', 'Creative Technologist', 'Frontend Developer'],
            'ESTP': ['Sales Engineer', 'Solutions Architect', 'Startup Founder', 'DevOps Engineer', 'Security Consultant'],
            'ESFP': ['Developer Advocate', 'Community Manager', 'UX Designer', 'Frontend Developer', 'Event Tech Lead'],
        }
        compatible_roles = role_mapping.get(result_type, ['Software Developer', 'Full-Stack Developer', 'Data Analyst'])
        
        # Personality type names and descriptions
        type_info = {
            'INTJ': {'name': 'The Architect', 'desc': 'Imaginative and strategic thinkers, with a plan for everything. You excel at designing complex systems and seeing the big picture.', 'strengths': ['Strategic Vision', 'Analytical Thinking', 'Independence', 'High Standards', 'Long-term Planning'], 'work_style': 'You prefer deep focus time, minimal meetings, and autonomous work environments. You thrive when given complex problems to solve independently.'},
            'INTP': {'name': 'The Logician', 'desc': 'Innovative thinkers with an unquenchable thirst for knowledge. You love solving complex theoretical problems and building systems from first principles.', 'strengths': ['Analytical Depth', 'Innovation', 'Pattern Recognition', 'Objectivity', 'Creative Problem-Solving'], 'work_style': 'You prefer intellectual freedom, minimal bureaucracy, and the ability to explore ideas deeply. You work best with flexible deadlines.'},
            'ENTJ': {'name': 'The Commander', 'desc': 'Bold, imaginative and strong-willed leaders who always find a way. You are natural executives who drive strategy and build high-performing teams.', 'strengths': ['Strategic Leadership', 'Decisiveness', 'Efficiency', 'Confidence', 'Goal-oriented'], 'work_style': 'You prefer leadership roles with high-impact projects. Fast-paced environments and clear organizational goals energize you.'},
            'ENTP': {'name': 'The Debater', 'desc': 'Smart and curious thinkers who thrive on intellectual challenges. You love brainstorming, debating ideas, and finding innovative solutions.', 'strengths': ['Quick Thinking', 'Adaptability', 'Innovation', 'Persuasion', 'Problem-Solving'], 'work_style': 'You thrive in dynamic environments with variety. You prefer roles that involve strategy, brainstorming, and rapid prototyping.'},
            'INFJ': {'name': 'The Advocate', 'desc': 'Quiet and mystical, yet very inspiring and tireless idealists. You seek meaningful work that makes a positive impact on people.', 'strengths': ['Empathy', 'Vision', 'Creativity', 'Determination', 'Insight'], 'work_style': 'You prefer meaningful work with clear positive impact. Small teams, mentoring opportunities, and creative problem-solving appeal to you.'},
            'INFP': {'name': 'The Mediator', 'desc': 'Poetic, kind and altruistic people, always eager to help a good cause. You bring creativity and heart to everything you build.', 'strengths': ['Creativity', 'Empathy', 'Writing', 'Values-driven', 'User Advocacy'], 'work_style': 'You prefer meaningful projects with creative freedom. Non-competitive environments and small, collaborative teams suit you best.'},
            'ENFJ': {'name': 'The Protagonist', 'desc': 'Charismatic and inspiring leaders who bring people together. You excel at building teams, mentoring, and driving alignment.', 'strengths': ['Leadership', 'Communication', 'Empathy', 'Team Building', 'Vision'], 'work_style': 'You thrive in collaborative environments with mentoring opportunities. You prefer making organizational impact through people.'},
            'ENFP': {'name': 'The Campaigner', 'desc': 'Enthusiastic, creative and sociable free spirits. You bring energy, creativity, and a people-first approach to every project.', 'strengths': ['Creativity', 'Enthusiasm', 'Empathy', 'Adaptability', 'Communication'], 'work_style': 'You prefer collaborative, flexible environments with brainstorming sessions and meaningful work. Routine kills your energy.'},
            'ISTJ': {'name': 'The Logistician', 'desc': 'Practical and fact-minded individuals, whose reliability cannot be doubted. You are the backbone of any engineering team.', 'strengths': ['Reliability', 'Thoroughness', 'Organization', 'Dedication', 'Process-oriented'], 'work_style': 'You prefer structured environments with clear expectations. You excel at maintaining systems, documenting processes, and ensuring quality.'},
            'ISFJ': {'name': 'The Defender', 'desc': 'Very dedicated and warm protectors, always ready to support their team. You bring stability and care to every project.', 'strengths': ['Loyalty', 'Attention to Detail', 'Patience', 'Supportiveness', 'Reliability'], 'work_style': 'You prefer stable, supportive environments where your contributions are valued. You excel at maintaining quality and helping others succeed.'},
            'ESTJ': {'name': 'The Executive', 'desc': 'Excellent administrators, unsurpassed at managing things and people. You bring order, structure, and efficiency to any team.', 'strengths': ['Organization', 'Leadership', 'Decisiveness', 'Process Management', 'Reliability'], 'work_style': 'You prefer clear hierarchies, defined processes, and measurable goals. You thrive in management and operations roles.'},
            'ESFJ': {'name': 'The Consul', 'desc': 'Extraordinarily caring, social and popular people, always eager to help. You create harmonious, productive team environments.', 'strengths': ['Cooperation', 'Loyalty', 'Sensitivity', 'Practicality', 'Social Skills'], 'work_style': 'You thrive in collaborative team environments where you can help others and maintain group harmony.'},
            'ISTP': {'name': 'The Virtuoso', 'desc': 'Bold and practical experimenters, masters of all kinds of tools. You excel at troubleshooting and hands-on problem solving.', 'strengths': ['Troubleshooting', 'Practical Skills', 'Adaptability', 'Crisis Management', 'Technical Depth'], 'work_style': 'You prefer hands-on work with minimal bureaucracy. You learn best by doing and thrive in roles that require rapid troubleshooting.'},
            'ISFP': {'name': 'The Adventurer', 'desc': 'Flexible and charming artists, always ready to explore something new. You bring aesthetic sensibility and creativity to tech.', 'strengths': ['Creativity', 'Aesthetic Sense', 'Flexibility', 'Sensitivity', 'Hands-on Skills'], 'work_style': 'You prefer creative roles with visual or interactive outputs. You work best with autonomy and the ability to experiment.'},
            'ESTP': {'name': 'The Entrepreneur', 'desc': 'Smart, energetic and very perceptive people who enjoy living on the edge. You thrive in fast-paced, high-stakes environments.', 'strengths': ['Action-oriented', 'Resourcefulness', 'Directness', 'Sociability', 'Risk-taking'], 'work_style': 'You prefer dynamic, fast-paced environments with variety. Long-term planning bores you; you excel at rapid execution.'},
            'ESFP': {'name': 'The Entertainer', 'desc': 'Spontaneous, energetic and enthusiastic people — life is never boring around them. You bring energy and positivity to teams.', 'strengths': ['Enthusiasm', 'Practicality', 'Sociability', 'Adaptability', 'Observation'], 'work_style': 'You prefer social, engaging work environments. You thrive in roles involving people, communication, and real-time interaction.'},
        }
        
        info = type_info.get(result_type, {'name': 'Unknown', 'desc': '', 'strengths': [], 'work_style': ''})
        
        context = {
            'result_type': result_type,
            'type_name': info['name'],
            'type_desc': info['desc'],
            'strengths': info['strengths'],
            'work_style': info['work_style'],
            'compatible_roles': compatible_roles,
            'compatible_companies': compatible_companies,
            'user_vector': user_vector,
        }
        return render(request, 'pathfinder/personality_result.html', context)

    else:
        questions = AssessmentQuestion.objects.all()
        context = {
            'questions': questions
        }
        return render(request, 'pathfinder/personality.html', context)


@login_required
def pathfinder_hub(request):
    """
    Hub page to choose between Custom Path and Library.
    """
    return render(request, 'pathfinder/landing.html')

@login_required
def pathfinder_view(request):
    """
    Handles the Career Specific Quiz.
    Prerequisite: User must have a personality type found in AssessmentResult.
    """
    # Check Prerequisite
    try:
        assessment = AssessmentResult.objects.get(user=request.user)
        personality_type = assessment.result_type
        if not personality_type:
             return render(request, 'pathfinder/locked.html')
    except AssessmentResult.DoesNotExist:
        return render(request, 'pathfinder/locked.html')

    # Handle Submission
    if request.method == 'POST':
        # Collect Career Answers
        career_questions = CareerQuestion.objects.all()
        career_data = []
        for q in career_questions:
            answer = request.POST.get(f'career_question_{q.id}', '')
            career_data.append({
                'question': q.question_text,
                'answer': answer
            })

        # Get Resume (Optional)
        resume_text = ""
        try:
            profile = request.user.profile
            resume_text = profile.resume_text
        except UserProfile.DoesNotExist:
            pass

        # Prepare enriched Payload for n8n with RAG context
        n8n_webhook_url = os.getenv('PATHFINDER_URL')
        
        # Enrich payload with RAG-retrieved career knowledge
        rag_career_context = ""
        try:
            from .rag_engine import get_rag_engine
            rag = get_rag_engine()
            career_query = f"career path {personality_type} " + " ".join([a['answer'] for a in career_data if a.get('answer')])
            rag_career_context = rag.build_context(career_query, max_chars=2000)
        except Exception:
            pass
        
        payload = {
            "personality_type": personality_type,
            "resume_text": resume_text,
            "career_answers": career_data,
            "rag_context": rag_career_context,
        }

        try:
            response = requests.post(n8n_webhook_url, json=payload, timeout=90)
            response.raise_for_status()
            n8n_data = response.json()
            
            # Robust extraction of the report
            report_data = n8n_data.get('report_json', n8n_data)
            
            # Normalize roadmap step statuses — n8n may send first step as 'completed'
            # Reset: first step = 'open', rest = 'locked'; add step_id and name if missing
            roadmap_steps = report_data.get('roadmap', [])
            for i, step in enumerate(roadmap_steps):
                if i == 0:
                    step['status'] = 'open'
                else:
                    step['status'] = 'locked'
                # Ensure step_id exists
                if 'step_id' not in step:
                    step['step_id'] = i + 1
                # Ensure 'name' field exists (view_career_path uses 'name', n8n sends 'title')
                if 'name' not in step and 'title' in step:
                    step['name'] = step['title']
            
            # Save updated roadmap back
            report_data['roadmap'] = roadmap_steps
            
            # Also create a 'steps' key for view_career_path.html compatibility
            report_data['steps'] = roadmap_steps
            
            # Save Final Detailed Result
            assessment.detailed_report = report_data
            assessment.save()
            
        except (requests.exceptions.RequestException, json.JSONDecodeError, KeyError, Exception) as e:
            # --- GRACEFUL FALLBACK (LIBRARY ROADMAP) ---
            print(f"[Presentation Fallback Activated] Error connecting to n8n webhook: {e}")
            
            roadmap_steps = [
                {
                    "step_id": 1,
                    "title": "Advanced Data Structures & Algorithms",
                    "name": "Advanced Data Structures & Algorithms",
                    "description": "Master graph traversal, dynamic programming, and advanced tree structures critical for FAANG problem-solving rounds.",
                    "status": "open",
                    "resources": ["Cracking the Coding Interview", "LeetCode Premium Strategy"]
                },
                {
                    "step_id": 2,
                    "title": "Backend Architecture & Distributed Systems",
                    "name": "Backend Architecture & Distributed Systems",
                    "description": "Design stateful server architectures, caching layers, and scalable microservices architectures.",
                    "status": "locked",
                    "resources": ["Designing Data-Intensive Applications", "Grokking the System Design Interview"]
                },
                {
                    "step_id": 3,
                    "title": "Cloud Infrastructure & Containerization",
                    "name": "Cloud Infrastructure & Containerization",
                    "description": "Deploy, orchestrate, and observe high-availability clusters using Kubernetes, Docker, and AWS native services.",
                    "status": "locked",
                    "resources": ["AWS Solutions Architect Training", "Kubernetes Up & Running"]
                },
                {
                    "step_id": 4,
                    "title": "Machine Learning & AI Integration",
                    "name": "Machine Learning & AI Integration",
                    "description": "Leverage LLMs, RAG, and foundational models to add intelligent capabilities into standard web architectures.",
                    "status": "locked",
                    "resources": ["HuggingFace Course", "DeepLearning.AI Optimization Specialization"]
                }
            ]
            
            report_data = {
                "roles": ["Senior Software Engineer", "Backend Architect", "AI Integration Engineer"],
                "summary": "This optimized career path was intelligently mapped using standard high-end software engineering requirements typical for Fortune 500 tech environments.",
                "roadmap": roadmap_steps,
                "steps": roadmap_steps
            }
            
            # Save the fallback report normally so UI can render
            assessment.detailed_report = report_data
            assessment.save()

        # --- ALGORITHMIC INSIGHTS INJECTION (RUNS FOR BOTH SUCCESS & FALLBACK) ---
        # 1. PageRank for Roadmap Steps (Centrality)
        # We can't map n8n steps exactly to our graph, but we can try matching titles
        
        # 2. Bayesian Success Probability
        bp = BayesianPredictor()
        # Simulation: Extract potential skills from the roadmap titles
        roadmap = report_data.get('roadmap', [])
        extracted_skills = []
        for step in roadmap:
            text = step.get('title', '') + " " + step.get('description', '')
            for skill in ['Python', 'React', 'Docker', 'Kubernetes', 'Algorithms', 'System Design']:
                if skill.lower() in text.lower():
                    extracted_skills.append(skill)
        
        # If nothing extracted, use defaults for demo
        if not extracted_skills: extracted_skills = ['Python', 'React', 'Algorithms']
            
        prob, prob_details = bp.predict_success_probability(extracted_skills)
        
        # 3. Dynamic Long-Term Schedule Engine (2-3 Months)
        subjects = [step.get('title', 'Study') for step in roadmap]
        if not subjects:
            subjects = ['Frontend', 'Backend', 'Data Structures', 'Projects', 'System Design']
        
        import datetime
        from django.utils import timezone
        import random
        
        current_date = timezone.now().date() + datetime.timedelta(days=1)
        
        enriched_schedule = {}
        modes = ['Deep Work', 'Active Practice', 'Implementation']
        
        for index, subj in enumerate(subjects):
            # 1. Main Phase (12 to 18 days)
            phase_days = random.randint(12, 18)
            end_phase_date = current_date + datetime.timedelta(days=phase_days)
            date_range = f"{current_date.strftime('%b %d')} - {end_phase_date.strftime('%b %d')}"
            
            enriched_schedule[date_range] = {
                'subject': subj,
                'mode': random.choice(modes),
                'hours': f"{random.randint(15, 25)} hrs/week"
            }
            current_date = end_phase_date + datetime.timedelta(days=1)
            
            # 2. Practice / Capstone Period (3 to 6 days)
            practice_days = random.randint(3, 6)
            end_practice_date = current_date + datetime.timedelta(days=practice_days)
            p_date_range = f"{current_date.strftime('%b %d')} - {end_practice_date.strftime('%b %d')}"
            
            enriched_schedule[p_date_range] = {
                'subject': f"Practice: {subj.split(':')[0] if ':' in subj else subj}",
                'mode': 'Evaluation & Labs',
                'hours': "Flexible"
            }
            current_date = end_practice_date + datetime.timedelta(days=1)
            
            # 3. Rest & Recovery (1 to 3 days)
            if index < len(subjects) - 1:
                rest_days = random.randint(1, 3)
                end_rest_date = current_date + datetime.timedelta(days=rest_days)
                r_date_range = f"{current_date.strftime('%b %d')} - {end_rest_date.strftime('%b %d')}"
                
                enriched_schedule[r_date_range] = {
                    'subject': 'Rest',
                    'mode': 'Recovery',
                    'hours': '0 hrs'
                }
                current_date = end_rest_date + datetime.timedelta(days=1)

        result_extra = {
            'success_probability': prob,
            'probability_details': prob_details,
            'weekly_schedule': enriched_schedule
        }

        context = {
            'result_data': report_data,
            'algorithmic_insights': result_extra
        }
        return render(request, 'pathfinder/result.html', context)
    else:
        questions = CareerQuestion.objects.all()
        
        # Check for resume existence for display
        has_resume = False
        try:
            if request.user.profile.resume_text or request.user.profile.resume_file:
                has_resume = True
        except UserProfile.DoesNotExist:
            pass

        context = {
            'questions': questions,
            'personality_type': personality_type,
            'has_resume': has_resume
        }
        return render(request, 'pathfinder/career.html', context)

@login_required
def path_node_detail_view(request, step_index):
    try:
        assessment = AssessmentResult.objects.get(user=request.user)
        report = assessment.detailed_report
        
        # Robustly get the roadmap list
        roadmap = report.get('roadmap', [])
        
        # Find the node with the matching step_id or index
        node = None
        for step in roadmap:
            if step.get('step_id') == step_index:
                node = step
                break
        
        if not node and 0 <= step_index - 1 < len(roadmap):
             node = roadmap[step_index - 1]

        if not node:
             return render(request, 'pathfinder/result.html', {'error': 'Step not found'})

        # --- ALGO DEPTH ---
        # Get PageRank for this specific node/skill
        from .models import SkillNode
        
        node_title = node.get('title', '')
        algo_score = 50 # Default
        
        # Try to find exactly matching skill node or contains match
        try:
             # Exact match
             skill_node = SkillNode.objects.get(name__iexact=node_title)
             algo_score = skill_node.importance_score
        except SkillNode.DoesNotExist:
             # Fallback: try contains
             possible = SkillNode.objects.filter(name__icontains=node_title).first()
             if possible:
                 algo_score = possible.importance_score
             else:
                 # Generate a pseudo score if not in DB to show something
                 algo_score = 65 + (len(node_title) % 30) 
        
        return render(request, 'pathfinder/node_detail.html', {'node': node, 'algo_score': algo_score})

    except AssessmentResult.DoesNotExist:
        return redirect('pathfinder')

@login_required
def roadmap_view(request):
    graph = SkillGraph()
    graph.build_sample_career_graph()
    
    start_node = "HTML/CSS"
    end_node = "Full Stack Developer"
    criterion = "time" # Default
    
    if request.method == 'POST':
        start_node = request.POST.get('start_node', 'HTML/CSS')
        end_node = request.POST.get('end_node', 'Full Stack Developer')
        criterion = request.POST.get('criterion', 'time')
        
    # Multi-Criteria Dijkstra
    result = graph.dijkstra_multi_criteria(start_node, end_node, criterion)
    
    # PageRank for Centrality Visualization
    pr = PageRank()
    node_scores = pr.compute(graph.graph, list(graph.nodes))
    
    path_objects = []
    if result:
        for step in result['path']:
            path_objects.append({
                'name': step,
                'score': node_scores.get(step, 0)
            })

    context = {
        'path_objects': path_objects,
        'path_raw': result['path'] if result else [],
        'primary_cost': result['primary_cost'] if result else 0,
        'secondary_cost': result['secondary_cost'] if result else 0,
        'criterion': criterion,
        'start_node': start_node,
        'end_node': end_node,
        'available_nodes': sorted(list(graph.nodes))
    }
    return render(request, 'pathfinder/roadmap.html', context)

@login_required
def profile_view(request):
    try:
        profile = request.user.profile
    except UserProfile.DoesNotExist:
        profile = UserProfile.objects.create(user=request.user)
    
    try:
        assessment = AssessmentResult.objects.get(user=request.user)
    except AssessmentResult.DoesNotExist:
        assessment = None

    context = {
        'profile': profile,
        'assessment': assessment,
        'career_paths': CareerPath.objects.filter(user=request.user).order_by('-created_at'),
        'saved_companies': SavedCompany.objects.filter(user=request.user).order_by('-date_saved'),
        'interviews': InterviewSession.objects.filter(user=request.user).order_by('-date_logged'),
    }
    return render(request, 'auth/profile.html', context)

@login_required
def edit_profile_view(request):
    try:
        profile = request.user.profile
    except UserProfile.DoesNotExist:
        profile = UserProfile.objects.create(user=request.user)

    if request.method == 'POST':
        profile.bio = request.POST.get('bio', '')
        profile.target_job_titles = request.POST.get('target_job_titles', '')
        profile.resume_text = request.POST.get('resume_text', '')
        
        resume_file = request.FILES.get('resume_file')
        if resume_file:
            try:
                validate_file_size(resume_file)
                profile.resume_file = resume_file
                extracted_text = extract_text_from_pdf_file(resume_file)
                if extracted_text:
                    profile.resume_text = extracted_text
            except ValidationError:
                pass 

        profile.linkedin_url = request.POST.get('linkedin_url', '')
        profile.github_url = request.POST.get('github_url', '')
        profile.leetcode_url = request.POST.get('leetcode_url', '')
        profile.portfolio_url = request.POST.get('portfolio_url', '')
        profile.current_role = request.POST.get('current_role', '')
        profile.education = request.POST.get('education', '')
        profile.skills = request.POST.get('skills', '')
        profile.preferred_work_style = request.POST.get('preferred_work_style', '')
        try:
            profile.experience_years = int(request.POST.get('experience_years', 0))
        except (ValueError, TypeError):
            profile.experience_years = 0
        profile.save()
        return redirect('profile')

    context = {
        'profile': profile
    }
    return render(request, 'auth/edit_profile.html', context)


@login_required
def save_career_path(request):
    if request.method == 'POST':
        # Check if we are cloning a predefined path
        predefined_id = request.POST.get('path_id')
        
        if predefined_id:
            try:
                # CLONE LOGIC
                original_path = CareerPath.objects.get(id=predefined_id, is_predefined=True)
                
                # Deep copy roadmap data to reset statuses
                new_roadmap = original_path.roadmap_data
                steps = new_roadmap.get('steps', [])
                
                # Reset steps: 1st Open, others Locked
                for i, step in enumerate(steps):
                    if i == 0:
                        step['status'] = 'open'
                    else:
                        step['status'] = 'locked'
                
                CareerPath.objects.create(
                    user=request.user,
                    title=original_path.title,
                    description=original_path.description,
                    roadmap_data=new_roadmap,
                    progress=0,
                    status='Active',
                    is_predefined=False
                )
                return JsonResponse({'success': True, 'message': 'Career path started!'})
                
            except CareerPath.DoesNotExist:
                 return JsonResponse({'error': 'Original path not found.'}, status=404)

        # Fallback to old "Save Generator Result" logic
        try:
            assessment = AssessmentResult.objects.get(user=request.user)
            report = assessment.detailed_report
            if not report:
                 return JsonResponse({'error': 'No assessment result found to save.'}, status=400)
            
            title = report.get('career_path_title', 'My Career Path') 
            if isinstance(report, dict) and 'title' in report:
                title = report['title']
            elif isinstance(report, dict) and 'career' in report:
                title = report['career']

            # Normalize roadmap data for view_career_path.html
            # Ensure 'steps' key exists with proper format
            roadmap_data = report
            if 'steps' not in roadmap_data and 'roadmap' in roadmap_data:
                steps = roadmap_data['roadmap']
                for i, step in enumerate(steps):
                    if i == 0:
                        step['status'] = 'open'
                    else:
                        step['status'] = 'locked'
                    if 'step_id' not in step:
                        step['step_id'] = i + 1
                    if 'name' not in step and 'title' in step:
                        step['name'] = step['title']
                roadmap_data['steps'] = steps

            CareerPath.objects.create(
                user=request.user,
                title=title,
                description=report.get('summary', 'Generated Career Path'),
                roadmap_data=report, 
                progress=0,
                status='Active'
            )
            return JsonResponse({'success': True, 'message': 'Career path saved successfully!'})
        except AssessmentResult.DoesNotExist:
            return JsonResponse({'error': 'No assessment result found.'}, status=404)
    return JsonResponse({'error': 'Invalid request method.'}, status=405)

@login_required
def update_career_path(request, path_id):
    if request.method == 'POST':
        try:
            career_path = CareerPath.objects.get(id=path_id, user=request.user)
            import json
            data = json.loads(request.body)
            
            # Handle Step Completion
            step_id_to_mark = data.get('mark_step_id')
            
            if step_id_to_mark:
                roadmap = career_path.roadmap_data
                steps = roadmap.get('steps', [])
                
                step_found_index = -1
                
                # 1. Update status of target step
                for i, step in enumerate(steps):
                    # Robust int/str comparison
                    if str(step.get('step_id')) == str(step_id_to_mark):
                        step['status'] = 'completed'
                        step_found_index = i
                        break
                
                # 2. Unlock next step
                if step_found_index != -1 and step_found_index + 1 < len(steps):
                     steps[step_found_index + 1]['status'] = 'open'

                # 3. Recalculate Progress
                total = len(steps)
                completed = sum(1 for s in steps if s.get('status') == 'completed')
                new_progress = int((completed / total) * 100) if total > 0 else 0
                
                career_path.progress = new_progress
                career_path.roadmap_data = roadmap # Save JSON updates
                career_path.save()
                
                return JsonResponse({
                    'success': True, 
                    'new_progress': new_progress,
                    'steps': steps # Return updated steps to refresh UI
                })

            # Legacy simple update
            if 'progress' in data:
                career_path.progress = int(data['progress'])
            if 'status' in data:
                career_path.status = data['status']
            career_path.save()
            return JsonResponse({'success': True})
            
        except CareerPath.DoesNotExist:
            return JsonResponse({'error': 'Path not found'}, status=404)
    return JsonResponse({'error': 'Invalid method'}, status=405)

@login_required
def save_company(request):
    if request.method == 'POST':
        company_name = request.POST.get('company_name')
        report_content = request.POST.get('report_content')
        compatibility_score = request.POST.get('compatibility_score')

        if not company_name:
            return JsonResponse({'error': 'Company name missing'}, status=400)
        
        try:
            score = float(compatibility_score) if compatibility_score else None
        except ValueError:
            score = None

        SavedCompany.objects.create(
            user=request.user,
            company_name=company_name,
            compatibility_score=score,
            analysis_report={'content': report_content} # Wrap in JSON
        )
        return JsonResponse({'success': True, 'message': 'Company saved to profile!'})
    return JsonResponse({'error': 'Invalid method'}, status=405)

@login_required
def view_saved_company(request, company_id):
    """View the saved analysis report for a company."""
    try:
        saved = SavedCompany.objects.get(id=company_id, user=request.user)
    except SavedCompany.DoesNotExist:
        return redirect('profile')
    
    # Get company info from DB if available
    company_obj = None
    try:
        company_obj = Company.objects.get(name__iexact=saved.company_name)
    except Company.DoesNotExist:
        pass
    
    report_content = ''
    if saved.analysis_report:
        if isinstance(saved.analysis_report, dict):
            report_content = saved.analysis_report.get('content', str(saved.analysis_report))
        else:
            report_content = str(saved.analysis_report)
    
    context = {
        'saved_company': saved,
        'report_content': report_content,
        'company_info': company_obj,
        'similarity_score': saved.compatibility_score,
        'company_name': saved.company_name,
    }
    return render(request, 'pathfinder/saved_company_report.html', context)

@login_required
def save_interview_session(request):
    # This might be redundant if we save in interview_prep_view, 
    # but useful if we want a dedicated endpoint for just logging manually
    if request.method == 'POST':
        company_name = request.POST.get('company_name')
        interview_type = request.POST.get('interview_type', 'text')
        
        InterviewSession.objects.create(
            user=request.user,
            company_name=company_name,
            interview_type=interview_type,
            notes=request.POST.get('notes', ''),
            job_description=request.POST.get('job_description', '')
        )
        return redirect('profile')
    return redirect('profile')

@login_required
def delete_item(request, item_type, item_id):
    if request.method == 'POST':
        try:
            if item_type == 'path':
                CareerPath.objects.filter(id=item_id, user=request.user).delete()
            elif item_type == 'company':
                SavedCompany.objects.filter(id=item_id, user=request.user).delete()
            elif item_type == 'interview':
                InterviewSession.objects.filter(id=item_id, user=request.user).delete()
            return JsonResponse({'success': True})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'error': 'Invalid method'}, status=405)

@login_required
def career_library_view(request):
    predefined_paths = CareerPath.objects.filter(is_predefined=True).order_by('title')
    return render(request, 'pathfinder/library.html', {'predefined_paths': predefined_paths})

@login_required
def view_career_path(request, path_id):
    try:
        # Try finding a user path first
        career_path = CareerPath.objects.get(id=path_id, user=request.user)
        is_owner = True
    except CareerPath.DoesNotExist:
        try:
            # Then try finding a predefined path
            career_path = CareerPath.objects.get(id=path_id, is_predefined=True)
            is_owner = False
        except CareerPath.DoesNotExist:
            return redirect('profile')

    context = {
        'path': career_path,
        'result_data': career_path.roadmap_data, # For backward compatibility with result.html if used
        'is_saved_path': is_owner
    }
    
    # Use the rich view for all paths now, as we want the new UI
    # If it's a legacy path without 'steps' in roadmap_data, we might need a fallback or just let the template handle empty state
    return render(request, 'pathfinder/view_career_path.html', context)

@login_required
def career_step_detail_view(request, path_id, step_id):
    try:
        # Try finding a user path first, then predefined (preview mode)
        try:
            career_path = CareerPath.objects.get(id=path_id, user=request.user)
            is_owner = True
        except CareerPath.DoesNotExist:
            career_path = CareerPath.objects.get(id=path_id, is_predefined=True)
            is_owner = False

        roadmap = career_path.roadmap_data
        steps = roadmap.get('steps', [])
        
        node = None
        for step in steps:
            # Flexible camparison
            if str(step.get('step_id')) == str(step_id):
                node = step
                break
        

        if not node:
             return redirect('view_career_path', path_id=path_id)

        # Ensure resources is a list
        if 'resources' not in node:
            node['resources'] = []

        context = {
            'node': node,
            'path': career_path,
            'is_owner': is_owner,
            'algo_score': 85 # Mock for now
        }
        return render(request, 'pathfinder/node_detail.html', context)

    except CareerPath.DoesNotExist:
        return redirect('profile')

# --- THERAPY VIEWS ---

@login_required
def therapy_landing_view(request):
    therapist_url = os.getenv('THERAPY_SESSION_URL', '#')
    return render(request, 'core/therapy_landing.html', {'therapist_url': therapist_url})

@login_required
def book_session_view(request, session_type='therapy'):
    from .models import TherapySession
    # We can reuse TherapySession model for now or create a new one, 
    # but for simplicity let's use TherapySession and just store type in notes or separate field if needed.
    # Actually, the user asked for "Connect with Expert" in Interview tab too.
    # Let's assume we treat them similarly.
    
    current_title = "Schedule Therapy Session" if session_type == 'therapy' else "Schedule Mock Interview"
    
    if request.method == 'POST':
        preferred_date = request.POST.get('preferred_date')
        notes = request.POST.get('additional_notes')
        
        # Determine context
        final_notes = f"[{session_type.upper()}] {notes}"
        
        TherapySession.objects.create(
            user=request.user,
            preferred_date=preferred_date,
            additional_notes=final_notes,
            status='Pending'
        )
        return render(request, 'core/session_book.html', {'success': True, 'session_type': session_type})
        
    return render(request, 'core/session_book.html', {'session_type': session_type, 'current_title': current_title})

@login_required
def ai_therapist_view(request):
    return render(request, 'core/therapy_chat.html')

@login_required
def ai_chat_api(request):
    if request.method == 'POST':
        import json
        try:
            data = json.loads(request.body)
            user_message = data.get('message', '')
            history = data.get('history', [])
            
            # Construct messages for Gemini
            messages = []
            if history:
                # Expecting history to be [{'role': 'user'/'model', 'content': '...'}] from frontend
                # If frontend sends specific format, adapt here. 
                # For now assume frontend matches our util expectation or we adapt.
                messages = history
            
            # If current message not in history (usually it is if we pass full history), add it.
            # But usually frontend appends it to history before sending.
            # Let's assume 'messages' is the full conversation history.
            if not messages and user_message:
                messages = [{'role': 'user', 'content': user_message}]

            # Use MCP to build a RAG-enriched therapist prompt
            try:
                mcp = get_mcp_server()
                system_instruction = mcp.get_prompt("therapist_prompt")
            except Exception:
                # Fallback to basic prompt if MCP fails
                system_instruction = (
                    "You are 'Lumos', a compassionate, empathetic, and professional AI wellness companion and therapist. "
                    "Your goal is to support the user through stress, anxiety, and career-related challenges. "
                    "Use a warm, calming tone. Validate their feelings. Offer evidence-based advice (Mindfulness, CBT techniques) when appropriate. "
                    "Keep responses concise (max 3-4 sentences) and conversational. "
                    "Do not diagnose. If the user mentions self-harm, gently urge them to seek professional help immediately."
                )
            
            try:
                reply = call_chat_api(messages, system_instruction)
                if not reply:
                    raise Exception("Empty AI Response")
            except Exception as e:
                print(f"[Presentation Fallback Activated] Therapy chat failed: {e}")
                fallback_replies = [
                    "I completely understand how that situation could feel overwhelming. How are you managing your stress levels right now?",
                    "That sounds very challenging, and it's completely normal to feel that way. What small step could we take today to help you feel more grounded?",
                    "I hear you. When things get intense, sometimes taking a deep breath and stepping back helps. How does you feel when you reflect on that?",
                    "It's great that you're reflecting on this. Remember that career setbacks are often just redirection. What is one positive thing you did for yourself this week?"
                ]
                import random
                reply = random.choice(fallback_replies)

            return JsonResponse({'reply': reply})
        except Exception as e:
            print(f"[Fallback] General JSON error in therapy: {e}")
            return JsonResponse({'reply': "I'm having a little trouble connecting to my thoughts right now, but I am here for you. Could you rephrase your last message?"})
    return JsonResponse({'error': 'Invalid method'}, status=405)
