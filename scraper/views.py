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
from .utils import extract_text_from_pdf_file, validate_file_size, send_to_n8n_webhook
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

        company_name = request.POST.get('company_name', '')
        job_description = request.POST.get('job_description', '')
        
        similarity_score, matching_keywords, missing_keywords = calculate_similarity_score(resume_text, job_description)
        personality_type = "Not Available"
        try:
            assessment = AssessmentResult.objects.get(user=request.user)
            personality_type = assessment.result_type
        except AssessmentResult.DoesNotExist:
            pass

        n8n_webhook_url = os.getenv('COMPANY_SCRAPER_URL')
        payload = {
            "resume": resume_text,
            "company": company_name,
            "personality": personality_type,
            "similarity_score": similarity_score,
            "job_description": job_description
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

                # Final Cleanup: Remove surrounding quotes if they exist (rare but possible after str() conversion or JSON behavior)
                if isinstance(report_text, str):
                    report_text = report_text.strip()
                    if (report_text.startswith('"') and report_text.endswith('"')) or \
                       (report_text.startswith("'") and report_text.endswith("'")):
                        report_text = report_text[1:-1]

            except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
                report_text = f"Error: Could not parse the AI's response. {e}."
            
            # --- ALGORITHMIC INSIGHTS FOR MATCHER ---
            # Using Apriori for "Missing Skill Suggestions"
            # In production, we'd mine real DB. Here we use mock.
            ap = AprioriGenerator()
            rules = ap.generate_rules() # Generate general rules
            
            # Filter recommendations based on what the user currently has (matching_keywords)
            recommendations = []
            user_skills = set(matching_keywords)
            for rule in rules:
                if rule['from'] in user_skills and rule['to'] not in user_skills:
                    recommendations.append(rule)
            
            context = {
                'report_content': report_text,
                'similarity_score': similarity_score,
                'matching_keywords': matching_keywords,
                'missing_keywords': missing_keywords,
                'show_score': True if job_description else False,
                'company_name': company_name,
                'job_description': job_description,
                'skill_recommendations': recommendations[:3] # Show top 3 derived rules
            }
            return render(request, 'pathfinder/report.html', context)
            
        except requests.exceptions.RequestException as e:
            error_message = f"An error occurred while connecting to the analysis engine: {e}"
            return HttpResponseServerError(error_message)

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
            return redirect('home')
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

        n8n_webhook_url = os.getenv('INTERVIEW_COPILOT_URL')
        payload = {
            "company_name": company_name,
            "resume": resume_text
        }

        # Get Recommendations
        recommender = RecommenderSystem()
        recommendations = recommender.get_recommendations(company_name)

        try:
            response = requests.post(n8n_webhook_url, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Append recommendations to the response
            data['recommendations'] = recommendations
            
            return JsonResponse(data)

        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            return JsonResponse({'error': f'An error occurred: {e}'}, status=500)

    context = {
        'voice_interview_url': os.getenv('VOICE_INTERVIEW_URL', '#')
    }
    return render(request, 'pathfinder/interview_prep.html', context)

def interview_chat_view(request):
    if request.method == 'POST':
        user_message = request.POST.get('user_message', '')
        n8n_webhook_url = os.getenv('INTERVIEW_CHAT_URL')
        payload = {
            "history": f"User: {user_message}"
        }

        try:
            response = requests.post(n8n_webhook_url, json=payload, timeout=30)
            response.raise_for_status()
            return JsonResponse(response.json())
        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            return JsonResponse({'error': f'An error occurred: {e}'}, status=500)

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
        
        # Redirect to Profile to see result
        return redirect('profile')

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

        # Prepare Payload for n8n
        n8n_webhook_url = os.getenv('PATHFINDER_URL')
        payload = {
            "personality_type": personality_type,
            "resume_text": resume_text,
            "career_answers": career_data
        }

        try:
            response = requests.post(n8n_webhook_url, json=payload, timeout=90)
            response.raise_for_status()
            n8n_data = response.json()
            
            # Robust extraction of the report
            report_data = n8n_data.get('report_json', n8n_data)
            
            # Save Final Detailed Result
            assessment.detailed_report = report_data
            assessment.save()
            
            # --- ALGORITHMIC INSIGHTS INJECTION ---
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
            
            # 3. Simulated Annealing Schedule
            # Create a schedule based on the first 4 modules of the roadmap
            subjects = [step.get('title', 'Study') for step in roadmap[:4]]
            if len(subjects) < 2: subjects = ['Frontend', 'Backend', 'Data Structures', 'Projects']
            
            sa = SimulatedAnnealingScheduler(subjects)
            schedule, energy = sa.optimize()
            
            result_extra = {
                'success_probability': prob,
                'probability_details': prob_details,
                'weekly_schedule': schedule,
                'energy_score': energy
            }

            context = {
                'result_data': report_data,
                'algorithmic_insights': result_extra
            }
            return render(request, 'pathfinder/result.html', context)

        except (requests.exceptions.RequestException, json.JSONDecodeError, KeyError) as e:
            return HttpResponseServerError(f"Error processing AI report: {e}")

    # Render Career Quiz
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
        # Get PageRank for this specific node/skill if possible
        # Mocking mapping for demo
        pr = PageRank()
        # Create a mock graph just to get context, or reuse main graph
        # For now, just generate a demo score based on node length to be deterministic
        # In prod, this would look up the node in the centralized Graph DB
        algo_score = 65 + (len(node.get('title', '')) % 30) 
        
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
            except ValidationError as e:
                # In a real app, add a message. For now, we might just ignore or let it fail?
                # Better to just not save the file if invalid, or let Django form handle it.
                # Since we are manual, let's just print or ignore for this quick impl, or add a message.
                pass 

        profile.linkedin_url = request.POST.get('linkedin_url', '')
        profile.github_url = request.POST.get('github_url', '')
        profile.leetcode_url = request.POST.get('leetcode_url', '')
        profile.save()
        return redirect('profile')

    context = {
        'profile': profile
    }
    return render(request, 'auth/edit_profile.html', context)


@login_required
def save_career_path(request):
    if request.method == 'POST':
        try:
            assessment = AssessmentResult.objects.get(user=request.user)
            report = assessment.detailed_report
            if not report:
                 return JsonResponse({'error': 'No assessment result found to save.'}, status=400)
            
            # Extract relevant info from report
            # Assuming report structure, adjust as needed based on n8n output
            title = report.get('career_path_title', 'My Career Path') 
            if isinstance(report, dict) and 'title' in report:
                title = report['title']
            elif isinstance(report, dict) and 'career' in report:
                title = report['career']

            CareerPath.objects.create(
                user=request.user,
                title=title,
                description=report.get('summary', 'Generated Career Path'),
                roadmap_data=report, # Save the whole report as the roadmap data
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
            # Update progress or status
            import json
            data = json.loads(request.body)
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
    
    # Render the specific detail view for standard paths
    if career_path.is_predefined:
        return render(request, 'pathfinder/view_career_path.html', context)
        
    return render(request, 'pathfinder/result.html', context)
