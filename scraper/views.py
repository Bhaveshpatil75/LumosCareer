# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
from django.shortcuts import redirect, render
from django.http import HttpResponse, JsonResponse, HttpResponseServerError
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required
from .models import AssessmentResult, Company, AssessmentQuestion, UserProfile
import requests
import os
import dotenv
import json
from .algorithms import LSAEngine, SkillGraph, PersonalityClassifier, RecommenderSystem
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
                            
            except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
                report_text = f"Error: Could not parse the AI's response. {e}."
            
            context = {
                'report_content': report_text,
                'similarity_score': similarity_score,
                'matching_keywords': matching_keywords,
                'missing_keywords': missing_keywords,
                'show_score': True if job_description else False,
                'company_name': company_name,
                'job_description': job_description
            }
            return render(request, 'pathfinder/report.html', context)
            
        except requests.exceptions.RequestException as e:
            error_message = f"An error occurred while connecting to the analysis engine: {e}"
            return HttpResponseServerError(error_message)

    else:
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
        n8n_webhook_url = os.getenv('INTERVIEW_COPILOT_URL')
        payload = {
            "company_name": company_name
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

    return render(request, 'pathfinder/interview_prep.html')

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
def pathfinder_view(request):
    if request.method == 'POST':
        questions = AssessmentQuestion.objects.all()
        # Scores: E vs I, S vs N, T vs F, J vs P
        # We'll map them to a vector [-1, 1]
        # E/S/T/J = +1, I/N/F/P = -1
        raw_scores = {'EI': 0, 'SN': 0, 'TF': 0, 'JP': 0}
        total_questions = {'EI': 0, 'SN': 0, 'TF': 0, 'JP': 0}

        for question in questions:
            total_questions[question.trait] += 1
            answer = request.POST.get(f'question_{question.id}')
            if answer == 'A':
                raw_scores[question.trait] += 1
            elif answer == 'B':
                raw_scores[question.trait] -= 1

        # Normalize to [-1, 1] vector
        # Avoid division by zero if no questions for a trait
        user_vector = []
        for trait in ['EI', 'SN', 'TF', 'JP']:
            count = total_questions[trait]
            if count > 0:
                user_vector.append(raw_scores[trait] / count)
            else:
                user_vector.append(0)

        # Use Centroid Classifier
        classifier = PersonalityClassifier()
        result_type = classifier.classify(user_vector)

        AssessmentResult.objects.update_or_create(
            user=request.user,
            defaults={'result_type': result_type}
        )

        n8n_webhook_url = os.getenv('PATHFINDER_URL')
        payload = {
            "result_type": result_type
        }

        try:
            response = requests.post(n8n_webhook_url, json=payload, timeout=60)
            response.raise_for_status()
            n8n_data = response.json()
            
            # The n8n node should return a structure where 'report_json' contains the actual data
            # OR the entire body is the data. Let's robustness check.
            report_data = n8n_data.get('report_json', n8n_data)
            
            # Save the full detailed report to the database
            AssessmentResult.objects.update_or_create(
                user=request.user,
                defaults={
                    'result_type': result_type,
                    'detailed_report': report_data
                }
            )

            context = {
                'result_data': report_data 
            }
            return render(request, 'pathfinder/result.html', context)

        except (requests.exceptions.RequestException, json.JSONDecodeError, KeyError) as e:
            return HttpResponseServerError(f"Error processing AI report: {e}")

    else:
        questions = AssessmentQuestion.objects.all()
        context = {
            'questions': questions
        }
        return render(request, 'pathfinder/form.html', context)

@login_required
def path_node_detail_view(request, step_index):
    try:
        assessment = AssessmentResult.objects.get(user=request.user)
        report = assessment.detailed_report
        
        # Robustly get the roadmap list
        roadmap = report.get('roadmap', [])
        
        # Find the node with the matching step_id or index
        # We will assume step_index corresponds to step_id for now, or list index - 1
        # Let's try to match by 'step_id' if present, else use index
        node = None
        
        # Try to find by step_id first
        for step in roadmap:
            if step.get('step_id') == step_index:
                node = step
                break
        
        # If not found by ID (or if step_index is 1-based index but IDs are arbitrary), try list index
        if not node and 0 <= step_index - 1 < len(roadmap):
             node = roadmap[step_index - 1]

        if not node:
             return render(request, 'pathfinder/result.html', {'error': 'Step not found'})

        return render(request, 'pathfinder/node_detail.html', {'node': node})

    except AssessmentResult.DoesNotExist:
        return redirect('pathfinder')

@login_required
def roadmap_view(request):
    graph = SkillGraph()
    graph.build_sample_career_graph()
    
    path = []
    total_weight = 0
    start_node = "HTML/CSS" # Default start
    end_node = "Full Stack Developer" # Default end
    
    if request.method == 'POST':
        start_node = request.POST.get('start_node', 'HTML/CSS')
        end_node = request.POST.get('end_node', 'Full Stack Developer')
        
    path, total_weight = graph.dijkstra(start_node, end_node)
    
    context = {
        'path': path,
        'total_weight': total_weight,
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
        'assessment': assessment
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
