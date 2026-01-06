# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
from django.shortcuts import redirect, render
from django.http import HttpResponse, JsonResponse, HttpResponseServerError
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required
from .models import AssessmentResult, Company, AssessmentQuestion
import requests
import os
import dotenv
import json
from .algorithms import LSAEngine, SkillGraph, PersonalityClassifier, RecommenderSystem

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
        resume_text = request.POST.get('resume_text', '')
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
                report_text = data.get('content').get('parts')[0].get('text', 'No report content found.')
            except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
                report_text = f"Error: Could not parse the AI's response. {e}."
            
            context = {
                'report_content': report_text,
                'similarity_score': similarity_score,
                'matching_keywords': matching_keywords,
                'missing_keywords': missing_keywords,
                'show_score': True if job_description else False
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

        n8n_webhook_url = "http://localhost:7777/webhook/7e55999f-1503-459d-a85c-15986a018a0b"
        payload = {
            "result_type": result_type
        }

        try:
            response = requests.post(n8n_webhook_url, json=payload, timeout=60)
            response.raise_for_status()
            n8n_data = response.json()
            report_data = n8n_data['report_json']
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