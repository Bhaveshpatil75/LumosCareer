# core/views.py

from django.shortcuts import redirect, render
from django.http import HttpResponse, JsonResponse
import requests
import os
import dotenv
import json
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth import login, logout
# core/views.py

from django.shortcuts import render
from django.http import JsonResponse, HttpResponseServerError
import requests
import json

dotenv.load_dotenv()  # Load environment variables from .env file

# core/views.py - FINAL PRODUCTION VERSION

from django.shortcuts import render
from django.http import HttpResponseServerError
import requests
import json # Make sure json is imported at the top

def matcher_view(request):
    if request.method == 'POST':
        resume_text = request.POST.get('resume_text', '')
        company_name = request.POST.get('company_name', '')

        # Make sure this is your Production URL from n8n
        n8n_webhook_url =os.getenv('N8NURL')
        payload = {
            "resume": resume_text,
            "company": company_name
        }

        try:
            response = requests.post(n8n_webhook_url, json=payload, timeout=60)
            response.raise_for_status()

            # --- START: FINAL PARSING LOGIC ---
            report_text = json.loads(response.text).get('content').get('parts')[0].get('text')

            context = {
                'report_content': report_text
            }
            return render(request, 'report.html', context)

        except requests.exceptions.RequestException as e:
            error_message = f"An error occurred while connecting to the analysis engine: {e}"
            return HttpResponseServerError(error_message)

    else:
        return render(request, 'index.html')
    
def signup_view(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('home') # Redirect to the main page after signup
    else:
        form = UserCreationForm()
    return render(request, 'signup.html', {'form': form})

def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            return redirect('home') # Redirect to the main page after login
    else:
        form = AuthenticationForm()
    return render(request, 'login.html', {'form': form})

def logout_view(request):
    if request.method == 'POST':
        logout(request)
        return redirect('home') # Redirect to the main page after logout