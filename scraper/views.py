# core/views.py

from django.shortcuts import render
from django.http import JsonResponse
import requests
import json

# core/views.py

from django.shortcuts import render
from django.http import JsonResponse, HttpResponseServerError
import requests
import json

def matcher_view(request):
    if request.method == 'POST':
        # --- This block runs when the user submits the form ---
        resume_text = request.POST.get('resume_text', '')
        company_name = request.POST.get('company_name', '')

        

        # --- IMPORTANT: Switch to your PRODUCTION Webhook URL ---
        n8n_webhook_url = "http://localhost:7777/webhook/93add13d-0db8-43e1-847d-5240e7253852" # <-- Update this!
        payload = {
            "resume": resume_text,
            "company": company_name
        }

        try:
            response = requests.post(n8n_webhook_url, json=payload, timeout=60) # Added a timeout
            response.raise_for_status()

            # n8n now sends the final report text directly back
            report_text = response.text

            # We pass this text to our new report.html template
            context = {
                'report_content': report_text
            }
            return render(request, 'report.html', context)

        except requests.exceptions.RequestException as e:
            # Handle errors more gracefully
            error_message = f"An error occurred while generating your report: {e}"
            return HttpResponseServerError(error_message)

    else:
        # --- This block runs for a normal visit (GET request) ---
        return render(request, 'index.html')