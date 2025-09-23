import requests

url = "https://api-d7b62b.stack.tryrelevance.com/latest/agents/trigger"
# payload = { "agent_id": "4c91119e-1484-456e-94c1-b0b061a6b151", "message": { "role": "user", "content": "flipkart" } }

payload = {    "agent_id": "4c91119e-1484-456e-94c1-b0b061a6b151",    "message": {        "role": "user",        "content": "https://www.linkedin.com/company/flipkart/"   }}

headers = {"Authorization": "238349be-dd02-4cf3-bd19-510b1301c48c:sk-NWFiMTI4ZGMtODI0NS00OWE5LThjZWYtYWFmYTMyMmI5YTE2"}
response = requests.post(url, json=payload, headers=headers)
print(response.json())

job_id = response.json()['job_info']['job_id']  # <-- This is where job_id is defined



import time
import requests

# Step 1: Trigger the agent (already done)
# job_id = ... (from previous response)

# Step 2: Poll for result
status_url = f"https://api-d7b62b.stack.tryrelevance.com/latest/jobs/{job_id}"
# headers = {"Authorization": "YOUR_PROJECT_ID:YOUR_SECRET_KEY"}

# while True:
#     result = requests.get(status_url, headers=headers)
#     result_data = result.json()
#     state = result_data.get('state')
#     print(f"Current state: {state}")

#     if state in ['completed', 'failed', 'error']:
#         # Task is done, handle the result
#         print("Final result:", result_data)
#         break

#     # Wait a few seconds before checking again
#     time.sleep(3)