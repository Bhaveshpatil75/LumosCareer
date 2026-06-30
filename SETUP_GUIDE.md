# Setup Guide for LumosCareer

This guide will walk you through setting up the LumosCareer project on your local machine, from scratch.

## Prerequisites
- **Python 3.8+** installed
- **Git** installed
- **Node.js & npm** (optional, if you plan to run local n8n)
- **n8n** (either local, Docker, or cloud instance)

## Step 1: Clone the Repository
```bash
git clone <repository-url>
cd LumosCareer/lumos_career
```

## Step 2: Create a Virtual Environment
It is recommended to use a virtual environment to manage dependencies.
```bash
python -m venv venv
```
Activate the virtual environment:
- **Windows:**
  ```bash
  venv\Scripts\activate
  ```
- **macOS/Linux:**
  ```bash
  source venv/bin/activate
  ```

## Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

## Step 4: Configure Environment Variables
To make setup easy, we have provided an `.env.example` file. Copy this file and rename it to `.env` in the `lumos_career` directory.

```bash
# On Linux/macOS
cp .env.example .env

# On Windows (Command Prompt)
copy .env.example .env
```

Open the `.env` file and add your API keys and the Webhook URLs from n8n (covered in Step 6).
```env
GEMINI_API_KEY="your_gemini_api_key_here"
GROK_API_KEY="your_grok_api_key_here" (optional)

# n8n Webhook URLs (See Step 6)
COMPANY_SCRAPER_URL="http://localhost:5678/webhook/..."
PATHFINDER_URL="http://localhost:5678/webhook/..."

# Integrated Voice Agent URLs
VOICE_INTERVIEW_URL="https://anika-by-bhaveshpatil75.vercel.app"
THERAPY_SESSION_URL="https://hannibal-by-bhaveshpatil75.vercel.app"
```

## Step 5: Setup the Database
Apply Django migrations to set up your local SQLite database:
```bash
python manage.py migrate
```

*(Optional)* If you want to completely populate and seed the database with initial data and paths, we have created a single command that runs everything for you:
```bash
python seed_all.py
```
*(This automatically executes `populate_db.py`, `load_initial_data.py`, and `seed_db.py` in the correct sequence).*

## Step 6: Setting up n8n Workflows
LumosCareer relies on n8n for various background automation tasks (like pathfinding, company scraping, and interview flows). You need to import the provided workflow JSON files into your n8n instance.

1. **Start n8n**: Run your local n8n instance or log in to your n8n cloud dashboard. (To run locally: `npx n8n`).
2. **Create New Workflow**: In the n8n dashboard, click **Add Workflow**.
3. **Import from JSON**: 
   - Click the menu button (three dots) in the top right corner.
   - Select **Import from File** or simply copy the contents of the workflow `.json` files located in the **`n8n-workflows`** folder of this repository and paste them into the workflow canvas.
4. **Configure Nodes**: Make sure any nodes requiring credentials (like HTTP Requests or API nodes) are properly authenticated with your own API keys.
5. **Activate the Workflow**: Toggle the workflow to **Active** to enable it.
6. **Get Production Webhook URLs**: 
   - Open the primary Webhook node for each imported workflow.
   - Switch from "Test URL" to **"Production URL"**.
   - Copy this Production URL.
7. **Update `.env`**: Paste the copied Production URLs into your `.env` file for the respective variables (`COMPANY_SCRAPER_URL`, `PATHFINDER_URL`, etc.).

## Step 7: Run the Application
Start the Django development server:
```bash
python manage.py runserver
```

You can now access LumosCareer in your browser at: [http://127.0.0.1:8000](http://127.0.0.1:8000)

## Custom Voice Agents
LumosCareer comes with integrated conversational AI agents. If you are interested in creating your own custom voice agents, you can easily do so by referring to my dedicated repositories for this. 
- [Check out my GitHub profile (@bhaveshpatil75)](https://github.com/bhaveshpatil75) to explore the source code for the Mock Interviewer and AI Therapist agents.

## Need Help?
If you have any issues with setting up the project or run into any bugs, please write to my email:
**bhaveshpatil75@gmail.com**
