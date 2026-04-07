import os
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "lumos_career.settings")
django.setup()

from core.models import AssessmentQuestion, CareerQuestion

def populate():
    print("Clearing old questions data...")
    AssessmentQuestion.objects.all().delete()
    CareerQuestion.objects.all().delete()

    print("Populating AssessmentQuestions (24 MBTI questions - 6 per dimension)...")
    assessment_data = [
        # EI (Extraversion vs Introversion) — 6 questions
        ("At a party do you:", "EI", "Interact with many, including strangers", "Interact with a few, known to you"),
        ("Are you more likely to speak up in a group setting or listen quietly?", "EI", "Speak up and share ideas openly", "Listen quietly and reflect internally"),
        ("When working on a project, do you prefer:", "EI", "Brainstorming with a team in real-time", "Working independently then sharing results"),
        ("After a long day of meetings, you feel:", "EI", "Energized by the interactions", "Drained and needing alone time"),
        ("In your ideal work environment:", "EI", "Open floor plan with lots of collaboration", "Private office or quiet space for focus"),
        ("When solving a complex problem:", "EI", "You talk it through with colleagues immediately", "You think it through alone before discussing"),

        # SN (Sensing vs Intuition) — 6 questions
        ("Are you more attracted to:", "SN", "Practical, realistic solutions", "Imaginative, innovative ideas"),
        ("Do you prefer to focus on:", "SN", "The present reality and concrete facts", "Future possibilities and abstract patterns"),
        ("When learning a new technology, you prefer:", "SN", "Step-by-step tutorials with concrete examples", "Understanding the underlying theory and concepts first"),
        ("In a code review, you tend to focus on:", "SN", "Specific implementation details and edge cases", "Overall architecture and design patterns"),
        ("When reading documentation, you prefer:", "SN", "Detailed API references with examples", "High-level overviews and conceptual guides"),
        ("When planning a project roadmap:", "SN", "You create detailed timelines with specific milestones", "You outline the vision and adapt as you go"),

        # TF (Thinking vs Feeling) — 6 questions
        ("Which is a higher compliment:", "TF", "You are a very logical and analytical person", "You are a very empathetic and considerate person"),
        ("In making decisions do you rely more on:", "TF", "Objective logic, data, and facts", "Personal values, team morale, and feelings"),
        ("When giving feedback to a colleague:", "TF", "You prioritize being direct and accurate", "You prioritize being tactful and encouraging"),
        ("In a technical disagreement, you:", "TF", "Focus on which solution is objectively better", "Consider how the disagreement affects team dynamics"),
        ("When evaluating job offers:", "TF", "You create a spreadsheet comparing compensation and growth metrics", "You weigh how each role aligns with your values and life goals"),
        ("When a project fails:", "TF", "You analyze what went wrong systematically for future prevention", "You first check on how the team is feeling and offer support"),

        # JP (Judging vs Perceiving) — 6 questions
        ("Do you prefer your daily life to be:", "JP", "Structured, planned, and organized", "Flexible, spontaneous, and adaptable"),
        ("Do you like to get your work done:", "JP", "Well before the deadline with a clear plan", "In intense bursts of inspiration, often near the deadline"),
        ("Your ideal project management style is:", "JP", "Agile sprints with clear deliverables per cycle", "Kanban with flexible priorities that shift as needed"),
        ("When starting a new codebase:", "JP", "You plan the architecture thoroughly before writing any code", "You start coding a prototype and refactor as patterns emerge"),
        ("Your workspace is typically:", "JP", "Neat, organized, with everything in its place", "Creatively cluttered but you know where things are"),
        ("When unexpected requirements change:", "JP", "You feel frustrated and prefer to re-plan formally", "You embrace the change and adapt your approach on the fly"),
    ]

    for q_text, trait, choice_a, choice_b in assessment_data:
        AssessmentQuestion.objects.create(
            question_text=q_text,
            trait=trait,
            choice_a=choice_a,
            choice_b=choice_b
        )

    print("Populating CareerQuestions (12 deep career profiling questions)...")
    career_data = [
        ("What is your primary career goal in the next 1-3 years?", 1),
        ("What programming languages, tools, or frameworks are you most interested in learning?", 2),
        ("Do you prefer working on Front-end, Back-end, Full-stack, Data Science, AI/ML, DevOps, or Mobile?", 3),
        ("What kind of work environment do you prefer (e.g., startup, large tech company, remote, hybrid)?", 4),
        ("Are you more interested in deeply technical roles (IC) or leadership/management roles?", 5),
        ("What is your preferred team size? (Solo, 2-5, 5-10, 10+)", 6),
        ("How do you prefer to learn new skills? (Online courses, books, building projects, mentorship, bootcamps)", 7),
        ("What is your risk tolerance for career decisions? (Conservative/stable, moderate, aggressive/startup)", 8),
        ("What are your top 3 values in a workplace? (e.g., innovation, work-life balance, compensation, impact, learning, diversity)", 9),
        ("What industry sectors interest you most? (e.g., FinTech, HealthTech, AI, Gaming, E-commerce, SaaS, Aerospace)", 10),
        ("What is your expected salary range for your next role?", 11),
        ("Do you have any geographical preferences or constraints for work location?", 12),
    ]

    for q_text, order in career_data:
        CareerQuestion.objects.create(
            question_text=q_text,
            order=order
        )

    print(f"Questions Population Done! Assessment: {AssessmentQuestion.objects.count()}, Career: {CareerQuestion.objects.count()}")

if __name__ == '__main__':
    populate()
