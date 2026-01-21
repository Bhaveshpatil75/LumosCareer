from django.db import models
from django.contrib.auth.models import User
import os

class Company(models.Model):
    name = models.CharField(max_length=200, unique=True)
    interview_notes = models.TextField(
        blank=True, 
        help_text="Common interview questions, key cultural topics, or technical areas to focus on for this company."
    )

    class Meta:
        verbose_name_plural = "Companies"

    def __str__(self):
        return self.name
    

class AssessmentQuestion(models.Model):
    TRAIT_CHOICES = [
        ('EI', 'Extraversion (E) vs. Introversion (I)'),
        ('SN', 'Sensing (S) vs. Intuition (N)'),
        ('TF', 'Thinking (T) vs. Feeling (F)'),
        ('JP', 'Judging (J) vs. Perceiving (P)'),
    ]

    question_text = models.TextField()
    trait = models.CharField(max_length=2, choices=TRAIT_CHOICES)

    choice_a = models.CharField(max_length=200)
    choice_b = models.CharField(max_length=200)

    def __str__(self):
        return f"{self.trait}: {self.question_text[:50]}..."
    

    
    
class CareerQuestion(models.Model):
    question_text = models.TextField()
    order = models.IntegerField(default=0, help_text="Order of the question in the quiz.")

    class Meta:
        ordering = ['order']

    def __str__(self):
        return f"{self.order}. {self.question_text[:50]}..."


# core/models.py

class AssessmentResult(models.Model):
    # Links this result to a specific user.
    # OneToOneField means one user gets one result.
    # models.CASCADE means if the user is deleted, this result is also deleted.
    user = models.OneToOneField(User, on_delete=models.CASCADE, primary_key=True)

    # This field will store the final 4-letter type, e.g., "INTJ"
    result_type = models.CharField(max_length=4)
    
    # Store the full detailed report from the AI for deep dive pages
    detailed_report = models.JSONField(blank=True, null=True)

    # Keeps track of when the test was taken
    date_taken = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username}'s Result: {self.result_type}"

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    resume_file = models.FileField(upload_to='resumes/', blank=True, null=True, help_text="Upload your resume (PDF only, max 5MB).")
    resume_text = models.TextField(blank=True, help_text="Paste your resume here.")
    bio = models.TextField(blank=True, help_text="Short bio about yourself.")
    target_job_titles = models.TextField(blank=True, help_text="Comma-separated list of target job titles.")
    linkedin_url = models.URLField(blank=True, verbose_name="LinkedIn URL")
    github_url = models.URLField(blank=True, verbose_name="GitHub URL")
    leetcode_url = models.URLField(blank=True, verbose_name="LeetCode URL")

    @property
    def filename(self):
        if self.resume_file:
            return os.path.basename(self.resume_file.name)
        return ""

    def __str__(self):
        return f"{self.user.username}'s Profile"

class CareerPath(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='career_paths')
    title = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    roadmap_data = models.JSONField(help_text="JSON structure of the roadmap steps")
    progress = models.IntegerField(default=0, help_text="Percentage completion")
    status = models.CharField(max_length=20, choices=[('Active', 'Active'), ('Completed', 'Completed'), ('Paused', 'Paused')], default='Active')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.user.username} - {self.title}"

class SavedCompany(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='saved_companies')
    company_name = models.CharField(max_length=200)
    compatibility_score = models.FloatField(null=True, blank=True)
    analysis_report = models.JSONField(null=True, blank=True)
    date_saved = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name_plural = "Saved Companies"

    def __str__(self):
        return f"{self.user.username} - {self.company_name}"

class InterviewSession(models.Model):
    INTERVIEW_TYPES = [
        ('text', 'Text Based'),
        ('voice', 'Voice Based'),
    ]
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='interviews')
    company_name = models.CharField(max_length=200)
    interview_type = models.CharField(max_length=10, choices=INTERVIEW_TYPES)
    date_logged = models.DateTimeField(auto_now_add=True)
    job_description = models.TextField(blank=True)
    notes = models.TextField(blank=True)

    def __str__(self):
        return f"{self.user.username} - {self.company_name} ({self.interview_type})"

# --- ALGORITHMIC MODELS ---

class SkillNode(models.Model):
    """
    Represents a specific skill, role, or concept in the Knowledge Graph.
    Used by: Dijkstra, PageRank, SimulatedAnnealing.
    """
    name = models.CharField(max_length=100, unique=True)
    category = models.CharField(max_length=50, choices=[
        ('Language', 'Language'),
        ('Framework', 'Framework'),
        ('Tool', 'Tool'),
        ('Concept', 'Concept'),
        ('Role', 'Role'),
        ('Soft Skill', 'Soft Skill')
    ], default='Concept')
    description = models.TextField(blank=True)
    
    # Metadata for Algorithms
    difficulty_level = models.IntegerField(default=1, help_text="1-10 Scale")
    learning_weeks = models.IntegerField(default=1, help_text="Estimated weeks to learn")
    
    # PageRank Score (Cached)
    importance_score = models.FloatField(default=0.0, help_text="Calculated Centrality Score")

    def __str__(self):
        return self.name

class SkillEdge(models.Model):
    """
    Represents a directed connection (Prerequisite -> Target).
    Used by: Dijkstra (Structure).
    """
    source = models.ForeignKey(SkillNode, on_delete=models.CASCADE, related_name='outgoing_edges')
    target = models.ForeignKey(SkillNode, on_delete=models.CASCADE, related_name='incoming_edges')
    
    # Edge Weights
    weight_time = models.IntegerField(default=1, help_text="Transition time cost")
    weight_difficulty = models.IntegerField(default=1, help_text="Transition difficulty cost")
    
    class Meta:
        unique_together = ('source', 'target')

    def __str__(self):
        return f"{self.source} -> {self.target}"

class SkillSignal(models.Model):
    """
    Represents Market Data for a skill.
    Used by: BayesianPredictor, Apriori (as source).
    """
    skill = models.OneToOneField(SkillNode, on_delete=models.CASCADE, related_name='market_signal')
    
    # Bayesian Probabilities (P(Skill|Success) etc)
    # Storing as 'Lift' or raw probabilities? Let's store raw probs for flexibility.
    success_rate = models.FloatField(default=0.5, help_text="P(Skill|Hired) - Probability appearing in hired profiles")
    failure_rate = models.FloatField(default=0.1, help_text="P(Skill|Rejected) - Probability appearing in rejected profiles")
    
    # Demand Trend
    demand_trend = models.CharField(max_length=20, choices=[
        ('Rising', 'Rising'),
        ('Stable', 'Stable'),
        ('Falling', 'Falling')
    ], default='Stable')

    @property
    def lift(self):
        if self.failure_rate == 0: return 0
        return round(self.success_rate / self.failure_rate, 2)

    def __str__(self):
        return f"{self.skill.name} Signal (Lift: {self.lift}x)"
