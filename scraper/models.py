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
