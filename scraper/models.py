from django.db import models

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
