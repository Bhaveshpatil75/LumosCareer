from django.contrib import admin

# Register your models here.
from .models import Company, AssessmentQuestion, AssessmentResult, CareerQuestion

# Register your models here.
admin.site.register(Company)
admin.site.register(AssessmentQuestion)
admin.site.register(AssessmentResult)
admin.site.register(CareerQuestion)
