"""
URL configuration for comp_research project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""


from django.urls import path
from . import views

urlpatterns = [
    # When a user visits the root URL (''), it will show the form.
    # When they submit the form to '/match/', it will also be handled by this view.
    path('', views.landing_page_view, name='home'),
    path('analyzer/', views.matcher_view, name='matcher'),
    path('match/', views.matcher_view, name='match_submit'),
    path('signup/', views.signup_view, name='signup'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('interview/', views.interview_prep_view, name='interview_prep'),
    path('interview/chat/', views.interview_chat_view, name='interview_chat'),
    path('pathfinder/', views.pathfinder_hub, name='pathfinder'), # Landing Hub
    path('pathfinder/custom/', views.pathfinder_view, name='pathfinder_quiz'), # The Quiz
    path('personality-test/', views.personality_view, name='personality_test'),
    path('pathfinder/step/<int:step_index>/', views.path_node_detail_view, name='path_node_detail'),
    path('roadmap/', views.roadmap_view, name='roadmap'),
    path('profile/', views.profile_view, name='profile'),
    path('profile/edit/', views.edit_profile_view, name='edit_profile'),

    # New Features
    path('profile/save-path/', views.save_career_path, name='save_career_path'),
    path('profile/path/<int:path_id>/', views.view_career_path, name='view_career_path'),
    path('profile/update-path/<int:path_id>/', views.update_career_path, name='update_career_path'),
    path('profile/save-company/', views.save_company, name='save_company'),
    path('profile/save-interview/', views.save_interview_session, name='save_interview'),
    path('profile/delete/<str:item_type>/<int:item_id>/', views.delete_item, name='delete_item'),
    path('profile/path/<int:path_id>/step/<int:step_id>/', views.career_step_detail_view, name='career_step_detail'),
    path('career-library/', views.career_library_view, name='career_library'),
    
    # Therapy Features
    path('therapy/', views.therapy_landing_view, name='therapy_landing'),
    path('therapy/book/', views.book_session_view, {'session_type': 'therapy'}, name='book_therapy_session'),
    path('interview/book/', views.book_session_view, {'session_type': 'interview'}, name='book_interview_session'),
    path('therapy/chat/', views.ai_therapist_view, name='ai_therapist_view'),
    path('api/therapy/chat/', views.ai_chat_api, name='ai_chat_api'),
]