from django.urls import path
from . import views

app_name = 'matcher'

urlpatterns = [
    path('', views.home, name='home'),
    path('test/', views.ats_test, name='ats_test'),
    path('analyze/', views.analyze_resume, name='analyze_resume'),
]
