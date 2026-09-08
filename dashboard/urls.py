from django.urls import path
from . import views

urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("api/analyze/", views.analyze_api, name="analyze_api"),
]
