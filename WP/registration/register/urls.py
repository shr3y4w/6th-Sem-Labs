from django.urls import path,include
from .views import register,success

urlpatterns = [
    path('',register, name='register'),
    path('success/',success, name='success'),
]
