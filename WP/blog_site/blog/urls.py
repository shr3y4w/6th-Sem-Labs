from django.urls import path, include
from blog.views import archive, create_post

urlpatterns = [
    path('', archive, name = "archive"),
    path('create/', create_post, name = "create_post"),
]
