from django.urls import path
from . import views #imp

urlpatterns = [
    path('',views.grocery_list, name='grocery_list'),   #remember views.
]