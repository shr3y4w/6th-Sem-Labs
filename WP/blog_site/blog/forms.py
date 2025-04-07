from django import forms
from blog.models import BlogPostModel

class BlogForm(forms.ModelForm):
    class Meta:
        model = BlogPostModel
        exclude = ('time',)  #imp