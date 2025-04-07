from django.shortcuts import render
from blog.models import BlogPostModel
from blog.forms import BlogForm
from datetime import datetime
from django.http import HttpResponseRedirect #imp

def archive(request):
    posts = BlogPostModel.objects.all()
    return render(request, 'archive.html', {'posts':posts, 'form': BlogForm()})

def create_post(request):
    if request.method=='POST':
        form = BlogForm(request.POST)

        if form.is_valid():
            post = form.save(commit = False)
            post.time = datetime.now()
            post.save()
            return HttpResponseRedirect('/')   
        
# What happens if user refreshes the page?
# render() — browser tries to POST again (can lead to duplicate posts!)
# redirect() — browser just GETs / again. No form is resubmitted. ✅