from django.shortcuts import render
from .forms import RegistrationForm
# Create your views here.

def register(request):
    if request.method == 'POST':
        form = RegistrationForm(request.POST)

        if form.is_valid():   #the () is so imp
            name = form.cleaned_data.get('name')
            email = form.cleaned_data.get('email')
            contact = form.cleaned_data.get('contact')

            return render(request, 'success.html', {'name':name, 'email':email, 'contact':contact})
        
    else:
        form = RegistrationForm()

    return render(request, 'registration.html',{'form':form})

def success(request):
    return render(request,'success.html')
