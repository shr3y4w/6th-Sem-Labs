from django.shortcuts import render
from .forms import GroceryForm

def grocery_list(request):
    form = GroceryForm()
    selected_items=[]
    total = 0

    if request.method == "POST":
        form = GroceryForm(request.POST)

        if form.is_valid():
            selected_items = form.cleaned_data['items']
            total = sum([item.price for item in selected_items])
        
    return render(request, 'grocery_list.html',{'form':form, 'selected_items':selected_items, 'total':total})