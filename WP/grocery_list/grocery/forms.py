from django import forms
from .models import GroceryItem

class GroceryForm(forms.Form):
    items = forms.ModelMultipleChoiceField(queryset= GroceryItem.objects.all(),
                                           widget= forms.CheckboxSelectMultiple,
                                           required=True)
    
    