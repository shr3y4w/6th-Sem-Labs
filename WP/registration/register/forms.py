from django import forms

class RegistrationForm(forms.Form):
    name = forms.CharField(label="User name", required=True)
    password = forms.CharField(label="Password", widget=forms.PasswordInput, required=True)  #pwd!!
    contact  = forms.CharField(label="Contact No", required=True)
    email = forms.EmailField(label="Email id", required = True)

    