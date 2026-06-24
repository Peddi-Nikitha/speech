from django import forms
from .models import UserRegistrationModel

# from django import forms
# from .models import UserRegistrationModel

class UserRegistrationForm(forms.ModelForm):
    class Meta:
        model = UserRegistrationModel
        fields = ['name', 'loginid', 'password', 'mobile', 'email', 'locality', 'address', 'city', 'state']
        widgets = {
            'password': forms.PasswordInput(attrs={'pattern': '(?=.*\\d)(?=.*[a-z])(?=.*[A-Z]).{8,}', 'title': 'Must contain at least one number and one uppercase and lowercase letter, and at least 8 or more characters'}),
        }
  # could be 'waiting', 'activated', etc.


    class Meta():
        model = UserRegistrationModel
        exclude = ['status']

# detection/forms.py




