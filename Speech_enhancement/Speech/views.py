from django.shortcuts import render
from users.forms import UserRegistrationForm


def index(request):
    return render(request, 'index.html', {})

def AdminLogin(request):
    return render(request, 'AdminLogin.html', {})

def UserLogin(request):
    return render(request, 'UserLogin.html', {})


def UserRegister(request):
    form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})
# Speech/views.py

from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login

from django.shortcuts import redirect

def AdminLoginCheck(request):
    if request.method == "POST":
        uname = request.POST.get("loginid")
        pwd = request.POST.get("pswd")
        if uname == "admin" and pwd == "admin":
            return redirect('RegisterUsersView')  # This uses the name in urls.py
        else:
            return render(request, "AdminLogin.html", {"error": "Invalid credentials"})
