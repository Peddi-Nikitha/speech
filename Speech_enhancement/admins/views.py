from django.shortcuts import render, redirect
from django.contrib import messages
from users.models import UserRegistrationModel

from django.shortcuts import render

def AdminHome(request):
    return render(request, 'AdminHome.html') 

def RegisterUsersView(request):
    data = UserRegistrationModel.objects.all()
    return render(request, 'admins/viewregisterusers.html', {'data': data})

def ActivaUsers(request, uid):
    try:
        user = UserRegistrationModel.objects.get(id=uid)
        user.status = 'activated'
        user.save()
        messages.success(request, f'User {user.name} activated.')
    except UserRegistrationModel.DoesNotExist:
        messages.error(request, 'User not found.')
    return redirect('RegisterUsersView')

def DeleteUsers(request):
    if request.method == 'GET':
        uid = request.GET.get('uid')
        try:
            user = UserRegistrationModel.objects.get(id=uid)
            user.delete()
            messages.success(request, 'User deleted successfully.')
        except UserRegistrationModel.DoesNotExist:
            messages.error(request, 'User not found.')
    return redirect('RegisterUsersView')
