"""Speech URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

app_name = 'detection'

from django.contrib import admin
from django.urls import path
from Speech import views as mainView
from admins import views as admins
from users import views as usr
from django.conf.urls.static import static
from django.conf import settings

# 
from .views import AdminLogin
from .views import AdminLoginCheck


# 
from django.urls import path
from admins import views as admin_views
from users import views as user_views
urlpatterns = [
    path('admin/', admin.site.urls),
    path("", mainView.index, name="index"),
    # path("index/", mainView.index, name="index"),
    path("AdminHome/",mainView.AdminLogin,name="AdminLogin"),
    path("UserLogin/", mainView.UserLogin, name="UserLogin"),
    path("UserRegister/", mainView.UserRegister, name="UserRegister"),
    # path("AdminLoginCheck/", views.AdminLoginCheck, name="AdminLoginCheck"),
    
    #User Views
    path("UserRegisterActions/", usr.UserRegisterActions, name="UserRegisterActions"),
    path("UserLoginCheck/", usr.UserLoginCheck, name="UserLoginCheck"),
    path("UserHome/", usr.UserHome, name="UserHome"),
    path('denoise_audio_view/', usr.denoise_audio_view, name='denoise_audio_view'),



    

    # adminviews
    path('Adminlogin/', AdminLogin, name='AdminLogin'),
    path('AdminLoginCheck/', AdminLoginCheck, name='AdminLoginCheck'),
    # path("AdminLoginCheck/", AdminLoginCheck, name="AdminLoginCheck"),
    path("AdminHome/", admins.AdminHome, name="AdminHome"),
    path('RegisterUsersView/', admins.RegisterUsersView, name='RegisterUsersView'),
    path('activate_user/<int:uid>/', admins.ActivaUsers, name='ActivaUsers'),
    path('DeleteUsers/', admins.DeleteUsers, name='DeleteUsers'),
     
     
    # urlpatterns = [
    path('UserRegisterActions/', user_views.UserRegisterActions, name='UserRegisterActions'),
    path('RegisterUsersView/', admin_views.RegisterUsersView, name='RegisterUsersView'),
    path('activate_user/<int:uid>/', admin_views.ActivaUsers, name='ActivaUsers'),
    path('DeleteUsers/', admin_views.DeleteUsers, name='DeleteUsers'),
# ]

     ] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
