from django.shortcuts import render, HttpResponse
from django.contrib import messages
from .forms import UserRegistrationForm
from .models import UserRegistrationModel,VideoAnalysis
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score, accuracy_score
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, KFold, train_test_split
from scipy.stats import uniform, randint
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
import xgboost
from xgboost import XGBClassifier
from django.conf import settings
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import matplotlib.pyplot as plt
import numpy as np

# from django.conf import settings
# from .models import predictions
# import os
# import pandas as pd


# Create your views here.
from django.shortcuts import render
from django.contrib import messages
from .forms import UserRegistrationForm
from .models import UserRegistrationModel

def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        email = request.POST.get('email')
        mobile = request.POST.get('mobile')

        if UserRegistrationModel.objects.filter(email=email).exists():
            messages.error(request, 'Email already exists')
        elif UserRegistrationModel.objects.filter(mobile=mobile).exists():
            messages.error(request, 'Mobile number already exists')
        elif form.is_valid():
            user = form.save(commit=False)
            user.status = 'waiting'
            user.save()
            messages.success(request, 'You have been successfully registered')
            return render(request, 'UserRegistrations.html', {'form': UserRegistrationForm()})
        else:
            messages.error(request, 'Form is invalid')

        return render(request, 'UserRegistrations.html', {'form': form})
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})






def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHomePage.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):
    return render(request, 'users/UserHomePage.html', {})

import os
import torch
import numpy as np
import soundfile as sf
from django.http import JsonResponse
from django.views import View
from django.core.files.storage import FileSystemStorage


from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import torch
import soundfile as sf
import os
from django.conf import settings
import torch
import torch.nn as nn

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, padding=4),  # Output: [16, T]
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=9, padding=4),  # Output: [32, T]
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=9, padding=4),  # Output: [64, T]
            nn.ReLU(),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=9, padding=4),  # Output: [32, T]
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=9, padding=4),  # Output: [16, T]
            nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=9, padding=4),  # Output: [1, T]
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


model_path = settings.MEDIA_ROOT + '//' +'cae_speech_enhancement.pth' 
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import soundfile as sf
import torch
  # Ensure this import is correct

from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import soundfile as sf
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
  # Ensure this import is correct

def denoise_audio_view(request):
    context = {}

    if request.method == 'POST' and 'audio_file' in request.FILES:
        audio_file = request.FILES['audio_file']
        fs = FileSystemStorage()
        filename = fs.save(audio_file.name, audio_file)
        file_path = fs.path(filename)

        # Load your model
        model = ConvAutoencoder()
        # Adjust this path
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()

        # Read the noisy audio
        noisy_audio, sample_rate = sf.read(file_path)

        # Denoise the audio
        denoised_audio = denoise_audio(noisy_audio, model)

        # Save the denoised audio
        output_filename = 'denoised_' + audio_file.name
        output_path = fs.path(output_filename)
        sf.write(output_path, denoised_audio, sample_rate)

        # Create and save waveform plots
        noisy_plot_path = create_waveform_plot(noisy_audio, 'Noisy Audio Waveform', 'noisy_waveform.png')
        denoised_plot_path = create_waveform_plot(denoised_audio, 'Denoised Audio Waveform', 'denoised_waveform.png')

        # Set the URLs for both noisy and denoised audio and their waveforms
        context['noisy_audio_url'] = fs.url(filename)
        context['denoised_audio_url'] = fs.url(output_filename)
        context['noisy_plot_url'] = fs.url(os.path.basename(noisy_plot_path))
        context['denoised_plot_url'] = fs.url(os.path.basename(denoised_plot_path))

    return render(request, 'users/results.html', context)

def denoise_audio(noisy_audio, model):
    with torch.no_grad():
        noisy_tensor = torch.tensor(noisy_audio).unsqueeze(0).float()
        denoised_tensor = model(noisy_tensor)
        denoised_audio = denoised_tensor.squeeze(0).numpy()
        return denoised_audio

def create_waveform_plot(audio, title, filename):
    plt.figure(figsize=(10, 4))
    plt.plot(audio)
    plt.title(title)
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    
    # Save the plot
    fs = FileSystemStorage()
    plot_path = fs.path(filename)
    plt.savefig(plot_path)
    plt.close()
    
    return plot_path




