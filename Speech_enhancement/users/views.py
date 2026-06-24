import os

from django.conf import settings
from django.contrib import messages
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render

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

    form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})


def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            if check.status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                return render(request, 'users/UserHomePage.html', {})
            messages.success(request, 'Your Account Not at activated')
            return render(request, 'UserLogin.html')
        except UserRegistrationModel.DoesNotExist:
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):
    return render(request, 'users/UserHomePage.html', {})


def _get_conv_autoencoder():
    import torch
    import torch.nn as nn

    class ConvAutoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=9, padding=4),
                nn.ReLU(),
                nn.Conv1d(16, 32, kernel_size=9, padding=4),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=9, padding=4),
                nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.Conv1d(64, 32, kernel_size=9, padding=4),
                nn.ReLU(),
                nn.Conv1d(32, 16, kernel_size=9, padding=4),
                nn.ReLU(),
                nn.Conv1d(16, 1, kernel_size=9, padding=4),
            )

        def forward(self, x):
            return self.decoder(self.encoder(x))

    return ConvAutoencoder


def denoise_audio_view(request):
    context = {}

    if request.method == 'POST' and 'audio_file' in request.FILES:
        try:
            import matplotlib.pyplot as plt
            import soundfile as sf
            import torch
        except ImportError:
            messages.error(
                request,
                'Audio denoising is not available on this server. Run the app locally for ML features.',
            )
            return render(request, 'users/results.html', context)

        audio_file = request.FILES['audio_file']
        fs = FileSystemStorage()
        filename = fs.save(audio_file.name, audio_file)
        file_path = fs.path(filename)
        model_path = os.path.join(settings.MEDIA_ROOT, 'cae_speech_enhancement.pth')

        if not os.path.exists(model_path):
            messages.error(request, 'Model file not found. Upload cae_speech_enhancement.pth to the media folder.')
            return render(request, 'users/results.html', context)

        ConvAutoencoder = _get_conv_autoencoder()
        model = ConvAutoencoder()
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()

        noisy_audio, sample_rate = sf.read(file_path)
        denoised_audio = denoise_audio(noisy_audio, model)

        output_filename = 'denoised_' + audio_file.name
        output_path = fs.path(output_filename)
        sf.write(output_path, denoised_audio, sample_rate)

        noisy_plot_path = create_waveform_plot(noisy_audio, 'Noisy Audio Waveform', 'noisy_waveform.png')
        denoised_plot_path = create_waveform_plot(
            denoised_audio, 'Denoised Audio Waveform', 'denoised_waveform.png'
        )

        context['noisy_audio_url'] = fs.url(filename)
        context['denoised_audio_url'] = fs.url(output_filename)
        context['noisy_plot_url'] = fs.url(os.path.basename(noisy_plot_path))
        context['denoised_plot_url'] = fs.url(os.path.basename(denoised_plot_path))

    return render(request, 'users/results.html', context)


def denoise_audio(noisy_audio, model):
    import torch

    with torch.no_grad():
        noisy_tensor = torch.tensor(noisy_audio).unsqueeze(0).float()
        denoised_tensor = model(noisy_tensor)
        return denoised_tensor.squeeze(0).numpy()


def create_waveform_plot(audio, title, filename):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 4))
    plt.plot(audio)
    plt.title(title)
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')

    fs = FileSystemStorage()
    plot_path = fs.path(filename)
    plt.savefig(plot_path)
    plt.close()
    return plot_path
