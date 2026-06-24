from django.db import models

# Create your models here.
from django.db import models

class UserRegistrationModel(models.Model):
    name = models.CharField(max_length=100)
    loginid = models.CharField(max_length=100)
    password = models.CharField(max_length=100)
    mobile = models.CharField(max_length=100)
    email = models.EmailField(max_length=100, unique=True)
    locality = models.CharField(max_length=100)
    address = models.CharField(max_length=250)
    city = models.CharField(max_length=100)
    state = models.CharField(max_length=100)
    status = models.CharField(max_length=20, default='waiting')

    def __str__(self):
        return self.name
        
# detection/models.py
from django.db import models


class VideoAnalysis(models.Model):
    original_video = models.CharField(max_length=255)
    processed_video = models.CharField(max_length=255)
    frame_count = models.IntegerField()
    total_elephants = models.IntegerField()
    timestamp = models.DateTimeField(auto_now_add=True)



class FrameResult(models.Model):
    video_analysis = models.ForeignKey(VideoAnalysis, on_delete=models.CASCADE)
    frame_number = models.IntegerField()
    is_elephant_present = models.BooleanField()
    elephant_proportion = models.FloatField()
    num_elephants = models.IntegerField()


        
        
