from django.db import models
from django.utils import timezone

class History(models.Model):
    name_people = models.CharField(max_length=255)
    datetime_appear = models.DateTimeField(default=timezone.now)
    
    def __str__(self):
        return self.name_people