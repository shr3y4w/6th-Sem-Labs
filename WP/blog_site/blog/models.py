from django.db import models

# Create your models here.
class BlogPostModel(models.Model):
    title = models.CharField(max_length=50)
    content = models.TextField()
    time = models.DateTimeField()

    class Meta:
        ordering = ('-time',)   #imp