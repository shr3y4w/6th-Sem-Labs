from django.contrib import admin
from blog import models  #imp to do this

class BlogPostAdmin(admin.ModelAdmin):  #this
    list_display = ('title','time')

admin.site.register(models.BlogPostModel, BlogPostAdmin)  #imp