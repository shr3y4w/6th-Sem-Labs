from django.db import models

# Create your models here.
class GroceryItem(models.Model):
    name = models.CharField(max_length=50)
    price = models.DecimalField(max_digits=6, decimal_places=2)

    def __str__(self):
        return f"{self.name} (Rs. {self.price})"