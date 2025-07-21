from django.db import models

class PredictionHistory(models.Model):
    input_text = models.TextField()
    prediction = models.CharField(max_length=100)
    amount = models.FloatField(default=0.0)  # ✅ New field to store the amount spent
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.input_text[:30]}... → {self.prediction} | ₹{self.amount:.2f} at {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
