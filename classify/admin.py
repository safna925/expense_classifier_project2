from django.contrib import admin
from .models import PredictionHistory

@admin.register(PredictionHistory)
class PredictionHistoryAdmin(admin.ModelAdmin):
    list_display = ('input_text', 'prediction', 'amount', 'timestamp')
    search_fields = ('input_text', 'prediction')
    list_filter = ('prediction', 'timestamp')
