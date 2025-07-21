from django.urls import path
from . import views

urlpatterns = [
    path('', views.classify_expense, name='classify_expense'),         # Single Prediction
    path('upload/', views.upload_csv, name='upload_csv'),              # CSV Upload
    path('multiple/', views.multiple_prediction_view, name='multiple'),# Form-based Multiple Prediction
    path('dashboard/', views.dashboard, name='dashboard'),             # Dashboard
    path('download/', views.download_pdf, name='download_pdf'),        # PDF Download
    path('export_pdf/', views.download_pdf, name='export_pdf'),        # PDF Export alias
]
