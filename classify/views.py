from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.utils.timezone import localtime, now
from django.http import HttpResponse
from .models import PredictionHistory
from collections import defaultdict
from django.db.models import Sum
import pandas as pd
import joblib
import io
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet

# Load the trained model
model = joblib.load("classifier/expense_model.pkl")

# --- Single Text Classification View ---
def classify_expense(request):
    history = request.session.get("history", [])

    if request.method == "POST":
        if "clear" in request.POST:
            request.session["history"] = []
            history = []
        else:
            text = request.POST.get("text")
            amount = float(request.POST.get("amount", 0))

            prediction = model.predict([text])[0]

            history.append({"text": text, "prediction": prediction, "amount": amount})
            request.session["history"] = history

            PredictionHistory.objects.create(
                input_text=text,
                prediction=prediction,
                amount=amount
            )

            return render(request, "result.html", {
                "prediction": prediction,
                "amount": amount,
                "text": text
            })

    return render(request, "form.html", {"history": history})


# --- Bulk CSV Upload View ---
def upload_csv(request):
    results = None

    if request.method == "POST" and request.FILES.get("csv_file"):
        csv_file = request.FILES["csv_file"]

        if not csv_file.name.endswith(".csv"):
            return render(request, "upload.html", {"error": "File is not a CSV file."})

        fs = FileSystemStorage()
        filename = fs.save(csv_file.name, csv_file)
        file_path = fs.path(filename)

        try:
            df = pd.read_csv(file_path)

            if "text" not in df.columns or "amount" not in df.columns:
                return render(request, "upload.html", {
                    "error": "CSV must contain 'text' and 'amount' columns."
                })

            predictions = model.predict(df["text"])

            results = []
            for text, pred, amt in zip(df["text"], predictions, df["amount"]):
                try:
                    amt = float(amt)
                except:
                    amt = 0.0

                PredictionHistory.objects.create(
                    input_text=text,
                    prediction=pred,
                    amount=amt
                )

                results.append({
                    "text": text,
                    "prediction": pred,
                    "amount": amt
                })

        except Exception as e:
            return render(request, "upload.html", {"error": f"Error reading CSV: {str(e)}"})

    return render(request, "upload.html", {"results": results})


# --- Multiple Prediction View (Form-based without CSV) ---
def multiple_prediction_view(request):
    predictions = []

    if request.method == "POST":
        descriptions = request.POST.getlist("text[]")
        amounts = request.POST.getlist("amount[]")

        for text, amount in zip(descriptions, amounts):
            if text.strip():  # Skip empty rows
                try:
                    amt = float(amount)
                except:
                    amt = 0.0

                prediction = model.predict([text])[0]

                PredictionHistory.objects.create(
                    input_text=text,
                    prediction=prediction,
                    amount=amt
                )

                predictions.append({
                    "text": text,
                    "amount": amt,
                    "prediction": prediction
                })

    return render(request, "multiple.html", {"results": predictions})


# --- Dashboard View ---
def dashboard(request):
    history = PredictionHistory.objects.all().order_by('-timestamp')
    amount_per_category = defaultdict(float)
    count_per_category = defaultdict(int)

    for entry in history:
        amount_per_category[entry.prediction] += float(entry.amount)
        count_per_category[entry.prediction] += 1

    chart_labels = list(count_per_category.keys())
    chart_data = list(count_per_category.values())
    amount_labels = list(amount_per_category.keys())
    amount_data = list(amount_per_category.values())

    total_amount = sum(amount_data)
    total_predictions = history.count()
    avg_per_category = {}
    for cat in amount_per_category:
        if count_per_category[cat] > 0:
            avg_per_category[cat] = amount_per_category[cat] / count_per_category[cat]

    latest = history.first()

    context = {
        'history': history,
        'categories': chart_labels,
        'amounts': amount_data,
        'amount_per_category': dict(amount_per_category),
        'total_amount': total_amount,
        'total': total_predictions,
        'avg_per_category': avg_per_category,
        'latest': latest,
        'labels': chart_labels,
        'data': chart_data,
        'amount_labels': amount_labels,
        'amount_data': amount_data,
    }
    return render(request, 'dashboard.html', context)


# --- PDF Download View ---
def download_pdf(request):
    history = PredictionHistory.objects.all().order_by('timestamp')
    total_predictions = history.count()
    total_amount = history.aggregate(Sum("amount"))['amount__sum'] or 0

    category_amounts = defaultdict(float)
    for item in history:
        category_amounts[item.prediction] += item.amount or 0

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("💼 Expense Classification Report", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Generated on: {localtime(now()).strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
    story.append(Spacer(1, 20))
    story.append(Paragraph(f"Total Predictions: <b>{total_predictions}</b>", styles["Normal"]))
    story.append(Paragraph(f"Total Amount Spent: <b>₹{total_amount:.2f}</b>", styles["Normal"]))
    story.append(Spacer(1, 20))

    table_data = [["Category", "Total Amount (₹)"]]
    for cat, amt in category_amounts.items():
        table_data.append([cat, f"₹{amt:.2f}"])

    table = Table(table_data, hAlign='LEFT')
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    story.append(table)
    story.append(Spacer(1, 30))
    story.append(PageBreak())

    story.append(Paragraph("🧾 Detailed Expense Entries", styles["Heading2"]))
    story.append(Spacer(1, 12))
    entry_data = [["Description", "Category", "Amount (₹)", "Date & Time"]]
    for item in history:
        entry_data.append([
            item.input_text,
            item.prediction,
            f"₹{item.amount:.2f}",
            localtime(item.timestamp).strftime("%Y-%m-%d %H:%M")
        ])

    entry_table = Table(entry_data, repeatRows=1)
    entry_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ALIGN', (2, 1), (2, -1), 'RIGHT'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
    ]))
    story.append(entry_table)

    doc.build(story)
    buffer.seek(0)

    return HttpResponse(buffer, content_type='application/pdf', headers={
        'Content-Disposition': 'attachment; filename="expense_report_full.pdf"'
    })
