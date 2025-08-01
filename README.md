# 💰 AI-Driven Expense Classifier

An intelligent, user-friendly web application that classifies your expenses automatically using machine learning. Built with Django and scikit-learn, this project allows users to upload or manually enter financial transactions and receive categorized expense predictions along with visual dashboards and export features.

---

## 🚀 Features

✅ **ML-Powered Expense Classification**  
Classifies expenses like Groceries, Utilities, Transportation, Entertainment, etc., based on transaction description and amount.

✅ **CSV Upload**  
Upload a CSV file of multiple transactions and get predictions instantly.

✅ **Prediction History**  
Keeps a timestamped log of all past predictions.

✅ **PDF Export**  
Download expense summaries as polished, branded PDF reports.

✅ **Dashboard with Charts**  
Visual overview of your spending by category and time.

✅ **Batch Input via Form**  
Add multiple expense rows manually (without a CSV).

---

## 🧠 Machine Learning Model

- **Algorithm:** Multinomial Naive Bayes (via scikit-learn)
- **Training Data:** Custom-labeled CSV dataset with categories and transaction texts
- **Vectorization:** CountVectorizer
- **Accuracy:** Achieved over **90% accuracy** with optimized dataset

---

## 🛠️ Tech Stack

| Layer       | Technology         |
|-------------|--------------------|
| Backend     | Django              |
| ML Model    | scikit-learn        |
| Frontend    | HTML, CSS (modern UI in progress) |
| Database    | SQLite (can be upgraded) |
| Charts      | Chart.js            |
| PDF Export  | ReportLab           |

---

## 📁 Project Structure

expense_classifier_project2/
│
├── classify/ # Django app
│ ├── templates/ # HTML templates
│ ├── static/ # CSS, JS
│ ├── views.py # Core logic
│ ├── models.py # PredictionHistory model
│ ├── urls.py # App-level routing
│
├── expense_classifier/ # Django project settings
│ ├── settings.py
│ ├── urls.py
│
├── classifier.py # ML training script
├── expense_model.pkl # Trained model
├── transactions.csv # Training dataset
├── db.sqlite3 # SQLite DB
├── README.md
└── requirements.txt

## 📦 How to Run the Project Locally

```bash
git clone https://github.com/safna925/expense_classifier_project2.git
cd expense_classifier_project2
python -m venv env
source env/bin/activate  # or env\Scripts\activate on Windows
pip install -r requirements.txt
python manage.py runserver
Then go to: http://127.0.0.1:8000/ in your browser.


📌 Future Improvements
 User authentication (login/signup)

 Better UI (Tailwind/Figma-based)

 Category-wise filters on dashboard

 Cloud deployment (Render/Heroku)



 Contact
Name: Safna Sajeev
GitHub: safna925
Email: safnasajeev1214@gmail.com



📄 License
This project is under the MIT License.

