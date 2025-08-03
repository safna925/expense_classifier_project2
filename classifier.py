import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# ✅ Ensure classifier directory exists
os.makedirs("classify", exist_ok=True)

# ✅ Load the improved dataset
data = pd.read_csv("transactions.csv")

X = data['text']
y = data['category']

# ✅ TF-IDF with bigrams and stop words removal
vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
X_vec = vectorizer.fit_transform(X)

# ✅ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42, stratify=y
)

# ✅ Train Linear SVC (better than Naive Bayes for text classification)
model = LinearSVC()
model.fit(X_train, y_train)

# ✅ Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ✅ Save both model and vectorizer
joblib.dump(model, "classify/expense_model.pkl")
joblib.dump(vectorizer, "classify/vectorizer.pkl")
print("✅ Model and Vectorizer saved successfully!")
