import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

# ✅ Make sure 'classifier' directory exists for saving the model
os.makedirs("classifier", exist_ok=True)

# ✅ Load the dataset
data = pd.read_csv("transactions.csv")  # Make sure this CSV exists with correct data

# ✅ Features and labels
X = data["text"]
y = data["category"]

# ✅ Stratified train-test split (preserves category distribution)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ✅ Build the pipeline
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# ✅ Train the model
model.fit(X_train, y_train)

# ✅ Predict on test set and calculate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model accuracy: {accuracy * 100:.2f}%")

# ✅ Save the model
joblib.dump(model, "classifier/expense_model.pkl")
print("✅ Model saved as classifier/expense_model.pkl")
