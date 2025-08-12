# classifier.py
import pandas as pd
import joblib
import re
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# ==========================
# Step 1: Load the dataset
# ==========================
df = pd.read_csv("transactions.csv")

# Ensure required columns exist
required_cols = {"merchant", "description", "category"}
if not required_cols.issubset(df.columns):
    raise ValueError(f"CSV must contain columns: {required_cols}")

# ==========================
# Step 2: Preprocessing function
# ==========================
def preprocess_text(text):
    if pd.isnull(text):
        return ""
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Combine merchant and description for better context
df["combined_text"] = (df["merchant"].astype(str) + " " + df["description"].astype(str)).apply(preprocess_text)

# ==========================
# Step 3: Split data
# ==========================
X = df["combined_text"]
y = df["category"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ==========================
# Step 4: Vectorization
# ==========================
vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ==========================
# Step 5: Train model
# ==========================
model = LogisticRegression(max_iter=500, solver="lbfgs", multi_class="auto", random_state=42)
model.fit(X_train_tfidf, y_train)

# ==========================
# Step 6: Evaluate model
# ==========================
y_pred = model.predict(X_test_tfidf)
print("\nModel Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ==========================
# Step 7: Save model + vectorizer
# ==========================
joblib.dump((model, vectorizer), "expense_model.pkl")
print("\nModel saved as expense_model.pkl")
