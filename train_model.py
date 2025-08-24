import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pickle

# --------------------------
# 1. Load Dataset
# --------------------------
print("[INFO] Loading dataset...")
df = pd.read_csv("synthetic_1M_safe_injection.csv")  # change filename if needed
print("Dataset shape:", df.shape)
print(df["label"].value_counts())

# --------------------------
# 2. Split Dataset (80% train / 20% test)
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)
print(f"[INFO] Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# --------------------------
# 3. Vectorize text with TF-IDF
# --------------------------
print("[INFO] Vectorizing text...")
vectorizer = TfidfVectorizer(
    max_features=50000,   # adjust if needed
    ngram_range=(1, 2),   # unigrams + bigrams
    stop_words="english"
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# --------------------------
# 4. Train Logistic Regression
# --------------------------
print("[INFO] Training Logistic Regression (Windows-friendly)...")
clf = LogisticRegression(
    max_iter=200,
    n_jobs=1,  # single-threaded for Windows
    class_weight="balanced",
    verbose=1
)
clf.fit(X_train_tfidf, y_train)

# --------------------------
# 5. Evaluate Model
# --------------------------
print("[INFO] Evaluating model...")
y_pred = clf.predict(X_test_tfidf)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix (text-based)
cm = confusion_matrix(y_test, y_pred, labels=["safe","injection"])
print("\nConfusion Matrix:")
print("Labels: ['safe','injection']")
print(cm)

# --------------------------
# 6. Save Model + Vectorizer
# --------------------------
with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("logreg_model.pkl", "wb") as f:
    pickle.dump(clf, f)

print("\n[INFO] Model and vectorizer saved successfully!")
