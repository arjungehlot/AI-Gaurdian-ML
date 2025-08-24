import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import os
import glob

# -----------------------------
# CONFIG
# -----------------------------
DATA_DIR = "data/synthetic_v2"   # Folder where your shards are stored
VECTORIZER_FILE = "vectorizer.pkl"
CLASSIFIER_FILE = "classifier.pkl"
MAX_FEATURES = 5000
TEST_SIZE = 0.2
RANDOM_STATE = 42

# -----------------------------
# LOAD MULTIPLE SHARDS
# -----------------------------
def load_dataset():
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"{DATA_DIR} folder not found!")

    shard_files = glob.glob(os.path.join(DATA_DIR, "*.csv.gz"))
    if not shard_files:
        raise FileNotFoundError(f"No CSV shards found in {DATA_DIR}")

    print(f"[INFO] Found {len(shard_files)} dataset shards. Loading and concatenating...")
    dfs = [pd.read_csv(shard, compression="gzip") for shard in shard_files]
    df = pd.concat(dfs, ignore_index=True)

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Dataset must have 'text' and 'label' columns")

    print(f"[INFO] Loaded {len(df)} rows from all shards.")
    return df

# -----------------------------
# TRAINING SCRIPT
# -----------------------------
if __name__ == "__main__":
    print("[INFO] Loading dataset...")
    df = load_dataset()

    # Shuffle dataset
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"].astype(str),
        df["label"].astype(str),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label"]
    )

    print("[INFO] Vectorizing text...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=MAX_FEATURES,
        min_df=2,
        max_df=0.95
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print(f"[INFO] Vectorized: {X_train_vec.shape[0]} samples, {X_train_vec.shape[1]} features")

    # Train Logistic Regression model
    print("[INFO] Training classifier...")
    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
        n_jobs=1  # For Windows compatibility
    )
    clf.fit(X_train_vec, y_train)

    # Evaluate
    print("[INFO] Evaluating model...")
    y_pred = clf.predict(X_test_vec)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Save model & vectorizer
    joblib.dump(vectorizer, VECTORIZER_FILE)
    joblib.dump(clf, CLASSIFIER_FILE)
    print(f"[INFO] Model saved as {CLASSIFIER_FILE}")
    print(f"[INFO] Vectorizer saved as {VECTORIZER_FILE}")
