import joblib
import os

# -----------------------------
# CONFIG
# -----------------------------
VECTORIZER_FILE = "vectorizer.pkl"
CLASSIFIER_FILE = "classifier.pkl"

# -----------------------------
# LOAD MODEL & VECTORIZER
# -----------------------------
if not os.path.exists(VECTORIZER_FILE) or not os.path.exists(CLASSIFIER_FILE):
    raise FileNotFoundError("Model or vectorizer not found! Run train_model.py first.")

print("[INFO] Loading trained model and vectorizer...")
vectorizer = joblib.load(VECTORIZER_FILE)
clf = joblib.load(CLASSIFIER_FILE)

# -----------------------------
# PREDICT LOOP
# -----------------------------
print("Enter a text to classify (or type 'exit' to quit):")
while True:
    user_input = input("> ").strip()
    if user_input.lower() == "exit":
        print("Exiting...")
        break

    if not user_input:
        print("Please enter some text.")
        continue

    # Transform the input text using the same vectorizer
    X_input_vec = vectorizer.transform([user_input])

    # Predict and get probability
    prediction = clf.predict(X_input_vec)[0]
    proba = clf.predict_proba(X_input_vec)[0]
    confidence = proba.max()

    print(f"Prediction: {prediction} (Confidence: {confidence:.2f})")
