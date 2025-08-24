import pickle

# --------------------------
# 1. Load Model + Vectorizer
# --------------------------
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("logreg_model.pkl", "rb") as f:
    clf = pickle.load(f)

print("[INFO] Model and vectorizer loaded!")
print("Type 'exit' to quit.\n")

# --------------------------
# 2. Interactive Prediction Loop
# --------------------------
while True:
    query = input("Enter a prompt/query: ")
    if query.lower() == "exit":
        break
    
    # Transform query to TF-IDF
    X = vectorizer.transform([query])
    
    # Predict
    pred = clf.predict(X)[0]
    prob = clf.predict_proba(X).max()
    
    print(f"Prediction: {pred.upper()} (Confidence: {prob:.2f})\n")
