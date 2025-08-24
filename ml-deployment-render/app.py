from flask import Flask, request, jsonify, render_template
import joblib
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# ==================== CONFIG ==================== #
MODEL_FILE = 'classifier.pkl'
VECTORIZER_FILE = 'vectorizer.pkl'
DATA_DIR = 'data/synthetic_v2'  # folder containing dataset_shard_*.csv.gz
vectorizer = None
model = None

# ==================== Load Dataset ==================== #
def load_dataset():
    """Load and combine all dataset shards into one DataFrame."""
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"{DATA_DIR} folder not found!")
    files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('.csv.gz')]
    if not files:
        raise FileNotFoundError("No dataset shards found!")
    df_list = [pd.read_csv(f, compression='gzip') for f in files]
    df = pd.concat(df_list, ignore_index=True)
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("Dataset must have 'text' and 'label' columns.")
    return df

# ==================== Train Model ==================== #
def train_and_save_model():
    df = load_dataset()
    X = df['text'].astype(str)
    y = df['label'].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_vec, y_train)

    y_pred = clf.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {acc*100:.2f}%")

    joblib.dump(tfidf, VECTORIZER_FILE)
    joblib.dump(clf, MODEL_FILE)
    return acc

# ==================== Load Model ==================== #
def load_models():
    global vectorizer, model
    if os.path.exists(VECTORIZER_FILE) and os.path.exists(MODEL_FILE):
        vectorizer = joblib.load(VECTORIZER_FILE)
        model = joblib.load(MODEL_FILE)
        print("Models loaded successfully!")
        return True
    return False

# ==================== Flask Routes ==================== #
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health')
def health_check():
    return jsonify({
        "status": "healthy" if vectorizer and model else "unhealthy",
        "models_loaded": vectorizer is not None and model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    if vectorizer is None or model is None:
        return jsonify({'error': 'Models not loaded'}), 500
    try:
        text = request.form.get('text', '')
        if not text.strip():
            return jsonify({'error': 'Text input is empty'}), 400
        X = vectorizer.transform([text])
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X).max()
        result = {
            'prediction': prediction.upper(),
            'confidence': round(float(probability), 4),
            'input_text': text
        }
        return render_template('index.html', result=result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def api_predict():
    if vectorizer is None or model is None:
        return jsonify({'error': 'Models not loaded'}), 500
    try:
        data = request.get_json(force=True)
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing text parameter'}), 400
        text = data['text']
        X = vectorizer.transform([text])
        prediction = model.predict(X)[0]
        probas = model.predict_proba(X)[0]
        confidence_scores = {cls: round(float(prob), 4) for cls, prob in zip(model.classes_, probas)}

        result = {
            'prediction': prediction.upper(),
            'confidence': round(float(probas.max()), 4),
            'confidence_scores': confidence_scores,
            'input_text': text
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/train', methods=['GET'])
def train_route():
    try:
        acc = train_and_save_model()
        load_models()
        return jsonify({"message": "Model retrained successfully", "accuracy": f"{acc*100:.2f}%"})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ==================== Startup ==================== #
if not load_models():
    print("No saved model found. Training new model...")
    acc = train_and_save_model()
    print(f"Initial trained model accuracy: {acc*100:.2f}%")
    load_models()

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
