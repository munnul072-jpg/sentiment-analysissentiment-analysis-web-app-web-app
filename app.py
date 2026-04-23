"""
Flask Web Application for Sentiment Analysis
B.Tech Major Project - KIET Group of Institutions
"""

from flask import Flask, render_template, request, jsonify
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings('ignore')

# Initialize Flask application
app = Flask(__name__)

# Download NLTK data (if not already)
try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

# Initialize stemmer
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Global variables
sentiment_model = None
feature_vectorizer = None

def preprocess_text(text):
    """Preprocess input text"""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    try:
        tokens = word_tokenize(text)
    except:
        tokens = text.split()
    
    processed = [stemmer.stem(token) for token in tokens 
                 if token not in stop_words and len(token) > 2]
    
    return " ".join(processed)

@app.route('/')
def home():
    """Home page"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze sentiment of text"""
    global sentiment_model, feature_vectorizer
    
    # Lazy load models (load only when needed)
    if sentiment_model is None or feature_vectorizer is None:
        try:
            sentiment_model = joblib.load('models/sentiment_model.pkl')
            feature_vectorizer = joblib.load('models/vectorizer.pkl')
            print("[SUCCESS] Models loaded on demand")
        except Exception as e:
            print(f"[ERROR] Failed to load models: {e}")
            return jsonify({'error': 'Model loading failed'}), 500
    
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text or not text.strip():
            return jsonify({'error': 'Please enter some text'}), 400
        
        # Preprocess
        processed = preprocess_text(text)
        
        # Vectorize
        vectorized = feature_vectorizer.transform([processed])
        
        # Predict
        prediction = sentiment_model.predict(vectorized)[0]
        
        # Get confidence
        if hasattr(sentiment_model, 'predict_proba'):
            probs = sentiment_model.predict_proba(vectorized)[0]
            confidence = max(probs) * 100
        else:
            confidence = 85.0
        
        # Map results
        emoji_map = {'Positive': '😊', 'Negative': '😔', 'Neutral': '😐'}
        color_map = {'Positive': '#10b981', 'Negative': '#ef4444', 'Neutral': '#f59e0b'}
        
        return jsonify({
            'sentiment': prediction,
            'confidence': round(confidence, 2),
            'emoji': emoji_map.get(prediction, '😐'),
            'color': color_map.get(prediction, '#6b7280'),
            'text': text
        })
        
    except Exception as e:
        print(f"[ERROR] Analysis failed: {e}")
        return jsonify({'error': 'Analysis failed. Please try again.'}), 500

@app.route('/health')
def health():
    """Health check"""
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    print("\n" + "="*60)
    print("SENTIMENT ANALYSIS WEB APPLICATION")
    print("B.Tech Major Project 2025-26")
    print("KIET Group of Institutions")
    print("="*60)
    
    print("\n[INFO] Server starting...")
    print("[INFO] Models will load on first request")
    print("\n🌐 Open: http://localhost:5000")
    print("⏹️  Press CTRL+C to stop\n")
    
    app.run(debug=False, host='127.0.0.1', port=5000)