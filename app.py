from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

nltk.download('punkt',      quiet=True)
nltk.download('stopwords',  quiet=True)
nltk.download('wordnet',    quiet=True)
nltk.download('punkt_tab',  quiet=True)
nltk.download('omw-1.4',    quiet=True)

app = Flask(__name__)
CORS(app)

with open('sentiment_model.pkl',  'rb') as f:
    model = pickle.load(f)
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

analyzer   = SentimentIntensityAnalyzer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
keep_words = {'not', 'no', 'never', 'nothing', 'neither', 'nor', 'very', 'too', 'but'}
stop_words = stop_words - keep_words

def preprocess(text):
    text   = str(text).lower()
    text   = re.sub(r'http\S+|<.*?>', '', text)
    text   = re.sub(r"[^a-z\s]", ' ', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens
              if t not in stop_words and len(t) > 2]
    return ' '.join(tokens)

@app.route('/')
def home():
    return send_file('sentiment_website.html')

@app.route('/predict', methods=['POST'])
def predict():
    data   = request.get_json()
    text   = data.get('text', '')

    if not text.strip():
        return jsonify({'error': 'No text provided'}), 400

    clean      = preprocess(text)
    vec        = tfidf.transform([clean])
    pred       = model.predict(vec)[0]
    proba      = model.predict_proba(vec)[0]
    confidence = round(float(max(proba)) * 100, 1)
    vader_s    = analyzer.polarity_scores(text)['compound']

    return jsonify({
        'sentiment':  pred,
        'confidence': confidence,
        'vader':      round(vader_s, 3),
        'text':       text
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'running'})

if __name__ == '__main__':
    print("Starting TasteSense server...")
    print("Open your browser and go to: http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
