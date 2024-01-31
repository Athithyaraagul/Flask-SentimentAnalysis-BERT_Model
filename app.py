from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap
from transformers import BertTokenizer, BertForSequenceClassification
import torch

app = Flask(__name__)
Bootstrap(app)  # Initialize Bootstrap

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('/Users/Athithyaraagul/Developer/Machine_Learning/srm_researchpaper/BERT_Model_Tokenizer')
model = BertForSequenceClassification.from_pretrained('/Users/Athithyaraagul/Developer/Machine_Learning/srm_researchpaper/BERT_SavedModel')

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for processing user input
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user input from the form
        sentence = request.form['sentence']

        # Tokenize and predict using BERT model
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)
        prediction = outputs.logits.argmax(dim=1).item()

        # Map the predicted label to sentiment
        sentiments = ['Negative', 'Neutral', 'Positive']
        predicted_sentiment = sentiments[prediction]

        # Render the result template with the predicted sentiment
        return render_template('result.html', sentence=sentence, predicted_sentiment=predicted_sentiment)

if __name__ == '__main__':
    app.run(debug=True)

