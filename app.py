from flask import Flask, render_template, request
from transformers import BertTokenizer, BertForSequenceClassification
import torch

app = Flask(__name__)

# Load the pre-trained sentiment analysis model
tokenizer = BertTokenizer.from_pretrained('BERT_SavedTokenizer')
model = BertForSequenceClassification.from_pretrained('BERT_SavedModel', num_labels=3)
model.eval()

# Function to predict sentiment
def predict_sentiment(text):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits).item()
    if predicted_class == 0:
        sentiment = "Negative"
    elif predicted_class == 1:
        sentiment = "Neutral"
    else:
        sentiment = "Positive"
    return sentiment

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/result', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        text = request.form['sentence']
        sentiment = predict_sentiment(text)
        return render_template('result.html', text=text, sentiment=sentiment)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
