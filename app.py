from flask import Flask, jsonify, request, render_template
from joblib import load 
import os
from model import get_corpus, tfidf_rf_model

#app
app = Flask(__name__)

#load the model:
if not os.path.isfile('tfidf_rf_model.joblib'):
    print('Model file not found. Creating new model...')
    df = get_corpus()
    tfidf_rf_model(df)

print('Loading Model...')
model = load('tfidf_rf_model.joblib')

test_data = {'text': 'The discussions are a significant diplomatic escalation between the two powers as economic outcomes are increasingly linked to China retreating on what it views as core domestic security issues. The EU and its major economies, Germany and France, have historically been more cautious than the US and Australia in their approach to Beijing.'}
#routes:
def test_input(test_data):
    return jsonify(test_data)

test_input = test_input(test_data)

@app.route('/', methods = ['GET', 'POST'])
def home(test_input):
    if 'text' in test_input:
        
    return "<h1>HELLO WORLD</h1>" #render_template('index.html')



@app.route('/prediction', methods = ['POST'])
def predict():
    #get the data:
    payload = request.get_json(force = True)

    #filter JSON to extract just the data

    #if to check whether URL or Text

    #clean the data 
    data = 0

    #pass the data to the model:
    pred = model.predict(data)

    #get output as JSON:
    output = {
        'results': pred
    }
    output = jsonify(output)


if __name__ == '__main__':
    app.run(port = 5000, debug = True)
