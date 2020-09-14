from flask import Flask, jsonify, request, render_template
from joblib import load 
import os
from model import tfidf_rf_model
from preproccesing import get_webpage

#load the model:
if not os.path.isfile('tfidf_rf_model.joblib'):
    df = get_webpage('https://www.theage.com.au/')
    tfidf_rf_model(df)

model = load('tfidf_rf_model.joblib')


#app
app = Flask(__name__)

#routes:

@app.route('/')
def home():
    return render_template('index.html')


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
