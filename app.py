from flask import Flask, jsonify, request, render_template, url_for, redirect
from joblib import load 
import os
from model import get_corpus, tfidf_rf_model, tfidf_rf_pred
from json import dumps, loads
from preproccesing import lemmatizer, string_normalise, remove_stopwords
from nltk import word_tokenize
from flask_wtf import FlaskForm
from wtforms import TextAreaField, SubmitField, SelectField 
from wtforms.validators import DataRequired
from utilities import  get_individual_article, input_cleaner, create_dict

#app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'karan'

#load the model:
if not os.path.isfile('tfidf_rf_model.joblib'):
    print('Model file not found. Creating new model...')
    df = get_corpus()
    tfidf_rf_model(df)

print('Loading Model...')
model = load('tfidf_rf_model.joblib')


#Configuring the form:
class InputForm(FlaskForm):
    form_input = TextAreaField('Input:', 
        validators=[DataRequired(message="Please enter Text or URL to predict")])
    input_type = SelectField('Input Type:', 
        validators=[DataRequired(message="Please select based on the input provided")], 
        choices=[('text', 'Text'), ('url','URL')])
    submit = SubmitField('Predict')



@app.route('/', methods = ['GET', 'POST'])
def home():
    #form = InputForm()

    if request.method == 'POST':
        input = request.form['input'] 
        input_type = request.form['input_type']

        if (input_type == 'text'):
            processed_text = input_cleaner(input)
            single_sample = []
            single_sample.append(processed_text)
            processed_text = tfidf_rf_pred(single_sample)
            pred = model.predict(processed_text)
            result = pred[0]
        elif (input_type == 'url'):
            processed_text = get_individual_article(input)
            processed_text = input_cleaner(processed_text)
            single_sample = []
            single_sample.append(processed_text)
            processed_text = tfidf_rf_pred(single_sample)
            pred = model.predict(processed_text)
            result = pred[0]
        
        output = create_dict(input, input_type, result)
        
        return render_template('results.html', output = output) #redirect(url_for('predict'))
    
    #return render_template('home.html', form=form)
    return render_template('home.html')

    # #input = loads(test_input)
    # #print('Here is the INPUT:')
    # #print(input)
    # if 'text' in input:
    #     text = input["text"]
    #     processed_text = string_normalise(text)
    #     processed_text = word_tokenize(processed_text)
    #     processed_text = remove_stopwords(processed_text)
    #     processed_text = lemmatizer(processed_text)
    #     print(processed_text)
    #     processed_text =  " ".join(processed_text)
    #     print(processed_text)
    #     #processed_text = tfidf_rf_pred(processed_text)
    #     single_sample = []
    #     single_sample.append(processed_text)
    #     processed_text = tfidf_rf_pred(single_sample)
    #     pred = model.predict(processed_text)
    #     print(pred)

    # return "<h1>HELLO WORLD</h1>" #render_template('index.html')



# @app.route('/prediction')
# def predict():
#     result = pred[0]
#     return render_template('results.html', result)
    # #get the data:
    # payload = request.get_json(force = True)

    # #filter JSON to extract just the data

    # #if to check whether URL or Text

    # #clean the data 
    # data = 0

    # #pass the data to the model:
    # pred = model.predict(data)

    # #get output as JSON:
    # output = {
    #     'results': pred
    # }
    # output = jsonify(output)


if __name__ == '__main__':
    #app.run(port = 5000, debug = True)
    app.run()
