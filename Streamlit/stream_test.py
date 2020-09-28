import streamlit as st
import pandas as pd
#import SessionState
from joblib import load, dump
from preprocessing import string_normalise, word_tokenize, remove_stopwords, lemmatizer, get_individual_article, input_cleaner
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os 


st.title('The Age Newspaper Classifier')
st.markdown("By Karan Shah")

st.markdown("Please enter either a URL or the text from a The Age article, and this app will return the category of the article from ...")

input = st.text_area('Input your article here:') 
selection = st.selectbox('What is the input format?', ['URL', 'Text'])


def dummy_fun(doc):
    return doc


def get_corpus():
    print('Getting Dataset...')
    df = pd.read_csv('theage.csv')
    return df


def tfidf_rf_model(df):

    #initialising tfidf vectorizer:
    print('Initialising TF-IDF...')
    tfidf = TfidfVectorizer(
        analyzer='word',
        tokenizer=dummy_fun,
        preprocessor=dummy_fun,
        token_pattern=None)

    X = df['processed_text']
    y = df['Categories']
    
    print('Fitting TF-IDF...')
    results = tfidf.fit_transform(df['processed_text']) #incorrect, needs to be separated first.
    X_train, X_test, y_train, y_test = train_test_split(results, y, test_size=0.2, random_state=1, shuffle = y)

    print('Fitting Random Forest...')
    erf = RandomForestClassifier(n_estimators=1000, random_state=0)
    erf.fit(X_train, y_train)

    y_pred = erf.predict(X_test)
#https://towardsdatascience.com/deploying-models-to-flask-fb62155ca2c4
    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))
    print(accuracy_score(y_test, y_pred))

    print('Saving model...')
    #upgrade to dumping a pipeline
    dump(tfidf, 'tfidf_vectorizier.joblib')
    dump(erf, 'tfidf_rf_model.joblib')


if not os.path.isfile('tfidf_rf_model.joblib'):
    print('Model file not found. Creating new model...')
    df = get_corpus()
    tfidf_rf_model(df)


#session = SessionState.get(button_predict = False)
tfidf = load('tfidf_vectorizier.joblib')
model = load('tfidf_rf_model.joblib')


def tfidf_rf_pred(lists): 
    output = tfidf.transform(lists)
    return output


btn_predict = st.button('Predict')
if btn_predict:
    if selection == 'Text':
        processed_text = input_cleaner(input)
        single_sample = []
        single_sample.append(processed_text)
        processed_text = tfidf_rf_pred(single_sample)
        pred = model.predict(processed_text)
        st.header('The prediction for your article is: ' + pred[0])

    if selection == 'URL':
        processed_text = get_individual_article(input)
        processed_text = input_cleaner(processed_text)
        single_sample = []
        single_sample.append(processed_text)
        processed_text = tfidf_rf_pred(single_sample)
        pred = model.predict(processed_text)
        st.header('The prediction for your article is: ' + pred[0])



    