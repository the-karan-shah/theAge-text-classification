import pandas as pd
import numpy as np
import nltk 
import contractions
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from gensim.models import Word2Vec
import multiprocessing
#nltk.download('wordnet')
#nltk.download('averaged_perceptron_tagger')
import re
import swifter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from joblib import dump, load
import ast

#dummy func for sk-learn's pre-processing 
def dummy_fun(doc):
    return doc


def get_corpus():
    print('Getting Dataset...')
    df = pd.read_csv('theage.csv', converters={"processed_text": ast.literal_eval})
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
    X_train, X_test, y_train, y_test = train_test_split(results, y, test_size=0.2, random_state=1, shuffle = True, stratify=y)

    print('Fitting Random Forest...')
    erf = RandomForestClassifier(n_estimators=1000, random_state=0)
    erf.fit(X_train, y_train)

    #test should be fed to TfidfVectorizer.transform() and then fed to predict. 
    y_pred = erf.predict(X_test)
    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))
    print(accuracy_score(y_test, y_pred))

    print('Saving model...')
    #upgrade to dumping a pipeline
    dump(tfidf, 'tfidf_vectorizier.joblib')
    dump(erf, 'tfidf_rf_model.joblib')


def tfidf_rf_pred(lists):
    tfidf = load('tfidf_vectorizier.joblib')
    output = tfidf.transform(lists)
    return output


def model_update():
    print('Updating Model...')
    df = get_corpus()
    tfidf_rf_model(df)