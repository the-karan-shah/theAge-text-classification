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
from joblib import dump

#dummy func for sk-learn's pre-processing 
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
    results = tfidf.fit_transform(df['processed_text'])
    X_train, X_test, y_train, y_test = train_test_split(results, y, test_size=0.2, random_state=1, shuffle = y)

    print('Fitting Random Forest...')
    erf = RandomForestClassifier(n_estimators=1000, random_state=0)
    erf.fit(X_train, y_train)

    y_pred = erf.predict(X_test)

    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))
    print(accuracy_score(y_test, y_pred))

    print('Saving model...')
    dump(erf, 'tfidf_rf_model.joblib')


