import requests
from bs4 import BeautifulSoup
from lxml import html
import pandas as pd
import numpy as np
import re
from preproccesing import string_normalise, word_tokenize, remove_stopwords, lemmatizer


def get_individual_article(article):
    response = requests.get(article)
    soup = BeautifulSoup(response.content, 'html.parser')
    texts = soup.find_all("p")
    full_text = ""
    for i in texts: 
        text = i.text
        text = text + " "
        full_text += text
    
    return full_text

    
def input_cleaner(text):
    processed_text = string_normalise(text)
    processed_text = word_tokenize(processed_text)
    processed_text = remove_stopwords(processed_text)
    processed_text = lemmatizer(processed_text)
    processed_text =  " ".join(processed_text)
    return processed_text


def create_dict(input, input_type, result):
    ''' This function captures all the user inputs and stores them in a dictionary, 
        which can be made into JSON and returned to the results webpage to populate fields. '''
    dictionary = {
        "input": input,
        "input_type": input_type,
        "result": result
    }
    return dictionary
