import requests
from bs4 import BeautifulSoup
from lxml import html
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



#getting the overarching categories
def get_categories(links):
    final = []
    for link in links:
        category = link.text
        url = link['href']
        if not url.startswith('http'):
            url = "https://www.theage.com.au"+url #TODO replace hardcoded url with variable
        final.append([category, url])
    return final


#getting the urls of each of the articles in each category
def get_articles(final):
    all_articles = []
    for category, url in final:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        pagelinks = soup.find_all("a", {'data-test':"article-link"})
        for pagelink in pagelinks:
            url_1 = pagelink['href']
            if not url_1.startswith('http'):
                url_1 = "https://www.theage.com.au"+url_1
            all_articles.append([category, url_1])
    return all_articles


#opening each artile in each category and getting the full text
def get_text(all_articles):
    all_texts = []
    for cat, article in all_articles:
        response = requests.get(article)
        soup = BeautifulSoup(response.content, 'html.parser')
        texts = soup.find_all("p")
        full_text = ""
        for i in texts: #make into function? could speed up...
            text = i.text
            text = text + " "
            full_text += text
        all_texts.append(full_text)
    return all_texts   


#string cleaning function
def string_normalise(string):
    string = re.sub('Credit:AP','', string)
    string = re.sub('Copyright © 2020','', string)
    string = re.sub('staff reporters','', string)
    string = re.sub('AP','', string)
    string = re.sub('\$','dollar ', string)
    string = re.sub('per cent','per-cent', string)
    string = re.sub(r"[^-\w\s']", ' ', string)
    #string = re.sub(r"[^-A-Za-z_\s]",' ', string)
    string = re.sub(r"[\d]",' ', string)
    string = re.sub(r"[']", '', string)
    string = string.lower()
    return string


def remove_stopwords(words):
    stopword_list = set(stopwords.words('english'))
    #Remove stop words from list of tokenized words
    new_words = [word for word in words if word not in stopword_list]
    return new_words


def get_wordnet_pos(word):
    #Map POS tag to first character lemmatize() accepts
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

def lemmatizer(list_of_words):
    lemma = WordNetLemmatizer()
    new_words = [lemma.lemmatize(word, get_wordnet_pos(word)) for word in list_of_words]
    return new_words


def create_corpus(links, update = True):
    final = get_categories(links)
    all_articles = get_articles(final)
    all_texts = get_text(all_articles)
    
    df = pd.DataFrame(all_articles, columns=['Categories', 'Url'])
    df['text'] = all_texts
    df = df.drop_duplicates(subset=None, keep="first").reset_index()
    df = df.drop(columns='index')
    
    df['processed_text'] = df.text.swifter.apply(string_normalise)
    df['processed_text'] = df.processed_text.swifter.apply(word_tokenize)
    df['processed_text'] = df.processed_text.swifter.apply(remove_stopwords)
    df['processed_text'] = df.processed_text.swifter.apply(lemmatizer)
    df['word_count'] = df['processed_text'].apply(lambda x: len(str(x).split()))
    df = df[df.word_count > 30]
    df = df.reset_index()
    df = df.drop(columns='index')

    
    if (update):
        dfAge = pd.read_csv('theage.csv')
        df = pd.concat([dfAge, df], ignore_index = True)
        df = df.drop_duplicates(subset=['Categories', 'Url'], keep="first").reset_index()
        df = df.drop(columns='index')
        df.to_csv('theage.csv', index = False)
        return df
    else:
        return df  

def get_webpage(url):
    url = url
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    categories = soup.find("nav",attrs={'class':'_3-9zZ'})
    links = categories.find_all("a")
    df = create_corpus(links)
    return df