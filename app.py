from flask import Flask, jsonify, request, render_template
from joblib import load 
import os
from model import get_corpus, tfidf_rf_model, tfidf_rf_pred
from json import dumps, loads
from preproccesing import lemmatizer, string_normalise, remove_stopwords
from nltk import word_tokenize
#app
app = Flask(__name__)

#load the model:
if not os.path.isfile('tfidf_rf_model.joblib'):
    print('Model file not found. Creating new model...')
    df = get_corpus()
    tfidf_rf_model(df)

print('Loading Model...')
model = load('tfidf_rf_model.joblib')

test_data = {'text' : '''Some romantics keep old photos in a shoebox. Others collect trinkets from the past. For Geraldine Viswanathan, words are what she holds dear."Everything handwritten, I've kept," said the Australian actress, who made a major leap from Blockers breakout to rom-com heroine in Sony's The Broken Hearts Gallery, released in cinemas on Thursday. "Every single birthday card. I'll even write down what people say sometimes because I'm a psycho. But any words that I want to remember, I hold on to forever." As a teenager she knew that memories can be ephemeral. So Viswanathan, now 25 and living in the US, had a ritual: she'd inscribe happy moments on sticky notes and keep them in a jar, and read them again at the end of the year. "Those little things that made your day, it's easy to forget them," she mused. "I should start doing that again. It was a good tradition." Lucy, the 20-something New York City gallery assistant Viswanathan plays in The Broken Hearts Gallery, needs something a little more concrete. An outgoing aspiring curator who wears her heart on her sleeve, she clings so hard to souvenirs from failed relationships that her apartment is cluttered with mundane mementos of her exes: plane tickets, bags of string, even an assemblage of emotionally vivid doorknobs. "Doorknobs! I wonder where she gets all this stuff," Viswanathan laughed. Viswanathan's co-star Dacre Montgomery is also Australian, and shot to fame in 2017 with a role in Stranger Things.  Viswanathan's co-star Dacre Montgomery is also Australian, and shot to fame in 2017 with a role in Stranger Things. Nursing a freshly broken heart, Lucy befriends the emotionally closed-off Nick (fellow Australian actor Dacre Montgomery, of Stranger Things), who's renovating his dream boutique hotel, and builds a gallery in his lobby enshrining tokens of loves lost. Artefacts pile up as strangers exorcise their romantic demons by donating to the collection, but nothing quite comes close to one item Viswanathan saw while researching the real museum in Croatia that inspired the story. "I remember seeing a wound scab in the museum and that really affected me, because wow – that's really intense," she said with a laugh. It's early morning and Viswanathan is calling from the Palm Springs set of a secret project she can't say much about. The project has, however, led to her spending some quality time with donkeys on a desert ranch, which tickles her and also explains a recent Instagram post in which she's posing with an Equus asinus, grinning mischievously, with the caption, "Check out my ass." As she describes the creative freedom writer-director Natalie Krinsky gave her on set and the colourful lines about vibrators she improvised with co-stars Molly Gordon and Phillipa Soo, who play Lucy's best friends, Viswanathan interrupts herself with an excited shout. "Oh! The donkeys are arriving. I wish I could show you, I'm truly surrounded by donkeys!"'''
     }

test_input = dumps(test_data)
print(test_input)
@app.route('/', methods = ['GET', 'POST'])
def home():
    input = loads(test_input)
    print('Here is the INPUT:')
    print(input)
    if 'text' in input:
        text = input["text"]
        processed_text = string_normalise(text)
        processed_text = word_tokenize(processed_text)
        processed_text = remove_stopwords(processed_text)
        processed_text = lemmatizer(processed_text)
        print(processed_text)
        processed_text =  " ".join(processed_text)
        print(processed_text)
        #processed_text = tfidf_rf_pred(processed_text)
        single_sample = []
        single_sample.append(processed_text)
        processed_text = tfidf_rf_pred(single_sample)
        pred = model.predict(processed_text)
        print(pred)

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
