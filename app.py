from flask import Flask, jsonify, request
import joblib

#load the model:
model = joblib.load()

#app
app = Flask(__name__)

#routes:
@app.route('/', methods = ['POST'])
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
