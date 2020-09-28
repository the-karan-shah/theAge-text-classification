from flask import Flask, jsonify, request, render_template

#app
app = Flask(__name__)

#routes:

@app.route('/', methods = ['GET', 'POST'])
def home():
    return "<h1>HELLO WOLRD</h1>" #render_template('index.html')


if __name__ == '__main__':
    app.run(port = 5000, debug = True)