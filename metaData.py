import nltk as nltk
import os
import json
from flask import Flask, render_template,request,jsonify
from flask_cors import CORS
import joblib
app = Flask(__name__)
cors = CORS(app)
import pickle

# @app.route('/')
# def hello_world():
#     return 'Hello World!'
def remove_punctuation(sentence):
    cleaned_req = re.sub(r'[?|!|\'|"|#]', '', sentence)
    cleaned_req = re.sub(r'[,|.|;|:|(|)|{|}|\|/|<|>]|-', ' ', cleaned_req)
    cleaned_req = cleaned_req.replace("\n"," ")
    return cleaned_req

def keep_alphabets(sentence):
    alpha_req = re.sub('[^a-z A-Z]+', ' ', sentence)
    return alpha_req

def lower_case(sentence):
    lower_case_req = sentence.lower()
    return lower_case_req


@app.after_request
def after_request(response):
    response.headers["Cache-Control"] = "no-store"
    return response
@app.route('/')
def index():
    return "<p>Server running </p>"

@app.route('/predict', methods=['POST'])
def predict():
    sentence = request.json['text']
    print(sentence)
    pre_processed_sentence = remove_punctuation(sentence)
    pre_processed_sentence = keep_alphabets(pre_processed_sentence)
    pre_processed_sentence = lower_case(pre_processed_sentence)


    print(pre_processed_sentence)
    multi_label = joblib.load('models/multi_label1.sav')
    lr_classifier = joblib.load('models/lg_classifier-1550.sav')
    tfidf = joblib.load('models/tfidf-1550.sav')
    pre_processed_sentence = [pre_processed_sentence]
    xt = tfidf.transform(pre_processed_sentence)
    lr_classifier.predict(xt)
    predict1 = multi_label.inverse_transform(lr_classifier.predict(xt))
    print (predict1)
    result=jsonify({"preprocessed": pre_processed_sentence},{"prediction": predict1})

    return result

if name == '__main__':
    app.run()