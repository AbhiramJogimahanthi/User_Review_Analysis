from flask import Flask, render_template, flash, request, url_for, redirect, session
import numpy as np
import pandas as pd
import re
import os
import tensorflow as tf
from numpy import array
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import pad_sequences

IMAGE_FOLDER = os.path.join('static', 'img_pool')

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER


# COMMENTED THIS METHOD AS IT'S CAUSING THE ERROR 👇
# def init():
#     global model, graph
#     model = load_model('Sentiment_Analysis_file.h5')
#     graph = tf.compat.v1.get_default_graph()


@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template("home.html")


@app.route('/sentiment_analysis_prediction', methods=['POST', "GET"])
def sent_anly_prediction():
    if request.method == 'POST':
        # ADDED THE MODEL & GRAPH DECLERATION HERE 👇
        model = load_model('Sentiment_Analysis_file.h5')
        graph = tf.compat.v1.get_default_graph()

        text = request.form['text']
        Sentiment = ''
        max_review_length = 500
        word_to_id = imdb.get_word_index()
        strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
        text = text.lower().replace("<br />", " ")
        text = re.sub(strip_special_chars, "", text.lower())

        words = text.split()
        x_test = [[word_to_id[word] if (
            word in word_to_id and word_to_id[word] <= 20000) else 0 for word in words]]
        x_test = pad_sequences(x_test, maxlen=500)
        vector = np.array([x_test.flatten()])

        probability = model.predict(array([vector][0]))[0][0]
        class1 = probability.argmax(axis=-1)
        if probability <= 0.5:
            sentiment = 'Negative'
            img_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'sad.png')
        # if class1 <= 0:
            #sentiment = 'Negative'
            #img_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'sad.png')
        else:
            sentiment = 'Positive'
            img_filename = os.path.join(
                app.config['UPLOAD_FOLDER'], 'smile.png')
    return render_template('home.html', text=text, sentiment=sentiment, probability=probability, image=img_filename)


if __name__ == "__main__":
    # init() 👈 COMMENTED THIS FUNCTION CALL TOO
    app.run()
