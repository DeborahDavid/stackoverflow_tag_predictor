import os
import json
import pickle
import joblib
import pandas as pd
from flask import Flask, request
from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup
import scipy

# create and configure the app
app = Flask(__name__)
app.config.from_object(__name__)
try:
    os.makedirs(app.instance_path)
except OSError:
    pass


# Import the recommendations
clf = joblib.load(open("clf", 'rb'))
tf_transformer = pickle.load(open("tf_transformer", 'rb'))
count_vect = pickle.load(open("count_vect", 'rb'))
tags_list = pickle.load(open("tags_list", 'rb'))
stops = set(stopwords.words("english"))

def review_to_words(raw_body, raw_title, stops):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    review_text = raw_title + " " + raw_body
    #
    # 2. Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()
    #
    #
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]
    #
    # 6. Join the words back into one string separated by space,
    # and return the result.
    return( " ".join( meaningful_words ))

def title_to_words( raw_title, stops):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and
    # the output is a single string (a preprocessed movie review)
    #
    # 2. Remove ? at the end
    raw_title = raw_title.rstrip('\"?"')
    #
    # 3. Convert to lower case, split into individual words
    words = raw_title.lower().split()
    #
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]
    #
    # 6. Join the words back into one string separated by space,
    # and return the result.
    return( " ".join( meaningful_words ))


def proba_to_list(prediction_proba, indice):
    result = {}
    for i, tag in enumerate(tags_list):
        if prediction_proba[i][indice][1]>0.0000000000001:
            result[tag]= prediction_proba[i][indice][1]
    return [x[0] for x in sorted(result.items(), key=lambda kv: kv[1], reverse=True)[:10]]

@app.route('/predict_tags', methods=['GET'])
def predict_tags():
    # Processing
    input_data={}
    for input in ['title','body']:
        input_data[input]=request.args[input]
    input_data["Title_cleaned"]= title_to_words(input_data["title"], stops)
    input_data["text_cleaned"]= review_to_words(input_data["body"],input_data["title"], stops)
    df_custom = pd.DataFrame(pd.Series(input_data)).transpose()
    X_custom_body = tf_transformer.transform(count_vect.transform(df_custom.text_cleaned))
    for tag in tags_list:
        df_custom['title_contain_tag_'+tag]=df_custom.Title_cleaned.apply(lambda x : tag in x)
    X_custom_title = df_custom[['title_contain_tag_'+tag for tag in tags_list]]
    X_custom_title = scipy.sparse.csr_matrix(X_custom_title.values)*10
    X_custom = scipy.sparse.hstack([X_custom_body,X_custom_title])

    # Prediction
    prediction_custom = clf.predict_proba(X_custom)
    return json.dumps(proba_to_list(prediction_custom, 0))
