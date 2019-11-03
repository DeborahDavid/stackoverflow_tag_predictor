import os
import json
import pickle
import pandas as pd
from flask import Flask, request, render_template
import re
from collections import Counter
import scipy

# create and configure the app
app = Flask(__name__)
app.config.from_object(__name__)
try:
    os.makedirs(app.instance_path)
except OSError:
    pass


# Import the recommendations
tf_transformer = pickle.load(open("tf_transformer", "rb"))
count_vect = pickle.load(open("count_vect", "rb"))
tags_list = pickle.load(open("tags_list", "rb"))
svd = pickle.load(open("svd", "rb"))
nbrs = pickle.load(open("nbrs", "rb"))
df_train = pickle.load(open("df_train", "rb"))
stops = pickle.load(open("stops", "rb"))


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
    return " ".join(meaningful_words)


def title_to_words(raw_title, stops):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and
    # the output is a single string (a preprocessed movie review)
    #
    # 2. Remove ? at the end
    raw_title = raw_title.rstrip('"?"')
    #
    # 3. Convert to lower case, split into individual words
    words = raw_title.lower().split()
    #
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]
    #
    # 6. Join the words back into one string separated by space,
    # and return the result.
    return " ".join(meaningful_words)


def indice_to_tags(df_train, indices, i):
    posts_list = indices[i]
    temp_tags = [
        tag
        for sublist in list(df_train.iloc[posts_list].Tags_cleaned)
        for tag in sublist
    ]
    data = Counter(temp_tags)
    return [x[0] for x in data.most_common(10)]


def _predict_tags(title, body):
    input_data = {"title": title, "body": body}
    input_data["Title_cleaned"] = title_to_words(input_data["title"], stops)
    input_data["text_cleaned"] = review_to_words(
        input_data["body"], input_data["title"], stops
    )
    df_custom = pd.DataFrame(pd.Series(input_data)).transpose()
    X_custom_body = tf_transformer.transform(
        count_vect.transform(df_custom.text_cleaned)
    )
    for tag in tags_list:
        df_custom["title_contain_tag_" + tag] = df_custom.Title_cleaned.apply(
            lambda x: tag in x
        )
    X_custom_title = df_custom[["title_contain_tag_" + tag for tag in tags_list]]
    X_custom_title = scipy.sparse.csr_matrix(X_custom_title.values)
    X_custom = scipy.sparse.hstack([X_custom_body, X_custom_title])
    X_custom_svd = svd.transform(X_custom)

    # Prediction
    _, indices = nbrs.kneighbors(X_custom_svd)
    return indice_to_tags(df_train, indices, 0)


@app.route("/predict_tags", methods=["GET"])
def predict_tags():
    return json.dumps(_predict_tags(request.args["title"], request.args["body"]))


@app.route("/formulaire")
def mapbox_gl():
    print("load data")
    return render_template("formulaire.html",)


@app.route("/result", methods=["GET"])
def get_listing():
    result = _predict_tags(request.args["title"], request.args["body"])
    return render_template("formulaire_result.html", result=result)
