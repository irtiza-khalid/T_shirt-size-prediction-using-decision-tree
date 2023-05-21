import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

model = pickle.load(open(r"C:\Users\SWishHy\Desktop\flask projects\AI_1\mymodel_knn.pkl", "rb"))


# defining the homepage
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    features = [np.array(int_features)] # provides features in [[feature1, feature2]] form.
    predictions = model.predict(features)

    output = predictions[0]

    return render_template('index.html', prediction_text = "T shirt size is {}".format(output))

def main():
    app.run()

main()