from flask import Flask, render_template, request
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)
model = pickle.load(open('iris', 'rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('proj1.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        Sl = float(request.form['SepalLengthCm'])
        Sw=float(request.form['SepalwidthCm'])
        pl=float(request.form['petalLengthCm'])
        pw=float(request.form['petalwidthCm'])
        prediction=model.predict([[Sl,Sw,pl,pw]])
        return render_template('proj1.html',prediction_text="type is {}".format(prediction))
    else:
        return render_template("proj1.html")
if __name__=="__main__":
    app.run(debug=True)
