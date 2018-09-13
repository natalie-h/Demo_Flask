from flask import Flask, jsonify, request
from sklearn.linear_model import LogisticRegression
import pandas as pd
import pickle
from sklearn.externals import joblib

app = Flask(__name__)

my_data = [
    {"student": "Anthony", "Speed": 23.5, "Power": "over 9000"},
    {"student": "Ruairi", "Speed": 23.55, "Power": "over 9000"},
    {"student": "Sam", "Speed": 23.2255, "Power": "over 9000"},
    {"student": "Evan", "Speed": 23.52553234, "Power": "over 9000"}]

df = pd.DataFrame(my_data)

logreg = LogisticRegression()
model = logreg.fit(df[['Speed']].values, df['student'].values)

@app.route('/')
def hello_world():
	return 'This is a test for running Flask'

@app.route('/json-test')
def json_test():
    return jsonify(my_data)

@app.route('/predict-student')
def predict_student():

    speed = request.args.get("speed")

    if speed:
        predicted = model.predict([[float(speed)]]).tolist()
        probabilities = model.predict_proba([[float(speed)]]).tolist()
        result = {
            "response": "ok", 
            "predictions": predicted, 
            "probabilities": {student: probabilities[0][index] for index, student in enumerate(model.classes_.tolist())}
        } 
    else:
        result = {"response": "not found", "message": "Please provide a model parameter to predict!"}

    return jsonify(result)

@app.route('/predict-iris')
def predict_iris():

	unmodeled_pickle = joblib.load('logistic_model.pkl') 

	sepal_len = request.args.getlist("sepal_len")
	sepal_width = request.args.getlist("sepal_width")

	if sepal_len and sepal_width:
		for x in sepal_len and sepal_width:
			predicted = unmodeled_pickle.predict([[float(sepal_len), float(sepal_width)]]).tolist()
			def name_flower(predicted):
				for x in predicted:
					if x == 0:
						return 'setosa (0)'
					elif x == 1:
						return 'versicolor (1)'
					elif x == 2:
						return 'virginica (2)'
			probabilities = unmodeled_pickle.predict_proba([[float(sepal_len), float(sepal_width)]]).tolist()
			result = {
				"response": "ok", 
				"predictions": name_flower(predicted),
				"probabilities": {flower: probabilities[0][index] for index, flower in enumerate(unmodeled_pickle.classes_.tolist())}
			} 
	else:
		result = {"response": "not found", "message": "Please provide a model parameter to predict!"}

	return jsonify(result)