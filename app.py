#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pickle
from flask import Flask, render_template, jsonify, request


# In[2]:


app = Flask(__name__)
model = pickle.load(open('randomforest.pkl','rb'))


# In[3]:


@app.route('/')
def home():
    return render_template('Rainfall_Prediction.html')


# In[4]:


@app.route('/submit', methods=["POST","GET"])
def submit():
     
	if request.method == "POST":
		
		location = float(request.form['location'])
		rainfall = float(request.form['rainfall'])
		sunshine = float(request.form['sunshine'])
		windGustSpeed = float(request.form['windgustspeed'])
		windDir9am = float(request.form['winddir9am'])
		humidity9am = float(request.form['humidity9am'])
		pressure9am = float(request.form['pressure9am'])
		cloud9am = float(request.form['cloud9am'])
		cloud3am = float(request.form['cloud3am'])
		rainToday = float(request.form['raintoday'])
		feat = [np.array([location,rainfall,sunshine,windGustSpeed,windDir9am,humidity9am,pressure9am,cloud9am,cloud3am,rainToday])]
		pred = model.predict(feat)
		
		if pred == 0:
			return render_template("Rainfall_Prediction.html", prediction_text = 'It is going to be a Sunny Day tomorrow')
		else:
			return render_template("Rainfall_Prediction.html", prediction_text = 'It is going to be a Rainy Day tomorrow')
	

# In[5]:


if __name__ == "__main__":
    app.run(debug=True)


# In[ ]:




