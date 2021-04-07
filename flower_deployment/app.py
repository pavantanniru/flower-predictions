


from flask import Flask ,render_template,request,jsonify,url_for,session,redirect
import numpy as np
from tensorflow.keras.models import load_model
from flask_wtf import FlaskForm

from wtforms import TextField,SubmitField




import  joblib
import os
import pickle

flower_model = load_model("final_iris_model.h5")
flower_scaler = joblib.load("iris_scaler.pkl")

def return_prediction(model,scaler,sample_json):

    # For larger data features, you should probably write a for loop
    # That builds out this array for you

    s_len = sample_json['sepal_length']
    s_wid = sample_json['sepal_width']
    p_len = sample_json['petal_length']
    p_wid = sample_json['petal_width']

    flower = [[s_len,s_wid,p_len,p_wid]]

    flower = scaler.transform(flower)

    classes = np.array(['setosa', 'versicolor', 'virginica'])

    class_ind = model.predict_classes(flower)

    return classes[class_ind][0]



app = Flask(__name__)

app.config['SECRET_KEY'] = 'my_Key'


class MyForm(FlaskForm):
    sep_len = TextField("Sepal length")
    sep_wid = TextField("sepal width")
    ped_len = TextField("petal length")
    ped_wid = TextField("petal width")
    submit = SubmitField("Analyze")


@app.route("/",methods=["GET","POST"])

def index():
    form = MyForm()

    if form.validate_on_submit():
        session['sep_len'] = form.sep_len.data
        session['sep_wid'] = form.sep_wid.data
        session['ped_len'] = form.ped_len.data
        session['ped_wid'] = form.ped_wid.data

        return redirect(url_for("predictions"))

    return render_template('index.html',form=form)




@app.route("/predictions")

def predictions():

    content = {}

    content['sepal_length'] = float(session['sep_len'])
    content['sepal_width'] = float(session['sep_wid'])
    content['petal_length'] = float(session['ped_len'])
    content['petal_width'] = float(session['ped_wid'])


    results = return_prediction(flower_model ,flower_scaler,content)

    return render_template('predictions.html',results = results)


@app.route("/api/flower",methods=["POST"])

def flower_prediction():
    content = request.json
    result = return_prediction(flower_model,flower_scaler,content)
    return jsonify(result)




if __name__ == '__main__':
    app.run(debug=True)
