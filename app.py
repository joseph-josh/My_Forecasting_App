from flask import Flask, json, render_template, jsonify, session, request, redirect, url_for
from pandas.core.indexes.base import Index
from flask_session import Session
from datetime import timedelta
from flask_restful import Resource
import models
import pandas as pd
from flask_dropzone import Dropzone
import os
import re
import json

sess = Session()

basedir = os.path.abspath(os.path.dirname(__file__))


app = Flask(__name__)

app.config.update(
    UPLOADED_PATH=os.path.join(basedir, 'uploads'),
    # Flask-Dropzone config:
    DROPZONE_MAX_FILE_SIZE=1000,
    DROPZONE_ALLOWED_FILE_TYPE = 'image',
    DROPZONE_MAX_FILES=1,
    DROPZONE_REDIRECT_VIEW='configuration', 
    DROPZONE_DEFAULT_MESSAGE= "<i class='notika-icon notika-cloud' ></i><h4>Drop files here or click to upload.</h4>"  # set redirect view
)

dropzone = Dropzone(app)
  

@app.route('/', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        f = request.files.get('file')
        data = pd.read_csv(f)

        session["data"] = f
        return jsonify(data.to_json(orient="split"))
    return render_template('index.html')


@app.route("/configuration", methods= ["GET", "POST"])
def configuration():

    data = session.get('data', None)
    return jsonify(data)

    # results = models.columns_names(data)
    # columns = results["columns"]

    # session.permanent = True

    # if request.method == "POST":
    #     session.permanent = True
    #     form_data = request.form
    #     session['configuration'] = form_data
    #     return redirect("/detailed_ranking")
    # return render_template("configuration.html", columns = columns)


@app.route("/detailed_ranking")
def detailed_ranking():

    configuration = session.get('configuration', None)
    data = session.get('data', None)

    results = models.global_function(configuration, data)

    session.permanent = True
    session['metrics'] = results['metrics']
    session['labelencoder'] = results['labelencoder']
    session['onehotencoder'] = results['onehotencoder']
    session['cat_cols'] = results['cat_cols']
    session['num_cols'] = results['num_cols']
    session['cat_and_values'] = results['cat_and_values']
    session['prediction_reg_tree'] = results['prediction_reg_tree']

    return render_template("detailed_ranking.html", results = results)
    #return results


@app.route("/simple_ranking")
def simple_ranking():
    metrics = session.get('metrics', None)
    return render_template("simple_ranking.html", metrics = metrics)


@app.route("/eda")
def eda():
    data = session.get('data', None)
    models.eda(data)
    return render_template("eda/your_report.html")


@app.route("/ml_prediction", methods= ["GET", "POST"])
def ml_prediction():

    if request.method == "POST":
        form_data_pred = request.form
        
        lin_reg = models.load('ml_models/lin_reg.joblib') 
        svr = models.load('ml_models/svr.joblib')
        xgb_reg = models.load('ml_models/xgb_reg.joblib')
    
        predictors = {}

        for name, value in form_data_pred.items():
            if re.match('[a-zA-Z_]+', value):
                predictors[name] = value
            else:
                predictors[name] = float(value)

        predictors_df = pd.DataFrame.from_dict(predictors, orient="index").T


        labelencoder = session.get('labelencoder', None)
        onehotencoder = session.get('onehotencoder', None)
        cat_cols = session.get('cat_cols', None)
        prediction_reg_tree = session.get('prediction_reg_tree', None)

        OH_cols_predictors_df = models.getEncoded(predictors_df[cat_cols], labelencoder, onehotencoder)
        OH_cols_predictors_df = pd.DataFrame(OH_cols_predictors_df)

        num_predictors_df = predictors_df.drop(cat_cols, axis=1)
        OH_predictors_df = pd.concat([num_predictors_df, OH_cols_predictors_df], axis=1)
        
        prediction_reg_tree = [prediction_reg_tree]

        prediction = {
            "lin_reg": lin_reg.predict(OH_predictors_df).tolist(),
            "svr": svr.predict(OH_predictors_df).tolist(),
            "xgb_reg": xgb_reg.predict(OH_predictors_df).tolist(),
            "reg_tree": prediction_reg_tree
        }

        return render_template("ml_prediction_results.html", prediction = prediction)
        

    num_cols = session.get('num_cols', None)
    cat_and_values = session.get('cat_and_values', None)

    form_data = {
        "num_cols": num_cols,
        "cat_and_values": cat_and_values
    }

    return render_template("ml_prediction_form.html", form_data = form_data)



if __name__ == "__main__":
    # Quick test configuration. Please use proper Flask configuration options
    # in production settings, and use a separate file or environment variables
    # to manage the secret key!
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    #app.config['DROPZONE_ALLOWED_FILE_CUSTOM'] = True
    #app.config['DROPZONE_ALLOWED_FILE_TYPE'] = '.csv'
    app.config['PERMANENT_SESSION_LIFETIME'] =  timedelta(minutes=300000)

    sess.init_app(app)

    app.run(port= 5000, debug=True)