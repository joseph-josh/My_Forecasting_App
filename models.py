
### ONE-STEP-AHEAD FORECASTING

# #### Forecasting models:
# 
# - Naive model
# - Cumulative model
# - Moving average
# - Exponential smoothing
# - Holt 
# - Holt-Winter model
# 
# - Linear regression
# - Support Vector Regression
# - Regression tree
# - XG Boost regression
# 
# 
# 
# #### Metrics for models' accuracy assessment
#
# - RMSE
# - MAE
# - ME
#
#
#
# #### Resources:
# 
# - https://www.kaggle.com/koraycalisir/hearth-atack-possiblity-machine-learning-eda
# - https://www.analyticsvidhya.com/blog/2018/02/time-series-forecasting-methods/?
# - https://www.statsmodels.org/stable/examples/notebooks/generated/exponential_smoothing.html
# - One-step-ahead forecasting and multiple-steps forecasting <br>
# - https://machinelearningmastery.com/multi-step-time-series-forecasting/#:~:text=Multi%2DStep%20Forecasting,-Generally%2C%20time%20series&text=This%20is%20called%20a%20one,step%20time%20series%20forecasting%20problems.



import pandas as pd
import numpy as np

import matplotlib.pyplot as plt 
import seaborn as sns

from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import sklearn as skit
from sklearn.tree import DecisionTreeRegressor
from sklearn import svm
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
# import tensorflow as tf
# from tensorflow import keras

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score
from math import sqrt, floor
from pandas_profiling import ProfileReport
from joblib import dump, load


import datetime


def columns_names():
    
    data = pd.read_csv("uploads/data.csv", nrows = 500)
    # , nrows = 2000
    print(type(data))

    columns = list(data.columns)

    results = {
        "columns": columns,
        "data": data
    }

    return results


def eda(data):

    ## Exploratory Data Analysis
    profile = ProfileReport(data, title="Pandas Profiling Report")
    profile.to_file("templates/eda/your_report.html")


def getEncoded(data,labelencoder_dict,onehotencoder_dict):
    data_encoded_x = None
    for i in range(0,data.shape[1]):
        label_encoder =  labelencoder_dict[i]
        feature = label_encoder.transform(data.iloc[:,i])
        feature = feature.reshape(data.shape[0], 1)
        onehot_encoder = onehotencoder_dict[i]
        feature = onehot_encoder.transform(feature)
        if data_encoded_x is None:
            data_encoded_x = feature
        else:
            data_encoded_x = np.concatenate((data_encoded_x, feature), axis=1)
    return data_encoded_x


def global_function(configuration, data):

    # Importing the datafloor(len(train_tm)/seasonal_periods)
    # variable time : name of the column representing the date

    columns = list(data.columns)


    ## Getting configurations 
    target = []
    predictors = []
    categoricals = []
    time_interval = configuration["time_interval"]

    for key, value in configuration.items():
        if "target" in key:
            target.append(columns[int(value) - 1])
        elif "predictor" in key:
            predictors.append(columns[int(value) - 1])
        elif "categorical" in key:
            categoricals.append(columns[int(value) - 1])
    
    print(target, predictors, categoricals)
    target = target[0]


    # Renaming the target column to "cnt"
    data.rename(columns={target: "cnt"}, inplace=True)

    print(data.columns)



    # # Dropping duplicate rows
    # Dropping the first duplicate row of all the binomials

    data.drop_duplicates(keep ="last", inplace = True)



    # Using a fake date as the new index of the data
    index = np.arange(start=1, stop=len(data)+1, step=1)

    #data.set_index("timestamp", inplace=True)
    data["new_index"] = index

    def to_data(new_index):
        timestamp = datetime.datetime.fromtimestamp(new_index)
        return timestamp.strftime('%Y-%m-%d %H:%M:%S')

    data["new_index"] = data["new_index"].apply(to_data)
    data.set_index("new_index", inplace=True)




    # Drop a recording only when its field "target" is null
    # Target is the variable to forecast

    # This dataset will be used to build models that use only the target as input
    # data_with_target_missing_records_dropped
    data_tm = data.dropna(subset=["cnt"])

    # This dataset will be used to build models with multiple features in input
    # data_with_missing_values_records_dropped
    data_vm = data.dropna()


    ## Preparing data for ml_prediction_form

    num_cols = list(set(predictors) - set(categoricals))
    cat_cols = categoricals

    cat_and_values = {}

    for cat in cat_cols:
        cat_and_values[cat] =list(data_vm[cat].unique())



    # # Creating the training and testing datasets

    len_data_tm = len(data_tm)
    len_data_vm = len(data_vm)

    # print("Length of data_tm: ", len_data_tm)
    # print("Length of data_vm: ", len_data_vm)

    # 80% for training set and 20% for testing set
    split_rate = 0.5 

    limit_tm = round(len_data_tm*split_rate) 
    limit_vm = round(len_data_vm*split_rate)

    train_tm = data_tm[:limit_tm] 
    test_tm = data_tm[limit_tm:]

    train_vm = data_vm[:limit_vm] 
    test_vm = data_vm[limit_vm:]



    # ### Model evaluation

    def model_evaluation(model_title, target, y_hat, prediction, multi_param_model):
        # Root mean squared error
        rmse = sqrt(mean_squared_error(target, y_hat))
        
        # Mean absolute error
        mae = mean_absolute_error(target, y_hat)
        
        # Mean error
        me = (target - y_hat).mean()
        
        results = pd.DataFrame.from_dict({"Model": [model_title], "RMSE": [rmse], "MAE": [mae], "ME":[me]})
        
        if multi_param_model == False:
            models_names = list(prediction.keys())
            prediction_values = list(prediction.values())
            prediction_df = pd.DataFrame.from_dict({"Model": models_names, "Next Period Forecast": prediction_values})
            results = pd.merge(results, prediction_df, on="Model")

        return results   


    def best_model_selection(param_values, y_hat, prediction, multi_param_model=True):

        param_metrics = pd.DataFrame()

        for param_name in param_values.keys():
            param_metrics = param_metrics.append(model_evaluation(param_name, test_tm["cnt"], y_hat[param_name], prediction, multi_param_model), ignore_index="True")

        param_metrics = param_metrics.sort_values("RMSE")
        param_metrics.reset_index(inplace=True)
        
        models_names = list(prediction.keys())
        prediction_values = list(prediction.values())
        prediction_df = pd.DataFrame.from_dict({"Model": models_names, "Next Period Forecast": prediction_values})
        param_metrics = pd.merge(param_metrics, prediction_df, on="Model")
        
        best_m_av_model = param_metrics.loc[0]["Model"]

        return {"metrics": param_metrics,
                "best_model":best_m_av_model
            }



    models_forecasts = {}


    # # Models implementation
    # ### Naive model

    data_test = np.asarray(test_tm["cnt"]) 
    y_hat = test_tm.copy()

    y_hat["naive"] = train_tm["cnt"][len(train_tm)-1]
    y_hat["naive"][1:] = data_test[0:len(data_test)-1]

    prediction = {}
    prediction["naive"] = data_tm["cnt"][len(data_tm)-1]

    multi_param_model = False
    naive_metrics = model_evaluation("naive", test_tm["cnt"], y_hat["naive"], prediction, multi_param_model)
    models_forecasts['naive'] = list(y_hat["naive"])
    naive_metrics       



    # ### Cumulative model

    y_hat = test_tm.copy()
    y_hat["cumulative"] = train_tm["cnt"]

    prediction = {} 

    for i in range(0, len(test_tm)):
        y_hat["cumulative"][i] = data_tm["cnt"][:len(train_tm)+i].mean()
           
    prediction["cumulative"] = data_tm["cnt"][:len(data_tm)].mean()

    multi_param_model = False
    cumulative_metrics = model_evaluation("cumulative", test_tm["cnt"], y_hat["cumulative"], prediction, multi_param_model)
    models_forecasts["cumulative"] = list(y_hat["cumulative"])
    cumulative_metrics




    # ### Moving average

    y_hat = test_tm.copy()

    n_values = {"m_av_2": 2,
                "m_av_4": 4,
                "m_av_6": 6,
                "m_av_8": 8,
                "m_av_16": 16,
                "m_av_32": 32,
                "m_av_64": 64,
                "m_av_100": 100}         

    prediction = {}

    for m_av,n in n_values.items():
        
        y_hat[m_av] = test_tm["cnt"]

        for i in range(0, len(test_tm)):
            y_hat[m_av][i] = round(data_tm["cnt"][len(train_tm)-n+i: len(train_tm)+i].mean())
            
            if i==len(test_tm)-1:
                prediction[m_av] = round(data_tm["cnt"][len(train_tm)-n+i+1: len(train_tm)+i+1].mean())

        models_forecasts[m_av] = list(y_hat[m_av])


    results = best_model_selection(n_values, y_hat, prediction)
    m_av_metrics = results["metrics"]
    
    m_av_metrics

    best_m_av_model = results["best_model"]
    


    # ### Simple exponential smoothing

    ses_model = SimpleExpSmoothing(np.asarray(test_tm['cnt']), initialization_method="estimated").fit()

    prediction = {"ses": ses_model.forecast(1)[0]}

    multi_param_model = False
    ses_model_metrics = model_evaluation("ses", test_tm["cnt"], ses_model.fittedvalues, prediction, multi_param_model)
    ses_model_metrics
    models_forecasts["ses"] = list(ses_model.fittedvalues)



    # ### Holt model

    # ##### Holt model: using the optimizer alone

    h_simple_model = Holt(np.asarray(test_tm['cnt']), initialization_method="estimated").fit()

    prediction = {"h_simple_model": h_simple_model.forecast(1)[0]}

    multi_param_model = False
    h_simple_metrics = model_evaluation("h_simple_model", test_tm["cnt"], h_simple_model.fittedvalues, prediction, multi_param_model)
    h_simple_metrics
    models_forecasts["h_simple_model"] = list(h_simple_model.fittedvalues)



    # ##### Holt exponential model
    h_exp_model = Holt(np.asarray(test_tm['cnt']), exponential=True, initialization_method="estimated").fit()

    prediction = {"h_model_exp": h_exp_model.forecast(1)[0]}

    multi_param_model = False
    h_exp_metrics = model_evaluation("h_model_exp", test_tm["cnt"], h_exp_model.fittedvalues, prediction, multi_param_model)
    h_exp_metrics
    models_forecasts["h_model_exp"] = list(h_exp_model.fittedvalues)



    # ##### Holt additive model damped 

    h_add_d_model = Holt(np.asarray(test_tm['cnt']), damped_trend=True, initialization_method="estimated").fit()

    prediction = {"h_add_d_model": h_add_d_model.forecast(1)[0]}

    multi_param_model = False
    h_add_d_metrics = model_evaluation("h_add_d_model", test_tm["cnt"], h_add_d_model.fittedvalues, prediction, multi_param_model)
    h_add_d_metrics
    models_forecasts["h_add_d_model"] = list(h_add_d_model.fittedvalues)



    # ### Holt-Winter model

    # Holt-Winter model additive
    hw_add_model = ExponentialSmoothing(np.asarray(test_tm['cnt']), initialization_method="estimated").fit()
    # Holt-Winter model multiplicative
    hw_mult_model = ExponentialSmoothing(np.asarray(test_tm['cnt']), seasonal_periods=24, trend='add', seasonal='mul', use_boxcox=True, initialization_method="estimated").fit()
    # Holt-Winter model additive damped
    hw_add_d_model = ExponentialSmoothing(np.asarray(test_tm['cnt']), seasonal_periods=24, trend='add', seasonal='add', damped_trend=True, use_boxcox=True, initialization_method="estimated").fit()

    prediction_add = {"hw_add_model": hw_add_model.forecast(1)[0]}
    prediction_mult = {"hw_mult_model": hw_mult_model.forecast(1)[0]}
    prediction_add_d = {"hw_add_d_model": hw_add_d_model.forecast(1)[0]}

    multi_param_model = False
    hw_add_metrics = model_evaluation("hw_add_model", test_tm["cnt"], hw_add_model.fittedvalues, prediction_add, multi_param_model)
    hw_mult_metrics = model_evaluation("hw_mult_model", test_tm["cnt"], hw_mult_model.fittedvalues, prediction_mult, multi_param_model)
    hw_add_d_metrics = model_evaluation("hw_add_d_model", test_tm["cnt"], hw_add_d_model.fittedvalues, prediction_add_d, multi_param_model)

    hw_add_metrics
    hw_mult_metrics
    hw_add_d_metrics

    models_forecasts["hw_add_model"] = list(hw_add_model.fittedvalues)
    models_forecasts["hw_mult_model"] = list(hw_mult_model.fittedvalues)
    models_forecasts["hw_add_d_model"] = list(hw_add_d_model.fittedvalues)





    ## Machine learning models implementation

    #print(train_vm.head())

    features = predictors


    x_train  = train_vm[features]
    x_test = test_vm[features]

    x_cols_train_test_combined = pd.concat([x_train, x_test], ignore_index=True)

    y_train = train_vm["cnt"]
    y_test = test_vm["cnt"]

    
    ### Dealing with categorical columns


    if 0 > len(cat_cols):
        OH_x_train = x_train
        OH_x_test = x_test

    else:
        labelencoder_dict = {}
        onehotencoder_dict = {}
        OH_x_cols_train_test_combined = None
        new_data = x_cols_train_test_combined[cat_cols]

        for i in range(0, new_data.shape[1]):
            label_encoder = LabelEncoder()
            labelencoder_dict[i] = label_encoder
            feature = label_encoder.fit_transform(new_data.iloc[:,i])
            feature = feature.reshape(new_data.shape[0], 1)
            onehot_encoder = OneHotEncoder(sparse=False)
            feature = onehot_encoder.fit_transform(feature)
            onehotencoder_dict[i] = onehot_encoder

            if OH_x_cols_train_test_combined is None:
                OH_x_cols_train_test_combined = feature
            else:
                OH_x_cols_train_test_combined = np.concatenate((OH_x_cols_train_test_combined, feature), axis=1)
        OH_x_cols_train_test_combined = pd.DataFrame(OH_x_cols_train_test_combined)


        num_x_cols_train_test_combined = x_cols_train_test_combined.drop(cat_cols, axis=1)
        OH_x_train_test_combined = pd.concat([num_x_cols_train_test_combined, OH_x_cols_train_test_combined], axis=1)

        OH_x_train = OH_x_train_test_combined[:limit_vm] 
        OH_x_test = OH_x_train_test_combined[limit_vm:]



    ### Linear Regression

    lin_reg = LinearRegression()
    lin_reg.fit(OH_x_train, y_train)
    dump(lin_reg, 'ml_models/lin_reg.joblib')

    # Making forecasting
    forecasts = lin_reg.predict(OH_x_test)

    prediction = {"lin_reg": "None"}

    multi_param_model = False
    lin_reg_metrics = model_evaluation("lin_reg", y_test, forecasts, prediction, multi_param_model)
    models_forecasts["lin_reg"] = list(forecasts)



    ### Support Vector Machine

    svr = svm.SVR(C=1, kernel='linear', degree=8, gamma='scale', coef0=10)
    svr.fit(OH_x_train, y_train)
    dump(svr, 'ml_models/svr.joblib')

    # Making forecasting
    forecasts = svr.predict(OH_x_test)

    prediction = {"svr": "None"}

    multi_param_model = False
    svr_metrics = model_evaluation("svr", y_test, forecasts, prediction, multi_param_model)
    svr_metrics
    models_forecasts["svr"] = list(forecasts)



    ### XGBoost Regression

    xgb_reg = XGBRegressor()
    xgb_reg.fit(OH_x_train, y_train)
    dump(xgb_reg, 'ml_models/xgb_reg.joblib')

    # Making forecasting
    forecasts = xgb_reg.predict(OH_x_test)

    prediction = {"xgb_reg": "None"}

    multi_param_model = False
    xgb_reg_metrics = model_evaluation("xgb_reg", y_test, forecasts, prediction, multi_param_model)
    xgb_reg_metrics
    models_forecasts["xgb_reg"] = list(forecasts)



    ### Regression Tree

    # We will train the model by providing the data with a specific layout:
    # - 50 consecutive rows of input request (x_len = 50)
    # - The request for the next row out (y_len = 1)  <br>
    # The algorithm will learn the relationship between the last 50 rows of demand and the demand of the next row.

    ## Defining a data splitting function so as to create this specific layout

    # y_test_len is the number of rows in y_test

    def datasets(df, y_test_len, x_len=50, y_len=1):

        D = df.values
        periods = D.shape[1]

        # Training set creation: run through all the possible time windows
        loops = periods + 1 - x_len - y_len - y_test_len 
        train = []
        
        for col in range(loops):
            train.append(D[:,col:col+x_len+y_len])
        
        train = np.vstack(train)
        X_train, Y_train = np.split(train,[x_len],axis=1)

        # Test set creation: unseen “future” data with the demand just before
        max_col_test = periods - x_len - y_len + 1
        test = []
        for col in range(loops,max_col_test):
            test.append(D[:,col:col+x_len+y_len])
        test = np.vstack(test)
        X_test, Y_test = np.split(test,[x_len],axis=1)

        # this data formatting is needed if we only predict a single period
        if y_len == 1:
            Y_train = Y_train.ravel()
            Y_test = Y_test.ravel()
            
        return X_train, Y_train, X_test, Y_test



    ## Create a pivot of the data to show the periods as columns
    pivoted_data_vm = pd.DataFrame(data_vm['cnt'])
    pivoted_data_vm = pd.pivot_table(data=pivoted_data_vm, values="cnt", columns="new_index", aggfunc='sum', fill_value=0) 


    # Creation of the training and testing data
    y_test_len = len_data_vm - round(len_data_vm*split_rate)
    X_train, Y_train, X_test, Y_test = datasets(pivoted_data_vm, y_test_len)
    #print(X_train.shape, X_test.shape)

    # Creation of the regression tree model
    reg_tree = DecisionTreeRegressor(max_depth=5,min_samples_leaf=5) 
    reg_tree.fit(X_train,Y_train) 
    
    # Making forecastings using the test data
    forecasts = reg_tree.predict(X_test) 

    # Next period forecast
    last_50_cnt_of_data = list(pivoted_data_vm.iloc[0])
    length = len(last_50_cnt_of_data)
    last_50_cnt_of_data = last_50_cnt_of_data[length-50:length]


    prediction_reg_tree = reg_tree.predict([last_50_cnt_of_data])[0]

    prediction = {"reg_tree": prediction_reg_tree}

    multi_param_model = False
    reg_tree_metrics = model_evaluation("reg_tree", Y_test, forecasts, prediction, multi_param_model)
    reg_tree_metrics
    models_forecasts["reg_tree"] = list(forecasts)




    ### Models ranking by RMSE

    m_list = [naive_metrics, cumulative_metrics, m_av_metrics, ses_model_metrics, h_add_d_metrics,h_exp_metrics, 
          h_simple_metrics, hw_add_metrics, hw_mult_metrics, hw_add_d_metrics, lin_reg_metrics, reg_tree_metrics, svr_metrics, xgb_reg_metrics ]

    metrics_sum = pd.concat(m_list, ignore_index=True)
    metrics_sum.drop(columns=["index"], inplace=True)
    metrics_sum.sort_values("RMSE", inplace=True)



    ### Ranking the forecast lists of the models by RMSE

    ranked_models_forecasts = {}

    for i in range(len(metrics_sum)):
        model_name = metrics_sum.iloc[i]['Model']
        ranked_models_forecasts[model_name] = models_forecasts[model_name]



    ### Preparing the object to be returned

    train_data_index = list(np.arange(start=1, stop=len(data_vm)+1, step=1))
    test_data_index = list(np.arange(start=len(train_vm)+1, stop=len(data_vm)+1, step=1))
            
    results = {
        "data": data, 
        "train_data_index": train_data_index,
        "test_data_index": test_data_index,
        "train_data" : list(train_tm["cnt"]),
        "y_test" : list(y_test),
        "forecasts": ranked_models_forecasts,
        "metrics": metrics_sum.set_index('Model').T.to_dict('list'),
        "labelencoder": labelencoder_dict, 
        "onehotencoder": onehotencoder_dict,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "cat_and_values": cat_and_values,
        "prediction_reg_tree": prediction_reg_tree
    }


    
    return results










