This web app takes in input a file containing chronological data and ranks the forecasting models used to predict the next period value based on RMSE. 

<h2>Introduction</h2>
While working on some demand forecasting projects, I realized that I was spending a fairly large amount of time implementing various models to predict the demand of the following period before choosing the one that performs the most. And if I want to do the same thing for another demand dataset, I needed to go back to the code and modified many parts of it, which also consumes time. So I came up with the idea of creating a forecasting web application whose objective would be to automatically recommend the most performant model to be used for the next period demand prediction, and this, for any kind of data, while minimizing the execution time. And with regards to the variable being forecasted, apart from the demand of a product, it can be any indicators that vary chronologically such as sales, stock price, weather, migration, and earthquake.

<h2>Demo Video</h2>
Take a look at the demo video: <a href="https://www.youtube.com/watch?v=OA5WAaVd5xE">Demo Video.</a>

<h2>App Interfaces</h2>

The app consists of 6 user-friendly interfaces:
<img src="https://cdn-images-1.medium.com/max/1200/1*20zxzSs6dOALCiUITY1N8w.png">

<br>
<br>
<b>Home</b>
<br>
This is the first page of the application, where the user uploads the file containing chronological data.
<br>
<br>
<b>Configuration</b>
<br>
When the file is successfully uploaded, the user is redirected to the configuration page where they indicate the target variable, the predictors, the categorical variables, and the time regular time interval of recording.
<br>
<br>
<b>Detailed Ranking</b>
<br>
By clicking on the but "Run models", 20+ models are ran then ranked from the most accurate to the least based on the root mean squared error (RMSE). For each model, a graph illustrating how well it performed on the data is shown as well as the other evaluation metrics used, including the mean absolute error (MSE)and the mean error (ME), and the next period forecast.
<br>
<br>
<b>Simple Ranking</b>
<br>
On this page, the user can view only the ranking of the models, their evaluation metrics, and the forecast of the following period. The visualization is not displayed.
<br>
<br>
<b>ML Forecasting</b>
<br>
Following that, the 'ML Forecasting' page can be accessed to forecast the next period value using the machine learning algorithms previously run. To do this, all the predictors' fields should be filled with their respective values of the following period. It's not necessary to do that for the other models because they are univariate - the forecasting is only based on the variability of the target variable over time.
<br>
<br>
<b>EDA</b>
<br>
And finally, an exploratory data analysis report created using Pandas-profiling allows the user to gain insight into many aspects of the data uploaded, including data quality, correlation, and interaction between the predictors and the target variable.

<h2>App Structure</h2>
<img src="https://cdn-images-1.medium.com/max/1200/1*d9m5WEAGywkA7s8J0vlWGg.png">

The code consists of 4 main folders including 'uploads', 'static', and 'templates', and 4 other files.
uploads: It is in this folder that the uploaded file will be stocked.

<b>static:</b> 
<br>
This folder contains all the static files of the template such as CSS and JS files.
<br>
<br>
<b>templates: </b>
<br>
All the HTML files representing the application pages are stocked in this folder.
<br>
<br>
<b>ml_models:</b> 
<br>
When the forecasting models are run, the machine learning models are saved in this folder to be used afterward for machine learning forecasting.
<br>
<br>
<b>app.py:</b> 
This file contains code to manage the routing of the app and to launch the webserver.
<br>
<br>
<b>models.py:</b> 
<br>
In this file are implemented the forecasting models.
<br>
<br>
<b>requirements.txt: </b>
<br>
This file contains all Python libraries and their versions installed in the development environment while created the app. It's so that they would be installed on Heroku during the deployment.
