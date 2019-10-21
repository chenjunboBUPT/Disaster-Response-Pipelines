

# Quick Start

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
2. Run the following command in the app's directory to run your web app.
    `python run.py`
3. Go to http://localhost:3001/
4. Enter a message like, *we are more than 50 people sleeping on the street please help us find tent,food.*



<img src="https://img2018.cnblogs.com/blog/1821356/201910/1821356-20191021205555739-243361377.png" alt="screenshots" style="zoom:33%;" />

# Libraries

| Libraries  | Usage                                   |
| ---------- | --------------------------------------- |
| pandas     | Deal with tabular data                  |
| nltk       | Deal with natural language              |
| sklearn    | Use machine learning model              |
| plotly     | Plot a figure to show result            |
| sqlalchemy | Interact with the database              |
| re         | Use regular expression to deal with txt |

# Motivation

We can find a data set containing real messages that were sent during disaster events.

I create a machine learning pipeline to categorize these events so that we can send the messages to an appropriate disaster relief agency.

# What's included



```txt
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py # ETL pipeline to preprocess data

- models
|- train_classifier.py # machine learning pipeline to build a model with data

- README.md
```

# Summary

For the first part of my project,i build a ETL pipeline to read the dataset, clean the data, and then store it in a SQLite database. 

For the machine learning portion, i will split the data into a training set and a test set. Then, i create a machine learning pipeline that uses NLTK, as well as scikit-learn's Pipeline and GridSearchCV to output a final model that uses the `message` column to predict classifications for 36 categories (multi-output classification). Finally, you will export your model to a pickle file.

In the last step, i display my results in a Flask web app. 

# Feature work

* Deploy the web app to a cloud service provider.
* Improve the efficiency of the code in the ETL and ML pipeline.
* This dataset is imbalanced (ie some labels like water have few examples). Discuss how this imbalance, how that affects training the model, and my thoughts about emphasizing precision or recall for the various categories.