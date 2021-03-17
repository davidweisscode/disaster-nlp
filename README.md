# disaster-nlp

Short text message classification

## Summary

In this project we use labeled text data provided by Figure Eight for the Udacity Data Science Nano Degree to train a machine learning model and perform a multioutput classification in a web app. The model is build by using a sklearn pipeline with the custom estimator object StartingVerbExtractor. Gridsearch is then used to tune the model parameters.

A user can enter text inside a Flask web app and get his message classified into multiple categories.

## Getting started

1. Run ETL pipeline to load, clean, and store the data in a sqlite database
	`python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
2. Run ML pipeline to train the classifier and save the model in a pickle file
	`python model/train_classifier.py data/DisasterResponse.db model/classifier.pkl`
3. Run the following command in the app's directory to run the web app
	`python run.py`
4. Visit `http://0.0.0.0:3001/` to access the web app

## Repository

* `data/`: Holds the final ETL pipeline `process_data.py` and the development notebook. Also holds the database and the .csv files
* `model/`: Holds the final ML pipeline `train_classifier.py` and the development notebook. The pickled model is also stored here.
* `app/`: Flask web app `run.py` with templates
