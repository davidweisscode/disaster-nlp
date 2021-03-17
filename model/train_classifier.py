import sys
import re
import nltk
import pickle
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.base import BaseEstimator, TransformerMixin

nltk.download(["stopwords", "punkt", "wordnet", "averaged_perceptron_tagger"])
stop_words = stopwords.words("english")
lemmatizer = WordNetLemmatizer()

def load_data(database_filepath):
    """
    Load message and category data from sqlite database to feature and target dataframe
    
    INPUT:
    database_filepath - relative path to database
    
    OUTPUT:
    X - series with feature column
    Y - dataframe with multi class target columns
    category_names - list of classes
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('disaster', engine)

    X = df["message"]
    Y = df.iloc[:, 4:] # select all rows, but only category columns as target
    
    category_names = list(Y.columns)
    
    return X, Y, category_names

def tokenize(text):
    """
    Split text into normalized and lemmatized alpha-numerical tokens
    
    INPUT:
    text - message string to be tokenized
    
    OUTPUT:
    tokens - list of tokens
    """
    # detect url
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    # normalize case
    text = text.lower()
    # remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    # strip
    text = text.strip()
    # tokenize text
    tokens = word_tokenize(text)
    # lemmatize and remove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return tokens

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    def starting_verb(self, text):
        """
        Estimator to check if a sentence starts with a verb

        INPUT:
        text - message string from which the feature will be extracted

        OUTPUT:
        [0, 1] - 0: False, 1: True
        """
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            try: # a message might be empty
                first_word, first_tag = pos_tags[0]
                if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                    return 1
            except:
                return 0
        return 0

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

def build_model():
    """
    Build machine learning pipeline with feature union and a multioutput Random Forest classifier. Parameter tuning using GridsearchCV.
    
    OUTPUT:
    cv - machine learning model
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([
            
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),
    
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
#     print(pipeline.get_params())

    parameters = {
        'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
#         'features__text_pipeline__vect__max_df': (0.8, 1.0),
#         'features__text_pipeline__vect__max_features': (None, 2500, 5000),
#         'features__text_pipeline__tfidf__use_idf': (True, False),
#         'features__text_pipeline__tfidf__norm': ("l1", "l2"),
#         'clf__estimator__n_estimators': [50, 100],
#         'clf__estimator__min_samples_split': [2, 3, 4],
#         'clf__estimator__criterion': ("gini", "entropy"),
    }

    # create grid search object
    cv = GridSearchCV(pipeline, param_grid=parameters, cv=2, verbose=2)
    
    return cv

def display_results(model, Y_test, Y_pred):
    """
    Print labels, accuracy, and best model parameters
    
    INPUT:
    model - machine learning model
    Y_test - targets of test data
    Y_pred - predictions of the model
    """
    labels = np.unique(Y_pred)
    accuracy = (Y_pred == Y_test).mean()
    print("Labels:", labels)
    print("Accuracy:\n", accuracy)
    print("\nBest Parameters:", model.best_params_)

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Print precision, recall, f1-score for each category
    
    INPUT:
    model - machine learning model
    X_test - test data feature series
    Y_test - targets of test data
    category_names - list of target categories
    """
    Y_pred = model.predict(X_test)
    display_results(model, Y_test, Y_pred)
    print(classification_report(Y_test, Y_pred, target_names=category_names))

def save_model(model, model_filepath):
    """
    Dump the machine learning model to a pickle file
    
    INPUT:
    model - machine learning model
    model_filepath - relative file path of the pickle file
    """
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()