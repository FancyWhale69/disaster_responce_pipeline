# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from nltk.corpus import stopwords
import re
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle
nltk.download('punkt')
nltk.download('stopwords')
import sys


def load_data(database_filepath):
    # load data from database
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('table', engine)

    X= df['message']
    y=df.drop(['message', 'genre', 'original', 'id'], axis=1)

    return X, y, y.columns


def tokenize(text):
    """
    a method to preprocess text. entered text will be cleaned of punctation, stopwords, and edge whitespaces.
    text will be tokinazed, and each token will be stemmed
    
    Input- string containing the text 
    ------------
    output- list of tokens
    """
    #normilize and remove puncutation 
    text= re.sub("[^a-zA-Z0-9]"," ", text).lower()
    
    #tokinaze text
    tokens= word_tokenize(text)
    
    #remove stop words and clean edge white spaces
    tokens= [w.strip() for w in tokens if w not in stopwords.words('english')]
    
    #initilaze stemmer
    stem= PorterStemmer()
    
    #stem words
    stemmed_tokens=[stem.stem(w) for w in tokens]
    
    return stemmed_tokens


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ])

    parameters = {
    'clf__estimator__n_estimators':[100, 150, 200]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters,  verbose=2)

    return cv



def evaluate_model(model, X_test, Y_test, category_names):
    y_pred= model.predict(X_test)
    print(classification_report(Y_test, y_pred, target_names=category_names))


def save_model(model, model_filepath):
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