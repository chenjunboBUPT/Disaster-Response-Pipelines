import sys
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import classification_report
import nltk
import re
import numpy as np
import pickle
nltk.download('punkt')
nltk.download('stopwords')

def load_data(database_filepath):
    """ load data from database 
    INPUT:
        database_filepath: a string that show database path
    OUTPUT:
        X:  a array that contain message text(feature  variables)
        Y:  a dataframe that contain the result of different catagories(target variables)
        catagory_names: a list that contain the name of catagories
    """
    # load data from database 
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('DisasterMessage',engine)
    
    # use df to create X,Y,category_names
    X = np.array(df['message'])
    Y = df.drop(labels=['id','message','original','genre'],axis=1)
    category_names = Y.columns.tolist()
    return X,Y,category_names

def tokenize(text):
    """ tokenize the input text
    INPUT: 
        text: a string to be tokenize
    OUTPUT:
        llemmed: a list of token extracted from text
    """
    # Normalization
    text = text[0][0].lower()
    text = re.sub(r'\W',' ',text)
    
    # Tokenization
    words = word_tokenize(text)
    
    # Stop Word Removal
    words = [w for w in words if w not in stopwords.words('english')]
    
    # Lemmatization
    lemmed  = [WordNetLemmatizer().lemmatize(w) for w in words]
    
    return lemmed

def build_model():
    """ build a machine learning model
    INPUT:
        None
    OUTPUT:
        pipeline: a pipeline to classify message
    """
    # create a pipeline
    pipeline = Pipeline([
        ('vect',CountVectorizer(tokenizer=tokenize)),
        ('tfidf',TfidfTransformer()),
        ('clf',MultiOutputClassifier(RandomForestClassifier()))
    ])
    # specify parameters to be searched
    parameters = {
        'clf__estimator__n_estimators': [10,50],
    }
    cv = GridSearchCV(pipeline,parameters)

    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    """ evaluate the performance of model
    INPUT: 
        model: a pipeline trained with train dataset
        X_test: a feature variable used to predict
        Y_test: a target variable used to evaluate the result
        category_names: a list that contain the name of catagories
    OUTPUT:
        None
    """
    Y_pred = model.predict(X_test)
    for cat_name in category_names:
        Y_index = Y_test.columns.get_loc(cat_name)
        print(cat_name)
        print(classification_report(np.array(Y_test[cat_name]),Y_pred[:,Y_index]))
    return

def save_model(model, model_filepath):
    # save model to pkl
    pickle.dump(model,open(model_filepath,'wb'))
    return 

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