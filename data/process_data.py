import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """ load data csv file and merge it
    INPUT:
        messages_filepath: a string that show message path
        categories_filepath: a string that show categories path
    OUTPUT:
        df: a dataframe that merge message and categories with id
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages,categories,on='id')
    return df
def clean_data(df):
    """ split categories into several cols and drop duplicate message
    INPUT:
        df: a dataframe that is going to clean 
    OUTPUT:
        df: a dataframe with transformed categories and no duplicate message
    """
    # Create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=';',expand=True)
    
    # Use first row to rename the columns of `categories`
    row = categories.loc[0,:]
    category_colnames = row.apply(lambda x:x[:-2])
    categories.columns = category_colnames
    # Convert category values to just numbers 0 or 1
    for column in categories:
        categories[column] = categories[column].str.slice(-1,)
        categories[column] = categories[column].astype('int64')
    df = pd.concat([df.drop('categories',axis=1),categories],axis=1)
    df = df.drop_duplicates(subset=['message'])
    return df

def save_data(df, database_filename):
    """ save data to database
    INPUT:
        df: a dataframe that is alread cleaned
        database_filename: a string that show the database's name 
    OUTPUT:
        None
    """
    # Create connection with database 
    engine = create_engine('sqlite:///'+database_filename)
    
    # Save data to database
    df.to_sql('DisasterMessage', engine, index=False)

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()