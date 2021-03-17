import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load message and category data from .csv files to dataframe
    
    INPUT:
    messages_filepath - relative path to messages.csv
    categories_filepath - relative path to categories.csv
    
    OUTPUT:
    df - dataframe with feature and target columns
    """
    messages = pd.read_csv(messages_filepath)
    print("Messages loaded", messages.shape)
    categories = pd.read_csv(categories_filepath)
    print("Categories loaded", categories.shape)
    df = messages.merge(categories)
    print("Messages and categories merged", df.shape)
    
    categories = categories["categories"].str.split(";", expand=True)
    row = categories.loc[0]
    category_colnames = row.apply(lambda colname: colname[:-2]) # remove dash and number
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype("int32")
    
    df = df.drop(columns=["categories"])
    df = pd.concat([df, categories], axis=1, join='inner', sort=False)
    return df

def clean_data(df):
    """
    Clean data by dropping duplicates and non-binary category values
    
    INPUT:
    df - dataframe with message and category columns
    
    OUTPUT:
    df - dataframe with cleaned feature and target data
    """
    print("Duplicates before", df.duplicated().sum())
    df = df.drop_duplicates()
    print("Duplicates after", df.duplicated().sum())
    df = df[df["related"] != 2] # check for non binary values in categories
    print("Data cleaned", df.shape)
    return df

def save_data(df, database_filename):
    """
    Save the dataframe to a sqlite database
    
    INPUT:
    df - dataframe to be saved in database
    database_filename - relative path to database file
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('disaster', engine, index=False, if_exists="replace")

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