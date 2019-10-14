import os
import gensim
import nltk
from pathvalidate import sanitize_filename
import pandas as pd


def rename_to_index(start_year, end_year):
    '''
    Rename the files from companies' names into indices in folders 'forms' and 'cleaned'.
    :param start_year: first year of interest
    :param end_year: last year, not included
    :return: None
    '''
    for year in range(start_year, end_year):
        print(year)
        # Get index files for this year
        files = os.listdir('index' + os.sep + str(year))
        # For every index file
        for file in files:
            # Get the dataframe from index file
            df = pd.read_csv('index' + os.sep + str(year) + os.sep + file, sep='|')
            df = df[df['Form Type'] == '10-K']
            # For every link
            for i, row in df.iterrows():
                # Create filename
                name = sanitize_filename(row['Company Name'])
                old_name = 'forms'+os.sep+str(year)+os.sep+name+'.txt'
                new_name = 'forms'+os.sep+str(year)+os.sep+str(row['CIK'])+'.txt'
                # Check that the file exists
                if os.path.exists(old_name) and not os.path.exists(new_name):
                    # Rename
                    os.rename(old_name,
                              new_name)
                # Rename processed files as well
                old_name = 'cleaned' + os.sep + str(year) + os.sep + name + '.txt'
                new_name = 'cleaned' + os.sep + str(year) + os.sep + str(row['CIK']) + '.txt'
                if os.path.exists(old_name) and not os.path.exists(new_name):
                    os.rename(old_name,
                              new_name)


def preprocess(start_year, end_year, companies=['']):
    """ Preprocess input documents and save the results to
        folder name 'cleaned'.
        --------------------
        Parameter:
            start_year: starting year of interest
            end_year: ending year of interest
            companies (list of str): list of interested companies

        Return:
            None
    """
    for year in range(start_year, end_year):
        if companies[0] == '':
            companies = os.listdir(f'forms/{str(year)}')
        for company in companies:
            filename = f'forms/{str(year)}/{company}'
            #If company file not found, continue
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    text = f.read()
            except Exception: 
                continue
            # Remove stopwords
            text = gensim.parsing.remove_stopwords(text)
            # Bring to lower, strip punctuation, split by spaces
            text = gensim.utils.simple_preprocess(text, min_len=4)
            # Lemmatize to root.
            stemmer = nltk.stem.PorterStemmer()
            text = [stemmer.stem(w) for w in text]
            if str(year) not in os.listdir('cleaned'):
                os.mkdir('cleaned' + os.sep + str(year))
            cleaned = f'cleaned/{str(year)}/{company}'
            with open(cleaned, 'w', encoding='utf-8') as f:
                f.write(' '.join(text))
            # Print out the year to check missing years
        print(year)

def query_docs(start_year, end_year, companies=['']):
    """ Query documents from folder 'cleaned'.
        --------------------
        Parameter:
            start_year: starting year of interest
            end_year: ending year of interest
            companies (list of str): list of interested companies

        Return:
            list of textual documents
    """
    # Create an empty list for the docs
    docs = []
    # Create list of years in string format
    years = [str(year) for year in range(start_year, end_year)]
    # Open the docs in loop
    for year in years:
        text_dump = ''
        if companies[0] == '':
            companies = os.listdir('cleaned' + os.sep + str(year))
        # Open the report
        for company in companies:
            with open(f'cleaned/{year}/{company}', 'r', encoding='utf-8') as f:
                # Read the report
                text_dump += f.read()
            # Append report to the list
        docs.append(text_dump)
    return docs






