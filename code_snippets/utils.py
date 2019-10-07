import os
import gensim
import nltk

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
            with open(filename, 'r', encoding='utf-8') as f:
                text = f.read()
            # Remove stopwords
            text = gensim.parsing.remove_stopwords(text)
            # Bring to lower, strip punctuation, split by spaces
            text = gensim.utils.simple_preprocess(text, min_len=4)
            # Lemmatize to root.
            lemmatizer = nltk.stem.WordNetLemmatizer()
            text = [lemmatizer.lemmatize(w) for w in text]
            print(text)
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






