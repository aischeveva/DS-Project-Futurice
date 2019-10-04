# Implement TF-IDF using scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import os
import gensim

# Pre-process the data
# load data
def preprocess(start_year, end_year, companies=['']):
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
            text = gensim.utils.simple_preprocess(text, min_len=1)
            print(text)
            # # filter out stop words
            # stop_words = set(stopwords.words('english'))
            if str(year) not in os.listdir('cleaned'):
                os.mkdir('cleaned' + os.sep + str(year))
            cleaned = f'cleaned/{str(year)}/{company}'
            with open(cleaned, 'w', encoding='utf-8') as f:
                f.write(' '.join(text))
            # Print out the year to check missing years
            print(year)

def tf_idf(start_year, end_year, companies=['']):
    # Create an empty list for the reports
    reports = []
    # Create list of years in string format
    years = [str(year) for year in range(start_year, end_year)]
    # Open the reports in loop
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
        reports.append(text_dump)

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(reports)
    feature_names = vectorizer.get_feature_names()
    dense = vectors.todense()

    # Create dataframe with results
    df = pd.DataFrame(dense.T, columns=years, index=feature_names)
    # Return the top 20 words that have highest weight
    return df

# The words which have 0.0 weight is either mentioned in all reports or that word does not appear in the year
if __name__ == '__main__':
    result = tf_idf(2010, 2011)
    print(result.nlargest(20, '2010'))
    #preprocess(2010, 2011)
