# Implement TF-IDF using scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import os
import gensim
from utils import *

def tf_idf(start_year, end_year, office, sector, companies=None):
    """ Compute tf-idf scores for documents of each year
        and return a dataframe containing 20 highest-score
        words vs years.
        --------------------
        Parameter:
            start_year: starting year of interest
            end_year: ending year of interest
            office:
            sector:
            companies (list of str): list of interested companies

        Return:
            pd.DataFrame of tf_idf score vs years
    """
    # Query desired reports for tf-idf.
    reports = query_docs(start_year, end_year, office,
            sector, True, companies)

    # Create list of years in string format
    years = [str(year) for year in range(start_year, end_year)]

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
