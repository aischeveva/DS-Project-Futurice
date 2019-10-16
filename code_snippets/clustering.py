import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import *

def topic_clustering(year):
    """ Cluster the topics of the year given as a parameter.
        --------------------
        Parameter:
            year: the year of interest
        Return:
            None
    """

    preprocess(year,year)

    #query_docs didn't work(Memory error) so I wrote quite similar code below
    #reports = query_docs(2013, 2014)

    #Create list of reports
    reports=[]
    #Create list of year directory's reports
    companies = os.listdir('cleaned' + os.sep + str(year))
    #The command above inserted some "DS.store"-string in the beginning, so I remove it
    companies.remove(companies[0])
    for company in companies:
        # Open the report
        with open('cleaned/'+str(year)+'/'+company, 'r') as file:
            data = file.read().replace('\n', '')
        # Append report to the list
        reports.append(data)
        
    #tf-idf
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(reports)
    feature_names = vectorizer.get_feature_names()
    dense = vectors.todense()

    #Clustering
    num_clusters = 3
    km = KMeans(n_clusters=num_clusters,init='k-means++', max_iter=100, n_init=1)
    km.fit(dense)

    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    #df = pd.DataFrame(dense.T, columns=companies, index=feature_names)

    #Print the clustering results
    for i in range(len(km.cluster_centers_)):
        print("Cluster %d:" % i),
        for ind in order_centroids[i, :10]:
            print(' %s' % feature_names[ind]),
        print

def document_clustering(year):
    """ Cluster the documents of the year given as a parameter.
        --------------------
        Parameter:
            year: the year of interest
        Return:
            None
    """
    #No working solution yet


if __name__ == '__main__':
    topic_clustering(2013)
