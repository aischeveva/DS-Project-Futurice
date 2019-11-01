
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

    #preprocess(year,year)

    #query_docs didn't work(Memory error) so I wrote quite similar code below
    #reports = query_docs(2013, 2014)

    #Create list of reports
    reports=[]
    #Create list of year directory's reports
    companies = os.listdir('cleaned' + os.sep + str(year))
    #The command above inserted some "DS.store"-string in the beginning, so I remove it
    companies.remove(companies[0])
    for i in range(8148):
        # Open the report
        #report=[]
        #report.append(i)
        with open('cleaned/'+str(year)+'/'+companies[i], 'r') as file:
            data = file.read().replace('\n', '')
            #report.append
        # Append report to the list
        reports.append(data)
        
    #tf-idf
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(reports)
    feature_names = vectorizer.get_feature_names()
    dense = vectors.todense()

    #Clustering
    num_clusters = 5
    km = KMeans(n_clusters=num_clusters,init='k-means++', max_iter=100, n_init=1)
    km.fit(dense)

    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    #df = pd.DataFrame(dense.T, columns=companies, index=feature_names)

    #Print the clustering results
    for i in range(len(km.cluster_centers_)):
        print("Cluster %d:" % (i+1)),
        for ind in order_centroids[i, :10]:
            print(' %s' % feature_names[ind]),
        print

    clusters = km.labels_.tolist()

 

def document_clustering(year):
    """ Cluster the documents of the year given as a parameter.
        --------------------
        Parameter:
            year: the year of interest
        Return:
            None
    """
    #preprocess(year,year)

    #query_docs didn't work(Memory error) so I wrote quite similar code below
    #reports = query_docs(2013, 2014)

    #Create list of reports
    reports=[]
    #Create list of year directory's reports
    companies = os.listdir('cleaned' + os.sep + str(year))
    #The command above inserted some "DS.store"-string in the beginning, so I remove it
    companies.remove(companies[0])
    #Create list of selected companies
    company = []
    #
    amount_of_files=100
    for i in range(amount_of_files):
        # Open the report

        with open('cleaned/'+str(year)+'/'+companies[i], 'r') as file:
            data = file.read().replace('\n', '')
        # Append report to the list
        reports.append(data)
        #Append selected company to another list
        company.append(companies[i])


    #tf-idf
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(reports)
    transformer = TfidfTransformer(smooth_idf=False)
    tfidf = transformer.fit_transform(X)

    #K-means clustering
    num_clusters = 5
    km = KMeans(n_clusters=num_clusters,init='k-means++', max_iter=100, n_init=1)
    km.fit(tfidf)
    clusters = km.labels_.tolist()

    idea={'Filename':company, 'Cluster':clusters} #Creating dict having report's filename with the corresponding cluster number.
    frame=pd.DataFrame(idea,index=[clusters], columns=['Filename','Cluster']) # Converting it into a dataframe.

    #Printing the results
    for i in range(num_clusters):
        print("Cluster"+ str(i+1)+":")
        cluster_i=frame.loc[[i]]
        fra=cluster_i['Filename'].tolist()
        for i in fra:
            print(i)



if __name__ == '__main__':
    #topic_clustering(2013)
    document_clustering(2013)
