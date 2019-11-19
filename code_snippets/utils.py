import os
import gensim
import nltk
import re
import requests
import shutil
import pandas as pd
import lxml.html as lh
from pathvalidate import sanitize_filename
from bs4 import BeautifulSoup
from functools import reduce


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

def preprocess(start_year, end_year, companies=None):
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

    no_default_companies = False
    if not companies:
        no_default_companies = True

    for year in range(start_year, end_year):
        if no_default_companies:
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

def query_intersection(start_year, end_year, office, sector,
        combine, intersection=True):
    """ Query documents from folder 'industries'.
        --------------------
        Parameter:
            start_year: starting year of interest
            end_year: ending year of interest
            office: Office's name
            sector: sector's name
            combine: if True, then documents within 1 year are
                     combined into 1 long string; if False, each
                     year will be a list of documents
            companies (list of str): list of interested companies

        Return:
            list of documents
    """
    docs = []
    # Find intersection of companies over years:
    for year in range(start_year, end_year+1):
        path = "industries" + os.sep + str(year) + os.sep  \
             + "Office of " + office + os.sep + sector
        docs.append(os.listdir(path))
    companies = reduce(lambda x, y: set(x).intersection(set(y)), docs)

    # Create an empty list for the docs:
    docs = []
    # Open the docs in loop:
    for year in range(start_year, end_year+1):
        dump = []
        path = "industries" + os.sep + str(year) + os.sep  \
             + "Office of " + office + os.sep + sector
        # Open the report:
        for company in companies:
            try:
                with open(path + os.sep + company, 'r') as f:
                    # Read the report:
                    dump.append(f.read())
            except Exception:
                continue
        # Append report to the list:
        if combine:
            docs.append(' '.join(dump))
        else:
            docs.append(dump)
    return docs

def query_docs(start_year, end_year, office, sector,
        combine, companies=['']):
    """ Query documents from folder 'industries'.
        --------------------
        Parameter:
            start_year: starting year of interest
            end_year: ending year of interest
            office: Office's name
            sector: sector's name
            combine: if True, then documents within 1 year are
                     combined into 1 long string; if False, each
                     year will be a list of documents
            companies (list of str): list of interested companies

        Return:
            list of documents
    """
    # Create an empty list for the docs:
    docs = []
    # Create list of years in string format:
    years = [str(year) for year in range(start_year, end_year)]
    # Open the docs in loop:
    for year in years:
        dump = []
        if companies[0] == '':
            path = "industries" + os.sep + str(year) + os.sep  \
                 + "Office of " + office + os.sep + sector
            companies = os.listdir(path)
        # Open the report:
        for company in companies:
            try:
                with open(path + os.sep + company, 'r') as f:
                    # Read the report:
                    dump.append(f.read())
            except Exception:
                continue
        # Append report to the list:
        if combine:
            docs.append(' '.join(dump))
        else:
            docs.append(dump)
    return docs

def CIK_2_SIC_series():
    """ Get CIK <-> SIC convertion.
        CIK is the company's ID in Edgar database.
        SIC is the company's business sector code defined in Edgar.
        --------------------
        Parameter:
            None

        Return:
            pd.Series
    """
    # Read CSV file.
    df = pd.read_csv('Industry.csv', sep=';',
            usecols=['SEC ID', 'Latest SIC Industry Code'])
    # Rename 2 columns.
    df = df.rename(columns={'SEC ID': 'CIK', 'Latest SIC Industry Code': 'SIC'})
    # Filter empty cell, convert to type string and squeeze to series.
    series = df.dropna().astype(int).astype(str).set_index('CIK').squeeze()
    return series

def SIC_2_Industry_df():
    """ Get SIC <-> Office & Industry convertion.
        Within an 'Office' there are several 'Industry'.
        --------------------
        Parameter:
            None

        Return:
            pd.DataFrame
    """
    # Auxiliary function for merging industries
    # according to merge.csv (merge.csv is built manually).
    merger = pd.read_csv('merge.csv')
    def merge(data):
        sic = int(data[0])
        for (_, c) in merger.T.iteritems():
            if c['Start'] <= sic <= c['End']:
                return [data[0], data[1], c['Name']]
        return data

    # Get url of the SIC list.
    url='https://www.sec.gov/info/edgar/siccodes.htm'
    # Create a handle, page, to handle the contents of the website.
    page = requests.get(url)
    # Store the contents of the website under doc.
    doc = lh.fromstring(page.content)
    # Parse data that are stored between <tr>..</tr> of HTML.
    rows = doc.xpath('//tr')
    # Create empty dictionary.
    dic = {t.text_content(): [] for t in rows[0]}
    # Read data into dictionary.
    for row in rows[1:]:
        data = [t.text_content() for t in row.iterchildren()]
        data = merge(data)
        dic['SIC Code'].append(data[0])
        dic['Office'].append(data[1])
        dic['Industry Title'].append(data[2])
    # Return SIC dataframe.
    return pd.DataFrame(dic).rename(columns={'SIC Code':
        'SIC', 'Industry Title': 'Industry'}).set_index('SIC').T

def classify_industry(start_year, end_year):
    """ Classify company by industry.
        --------------------
        Parameter:
            start_year: starting year of interest
            end_year: ending year of interest

        Return:
            None
    """
    cik_2_sic = CIK_2_SIC_series()
    sic_2_ind = SIC_2_Industry_df()

    for year in range(start_year, end_year):
        print(year)
        # Get index files for this year
        companies = os.listdir('cleaned' + os.sep + str(year))
        for company in companies:
            try:
                # Get CIK, Office and Industry of the company.
                cik = company[:-4] if '.txt' in company else company
                office = sic_2_ind[cik_2_sic[cik]]['Office']
                industry = sic_2_ind[cik_2_sic[cik]]['Industry']
                # Get company file path and target industry directory.
                old = 'cleaned' + os.sep + str(year) + os.sep + company
                new = 'industries' + os.sep + str(year) + os.sep + office + os.sep + industry
                # Move file.
                if not os.path.exists(new):
                    os.makedirs(new)
                shutil.move(old, new)
            except Exception:
                continue

def change_to_txt(start, end):
    import os
    for year in range(start, end):
        path_year = "industries/" + str(year) + "/"
        for office in os.listdir(path_year):
            if office != '.DS_Store':
                path_office = path_year + office + "/"
                for sector in os.listdir(path_office):
                    if sector != '.DS_Store':
                        path_sector = path_office + sector + "/"
                        for company in os.listdir(path_sector):
                            if company != '.DS_Store':
                                old = path_sector + company
                                new = old + ".txt"
                                os.rename(old, new)

