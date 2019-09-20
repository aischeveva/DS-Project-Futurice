# Generate the list of index files archived in EDGAR since start_year (earliest: 1993) until the most recent quarter
import requests
import os
import re
import pandas as pd
from pathvalidate import sanitize_filename

# Please download index files chunk by chunk. For example, please first download index files during 1993–2000, then
# download index files during 2001–2005 by changing the following two lines repeatedly, and so on. If you need index
# files up to the most recent year and quarter, comment out the following three lines, remove the comment sign at
# the starting of the next three lines, and define the start_year that immediately follows the ending year of the
# previous chunk.
def download_index(start_year, current_year):
    current_quarter = 4  # do not change this line

    # start_year = 2016     # only change this line to download the most recent chunk
    # current_year = datetime.date.today().year
    # current_quarter = (datetime.date.today().month - 1) // 3 + 1

    years = list(range(start_year, current_year))
    quarters = ['QTR1', 'QTR2', 'QTR3', 'QTR4']
    for current_year in years:
        print(current_year)
        if str(current_year) not in os.listdir('index'):
            os.mkdir('index' + os.sep + str(current_year))
        for i in range(1, current_quarter + 1):
            url = 'https://www.sec.gov/Archives/edgar/full-index/%d/%s/master.idx' % (current_year, 'QTR%d' % i)
            index = requests.get(url).content.decode("utf-8", "ignore")
            with open('index' + os.sep + str(current_year) + os.sep + 'QTR%d.txt' % i, 'w', encoding='utf-8') as f:
                f.write(index)


def clean_index():
    years = os.listdir('index')
    for year in years:
        print(year)
        files = os.listdir('index' + os.sep + year)
        for file in files:
            with open('index' + os.sep + year + os.sep + file, 'r', encoding='utf-8') as f:
                text = f.read()
            text = re.sub('Description:.*?\n', '', text)
            text = re.sub('Last Data Received:.*?\n', '', text)
            text = re.sub('Comments:.*?\n', '', text)
            text = re.sub('Anonymous FTP:.*?\n', '', text)
            text = re.sub('Cloud HTTP:.*?\n', '', text)
            text = re.sub('-+?\n', '', text)
            text = re.sub('\s{2,}', '', text)
            with open('index' + os.sep + year + os.sep + file, 'w', encoding='utf-8') as f:
                f.write(text)


def get_links(filename):
    df = pd.read_csv(filename, sep='|')
    return df[df['Form Type'] == '10-K']


def download_10_k():
    years = os.listdir('index')
    for year in years[6:]:
        print(year)
        if year not in os.listdir('forms'):
            os.mkdir('forms' + os.sep + year)
        files = os.listdir('index' + os.sep + year)
        for file in files:
            links = get_links('index' + os.sep + year + os.sep + file)
            for i, row in links.iterrows():
                name = sanitize_filename(row['Company Name'])
                if not os.path.exists('forms'+os.sep+year+os.sep+name+'.txt'):
                    url = 'https://www.sec.gov/Archives/%s' % (row['Filename'])
                    text = requests.get(url).content.decode("utf-8", "ignore")
                    with open('forms' + os.sep + year + os.sep + name + '.txt', 'w', encoding='utf-8') as f:
                        f.write(text)




# download_index(2000, 2019)
# clean_index()
download_10_k()