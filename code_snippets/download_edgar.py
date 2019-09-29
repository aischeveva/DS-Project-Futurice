# Generate the list of index files archived in EDGAR since start_year (earliest: 1993) until the most recent quarter
import requests
import os
import re
import pandas as pd
from pathvalidate import sanitize_filename
from bs4 import BeautifulSoup

# Please download index files chunk by chunk. For example, please first download index files during 1993–2000, then
# download index files during 2001–2005 by changing the following two lines repeatedly, and so on. If you need index
# files up to the most recent year and quarter, comment out the following three lines, remove the comment sign at
# the starting of the next three lines, and define the start_year that immediately follows the ending year of the
# previous chunk.
def download_index(start_year, current_year):
    '''
    Download index files from Edgar
    :param start_year:
    :param current_year:
    :return:
    '''
    current_quarter = 4  # do not change this line

    # start_year = 2016     # only change this line to download the most recent chunk
    # current_year = datetime.date.today().year
    # current_quarter = (datetime.date.today().month - 1) // 3 + 1

    years = list(range(start_year, current_year))
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
    '''
    Clean index files from the meta-data
    :return:
    '''
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


def get_links(filename, company_name=''):
    # Read the index file into dataframe
    df = pd.read_csv(filename, sep='|')
    # If company name specified, filter company
    if company_name != '':
        df = df[df["Company Name"] == company_name]
    # Return annual reports
    print(df)
    return df[df['Form Type'] == '10-K']

def clean_text(text):
    """ Clean the raw text file. """
    # From the text file extract only DOCUMENT of
    # type 10-K (DOCUMENTs of other types don't contain
    # useful text.
    soup = BeautifulSoup('\n'.join(re.findall('<TYPE>10-K.*?</DOCUMENT>', text,
        re.S)))

    # Get pure text from html DOCUMENT and lowercase them.
    txt = soup.get_text().lower()

    # Remove all the non-word character to get only words.
    txt = re.sub('[^A-Za-z-\']*[^A-Za-z-\']', ' ', txt)

    return txt


def download_10_k(start_year, end_year, company_name=''):
    '''
    Downloads files form Edgar
    :param start_year: int, year from which dowloading starts
    :param end_year: int, where it ends
    :param company_name: str, company name EXACTLY as in index files
    :return:
    '''
    # For every year in range
    for year in range(start_year, end_year + 1):
        print(year)
        # If the folder for this year doesn't exist, create a folder
        if str(year) not in os.listdir('forms'):
            os.mkdir('forms' + os.sep + str(year))
        # Get index files for this year
        files = os.listdir('index' + os.sep + str(year))
        # For every index file
        for file in files:
            # Get all the links from the index file
            links = get_links('index' + os.sep + str(year) + os.sep + file, company_name)
            # For every link
            for i, row in links.iterrows():
                # Create filename
                name = sanitize_filename(row['Company Name'])
                # If the file doesn't exist already, download it from database
                if not os.path.exists('forms'+os.sep+str(year)+os.sep+name+'.txt'):
                    # Create url
                    url = 'https://www.sec.gov/Archives/%s' % (row['Filename'])
                    # Download text
                    text = requests.get(url).content.decode("utf-8", "ignore")
                    # Clean text
                    cleaned = clean_text(text)
                    # Write it in the file
                    with open('forms' + os.sep + str(year) + os.sep + name + '.txt', 'w', encoding='utf-8') as f:
                        f.write(cleaned)



if __name__ == '__main__':
    # download_index(2000, 2019)
    # clean_index()
    download_10_k(2010, 2018, 'APPLE INC')
    #print(os.listdir('index'))
