from flask import Flask
from flask import render_template
from flask import request
import re
import pandas as pd
import json
import os

# create Flask app
app = Flask(__name__, static_url_path='/static')
# open csv file
DATA = pd.read_csv('source/Office_of_Energy_& _Transportation_Mining.csv', index_col=0)
with open('example.txt', 'r') as infile:
    TOPICS = json.load(infile)

# call when main page is loaded
@app.route('/', methods=['GET', 'POST'])
def start_page():
    

    files = [name.split('_') for name in os.listdir('source') if name.startswith('Office')]
    dic = {}
    for p in files:
        office= p[0][10:]
        sector = p[1][:-4]
        if office in dic:
            dic[office].append(sector)
        else:
            dic[office] = [sector]
    
    data = DATA
    if request.form:
        print(request.form['keyword'])
        word = request.form['keyword']
        if word in DATA.columns:
            data = DATA[word]
            data = data.to_frame()
    data.columns = range(1, len(data.columns)+1)
    # render template with passing data as parameter
    return render_template('main_page.html', k=data.to_csv(), words=json.dumps(TOPICS), industries = dic)


if __name__ == '__main__':
    # run the app
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.debug = True
    app.run()