from flask import Flask
from flask import render_template
from flask import request
import re
import pandas as pd
import json
import os

# create Flask app
app = Flask(__name__, static_url_path='/static')

# Prepare names of 'office' and sector:
def load_data():
    files = [name.split('_') for name in os.listdir('source') if name.startswith('Office')]
    dic = {}
    for p in files:
        office= p[0]
        sector = p[1]
        if office in dic:
            dic[office].append(sector)
        else:
            dic[office] = [sector]
    return dic

@app.route('/')
def start_page():
    dic = load_data()
    return render_template('main_page.html', industries=dic)

# call when main page is loaded
@app.route('/<data>', methods=['GET', 'POST'])
def dashboard(data):
    # Load data from 'source':
    dic = load_data()
    data = pd.read_csv('source/' + data, index_col=0)
    topics = []
    for column in data.columns:
        words = column.split(' | ')
        topics.append([{'name': word, 'weight': len(words) + 5 - i} for i, word in enumerate(words)])
    data.columns = range(1, len(data.columns)+1)
    # Render template with passing data as parameter:
    return render_template('dashboard.html', k=data.to_csv(),
            words=json.dumps(topics), industries=dic)

if __name__ == '__main__':
    # run the app
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.debug = True
    app.run()
