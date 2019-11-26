from flask import Flask
from flask import render_template
from flask import request
import re
import pandas as pd
import json
import os

# create Flask app
app = Flask(__name__, static_url_path='/static')

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
    
    data = pd.read_csv('source/Office of Energy & Transportation_Mining.csv', index_col=0)
    topics = []
    for column in data.columns:
        words = column.split(' | ')
        topics.append([{'name': word, 'weight': len(words) + 5 - i} for i, word in enumerate(words)])
    data.columns = range(1, len(data.columns)+1)
    # render template with passing data as parameter
    return render_template('main_page.html', k=data.to_csv(), words=json.dumps(topics), industries=dic)


if __name__ == '__main__':
    # run the app
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.debug = True
    app.run()