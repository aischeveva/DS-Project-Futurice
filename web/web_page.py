from flask import Flask
from flask import render_template


# create Flask app
app = Flask(__name__)

# call when main page is loaded
@app.route('/')
def start_page():
    # open csv file
    with open('source/tf_idf_ordered.csv') as f:
        data = f.read()
    # render template with passing data as parameter
    return render_template('main_page.html', k=data)

if __name__ == '__main__':
    # run the app
    app.debug = True
    app.run()