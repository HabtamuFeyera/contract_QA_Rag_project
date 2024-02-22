
from flask import Flask, request, render_template
from contract_assistant import get_response_to_query

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    query = request.form['query']
    response = get_response_to_query(query)
    return response

if __name__ == '__main__':
    app.run(debug=True)
