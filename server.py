from flask import Flask, request, jsonify
import subprocess
import os

app = Flask(__name__)

@app.route('/send_query', methods=['GET', 'POST'])
def send_query():
    data = request.get_json()
    user_input = data.get('query')
    


if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')
