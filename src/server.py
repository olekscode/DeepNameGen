from flask import Flask
from flask import request

from namegen import NameGenerator

app = Flask(__name__)

namegen = NameGenerator()


@app.route('/post', methods=['POST'])
def get_name():
    sourceCode = request.form['source']
    print(sourceCode)
    name, attention = namegen.get_name_and_attention_for(sourceCode)
    return str({'name': name, 'attention': attention})
