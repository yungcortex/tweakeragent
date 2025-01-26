from flask import Flask, render_template, request, jsonify
from characters.test import app as character_app
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Register the character blueprint
app.register_blueprint(character_app)

if __name__ == '__main__':
    app.run(debug=True) 
