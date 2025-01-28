from flask import Flask, render_template
from characters.test import bp as character_bp
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.register_blueprint(character_bp)

@app.route('/')
def home():
    return render_template('index.html')

# Export the app variable
__all__ = ['app']

if __name__ == '__main__':
    app.run(debug=True) 
