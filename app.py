from flask import Flask
from characters.test import bp as character_blueprint
import os

app = Flask(__name__)
app.register_blueprint(character_blueprint)

# Export the app variable
__all__ = ['app']

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port) 
