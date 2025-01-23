from flask import Flask, request, jsonify, render_template
import os
import json
import time
import numpy as np
import pandas as pd
import requests
import logging
from datetime import datetime, timedelta
import random
import character_config
from character_config import get_character_response, get_market_sentiment, process_chat_input

# Add this near the top of your file with other constants
COINGECKO_IDS = {
    'btc': 'bitcoin',
    'eth': 'ethereum',
    'sol': 'solana',
    'bnb': 'binancecoin',
    'xrp': 'ripple',
    'ada': 'cardano',
    'doge': 'dogecoin',
    'dot': 'polkadot',
    'matic': 'matic-network',
    'link': 'chainlink',
    # Add more as needed
}

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the absolute path to the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Initialize Flask with the correct paths
app = Flask(__name__,
           static_url_path='/static',
           static_folder=os.path.join(project_root, 'static'),
           template_folder=os.path.join(project_root, 'templates'))

# Load character configuration
character_path = os.path.join(os.path.dirname(__file__), 'tweaker.character.json')
with open(character_path, 'r') as f:
    CHARACTER = json.load(f)

def get_character_response(message_type='default'):
    """Get a random message example or generate a response based on character style"""
    if message_type == 'greeting':
        return "initializing cynicism protocols... what's your damage? try asking about any coin's mental state..."

    # Get random message example if available
    if CHARACTER['messageExamples']:
        example = random.choice(CHARACTER['messageExamples'])
        return example[1]['content']['text']

    return "yeah, that's about as stable as my mental state..."

def format_analysis(analysis_data):
    """Format market analysis according to character style"""
    try:
        # Extract basic data
        price = analysis_data.get('price', 0)
        change = analysis_data.get('change_24h', 0)
        volume = analysis_data.get('volume_24h', 0)

        # Create unhinged analysis using character vocabulary
        vocab = CHARACTER['vocabulary']

        if change > 0:
            trend = vocab.get('uptrend', 'dopamine spike')
            state = "manic episode"
        else:
            trend = vocab.get('downtrend', 'depression spiral')
            state = "existential crisis"

        response_parts = [
            f"having another {state}... {abs(change):.2f}% {trend}",
            f"volume showing {vocab.get('volume', 'emotional damage')} at ${volume:,.2f}",
            f"price at ${price:.8f} like my deteriorating stability",
            f"market sentiment more unstable than my relationships"
        ]

        return " ... ".join(response_parts).lower()

    except Exception as e:
        logger.error(f"Error formatting analysis: {str(e)}")
        return "having a breakdown... give me a minute..."

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.json
        message = data.get('message', '').strip().lower()

        # Check if it's a crypto query
        if any(keyword in message for keyword in ['price', 'how is', 'check', 'analyze']):
            # Extract token from message
            tokens = message.split()
            for token in tokens:
                if token.upper() in ['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'DOGE']:
                    # Get market data
                    symbol = token.upper() + "USDT"
                    price_url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
                    ticker_url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"

                    price_data = requests.get(price_url).json()
                    ticker_data = requests.get(ticker_url).json()

                    analysis_data = {
                        'price': float(price_data['price']),
                        'change_24h': float(ticker_data['priceChangePercent']),
                        'volume_24h': float(ticker_data['volume']) * float(price_data['price'])
                    }

                    response = format_analysis(analysis_data)
                    return jsonify({"response": response})

            return jsonify({"response": "can't find that coin... like my lost hopes and dreams..."})

        # For non-crypto queries, get a character response
        return jsonify({"response": get_character_response()})

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({"response": "having a mental breakdown... try again later..."})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
