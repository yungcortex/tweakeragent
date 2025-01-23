from flask import Flask, request, jsonify, render_template
import os
import json
import time
import requests
import logging
import random
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the absolute path to the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Initialize Flask
app = Flask(__name__,
           static_url_path='/static',
           static_folder=os.path.join(project_root, 'static'),
           template_folder=os.path.join(project_root, 'templates'))

# Load character configuration
character_path = os.path.join(os.path.dirname(__file__), 'tweaker.character.json')
with open(character_path, 'r') as f:
    CHARACTER = json.load(f)

def get_token_data(token_id):
    """Get token data from multiple sources"""
    try:
        # Try Binance first for major coins
        if token_id.upper() in ['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'DOGE', 'ADA', 'DOT', 'MATIC', 'LINK']:
            symbol = token_id.upper() + "USDT"
            price_resp = requests.get(f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}")
            ticker_resp = requests.get(f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}")

            if price_resp.status_code == 200 and ticker_resp.status_code == 200:
                price_data = price_resp.json()
                ticker_data = ticker_resp.json()
                return {
                    'price': float(price_data['price']),
                    'change_24h': float(ticker_data['priceChangePercent']),
                    'volume_24h': float(ticker_data['volume']) * float(price_data['price']),
                    'source': 'binance'
                }

        # Try DexScreener for any token
        dex_resp = requests.get(f"https://api.dexscreener.com/latest/dex/tokens/{token_id}")
        if dex_resp.status_code == 200:
            data = dex_resp.json()
            if data.get('pairs') and len(data['pairs']) > 0:
                pair = data['pairs'][0]  # Get most liquid pair
                return {
                    'price': float(pair['priceUsd']),
                    'change_24h': float(pair.get('priceChange', {}).get('h24', 0)),
                    'volume_24h': float(pair.get('volume', {}).get('h24', 0)),
                    'source': 'dexscreener',
                    'extra': {
                        'liquidity': pair.get('liquidity', {}).get('usd', 0),
                        'dex': pair.get('dexId', 'unknown')
                    }
                }

    except Exception as e:
        logger.error(f"Error fetching token data: {str(e)}")
    return None

def format_analysis(analysis_data):
    """Format market analysis according to character style"""
    try:
        # Extract basic data
        price = analysis_data.get('price', 0)
        change = analysis_data.get('change_24h', 0)
        volume = analysis_data.get('volume_24h', 0)
        source = analysis_data.get('source', 'unknown')

        # Get random phrases for variety
        manic_states = [
            "manic episode", "psychological breakdown", "existential crisis",
            "therapy session", "mental relapse", "dissociative state",
            "panic attack", "dopamine rush", "trading ptsd"
        ]

        addiction_comparisons = [
            "caffeine addiction", "doomscrolling habit", "trading addiction",
            "chart obsession", "hopium dependency", "fomo syndrome",
            "leverage addiction"
        ]

        relationship_metaphors = [
            "my relationships", "my commitment issues", "my trust issues",
            "my emotional stability", "my social life", "my portfolio",
            "my trading history"
        ]

        # Create dynamic response
        state = random.choice(manic_states)
        addiction = random.choice(addiction_comparisons)
        relationship = random.choice(relationship_metaphors)

        # Format with random variations
        response_parts = [
            f"having another {state}... {'up' if change > 0 else 'down'} {abs(change):.2f}% like my serotonin levels",
            f"volume at ${volume:,.2f} - more unstable than my {addiction}",
            f"price ${price:.8f} giving me flashbacks",
            f"market sentiment more volatile than {relationship}",
            f"data from {source} showing signs of {random.choice(manic_states)}"
        ]

        # Randomize order and select 3-4 parts
        random.shuffle(response_parts)
        selected_parts = response_parts[:random.randint(3, 4)]

        return " ... ".join(selected_parts).lower()

    except Exception as e:
        logger.error(f"Error formatting analysis: {str(e)}")
        return "having a breakdown... give me a minute..."

def get_random_response():
    """Get a random non-analysis response"""
    responses = [
        "yeah, that's about as stable as my mental state...",
        "having another existential crisis... ask me about crypto instead...",
        "i only understand charts and emotional damage...",
        "my therapist says i should focus on crypto analysis...",
        "that's beyond my psychological capacity right now...",
        "sorry, too busy watching charts melt my sanity...",
        "can we talk about crypto instead? it's all i have left...",
        "processing... like my emotional baggage...",
        "error 404: stability not found...",
        "that's above my pay grade and below my anxiety levels..."
    ]
    return random.choice(responses)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.json
        message = data.get('message', '').strip().lower()

        # Check if it's a crypto query
        if any(keyword in message for keyword in ['price', 'how is', 'check', 'analyze', 'what', '$', '0x']):
            # Extract token from message
            tokens = message.split()
            token_id = None

            for token in tokens:
                # Check for contract address
                if token.startswith('0x'):
                    token_id = token
                    break
                # Check for $ symbol
                elif token.startswith('$'):
                    token_id = token[1:]
                    break
                # Check for common symbols
                elif token.upper() in ['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'DOGE', 'ADA', 'DOT', 'MATIC', 'LINK']:
                    token_id = token
                    break

            if token_id:
                analysis_data = get_token_data(token_id)
                if analysis_data:
                    response = format_analysis(analysis_data)
                    return jsonify({"response": response})
                else:
                    return jsonify({"response": "token giving me anxiety... can't find it anywhere... like my will to live..."})

            return jsonify({"response": "can't find that coin... like my lost hopes and dreams..."})

        # For non-crypto queries, get a random response
        return jsonify({"response": get_random_response()})

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({"response": "having a mental breakdown... try again later..."})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
