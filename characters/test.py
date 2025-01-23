from flask import Flask, request, jsonify, render_template
import os
import requests
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='../templates', static_folder='../static')

@app.route('/')
def home():
    return render_template('index.html')

def get_token_data(token_id):
    """Get real-time token data using Jupiter API"""
    try:
        # Normalize token input
        token_id = token_id.lower().strip()

        # Jupiter token mapping
        token_addresses = {
            'sol': 'So11111111111111111111111111111111111111112',
            'bonk': 'DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263',
            'wen': 'WENWENvqqNya429ubCdXG1xHz4XGxdC3SdqmeVNyHLr7',
            'jup': 'JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN',
            'rac': 'RAC3qCKy6CHt6HtMYDhDQvet8NyqRWgbKnTMhVzKfwt',
            'pyth': 'HZ1JovNiVvGrGNiiYvEozEVgZ88xvNbhB9bmHhxQuTc9',
            'jito': 'J1toso1uCk3RLmjorhTtrVwY9HJ7X8V9yYac6Y7kGCPn',
            'msol': 'mSoLzYCxHdYgdzU16g5QSh3i5K3z3KZK7ytfqcJm7So',
            'ray': '4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R'
        }

        token_address = token_addresses.get(token_id)
        if not token_address:
            return None

        # Get price from Jupiter
        url = f"https://price.jup.ag/v4/price?ids={token_address}"
        logger.info(f"Requesting price from Jupiter for {token_id}: {url}")

        response = requests.get(url)
        logger.info(f"Jupiter response: {response.text}")

        if response.status_code == 200:
            data = response.json()
            price_data = data.get('data', {}).get(token_address)

            if price_data:
                return {
                    'price': float(price_data['price']),
                    'volume_24h': float(price_data.get('volume24h', 0)),
                    'source': 'jupiter',
                    'extra': {
                        'dex': 'Jupiter',
                        'last_updated': datetime.now().strftime('%H:%M:%S')
                    }
                }

    except Exception as e:
        logger.error(f"Error in get_token_data: {str(e)}")
        logger.exception("Full traceback:")
    return None

def format_analysis(analysis_data):
    """Format market analysis"""
    try:
        price = analysis_data.get('price', 0)
        volume = analysis_data.get('volume_24h', 0)
        extra = analysis_data.get('extra', {})

        # Format price
        if price >= 1:
            price_str = f"${price:,.2f}"
        elif price >= 0.01:
            price_str = f"${price:.4f}"
        else:
            price_str = f"${price:.8f}"

        # Format volume
        if volume >= 1_000_000_000:
            volume_str = f"${volume/1_000_000_000:.2f}B"
        elif volume >= 1_000_000:
            volume_str = f"${volume/1_000_000:.2f}M"
        else:
            volume_str = f"${volume:,.0f}"

        parts = [
            f"Price: {price_str}",
            f"Volume: {volume_str}",
            f"Updated: {extra.get('last_updated', 'now')}"
        ]

        return " - ".join(parts)

    except Exception as e:
        logger.error(f"Error formatting analysis: {str(e)}")
        return "error formatting data"

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.json
        message = data.get('message', '').strip().lower()
        logger.info(f"Received message: {message}")

        # Clean up the message
        message = message.replace('price of ', '').replace('check ', '')
        message = message.replace('how is ', '').replace('what is ', '')
        message = message.replace('$', '').replace('#', '')

        # Get first word as token
        token = message.split()[0] if message else None

        if token:
            logger.info(f"Looking up token: {token}")
            analysis_data = get_token_data(token)

            if analysis_data:
                response = format_analysis(analysis_data)
                return jsonify({"response": response})
            else:
                return jsonify({"response": "can't find that token! check the symbol or try another one!"})

        return jsonify({"response": "what token should i look up? give me a symbol!"})

    except Exception as e:
        logger.error(f"Error in ask route: {str(e)}")
        logger.exception("Full traceback:")
        return jsonify({"response": "system hiccup! give me a moment to recalibrate!"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
