from flask import Flask, request, jsonify, render_template
import os
import json
import time
import requests
import logging
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ta
import aiohttp
import asyncio

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

def get_kline_data(symbol, interval='30s', limit=100):
    """Get recent kline/candlestick data"""
    try:
        url = f"https://api.binance.com/api/v3/klines"
        params = {
            'symbol': f"{symbol}USDT",
            'interval': interval,
            'limit': limit
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                           'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                                           'taker_buy_quote', 'ignored'])

            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)

            return df
        return None
    except Exception as e:
        logger.error(f"Error fetching kline data: {str(e)}")
        return None

def calculate_indicators(df):
    """Calculate technical indicators using ta library"""
    try:
        # Calculate RSI
        df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()

        # Calculate MACD
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_hist'] = macd.macd_diff()

        # Calculate Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['close'])
        df['bb_high'] = bollinger.bollinger_hband()
        df['bb_mid'] = bollinger.bollinger_mavg()
        df['bb_low'] = bollinger.bollinger_lband()

        # Calculate momentum
        df['momentum'] = ta.momentum.ROCIndicator(df['close']).roc()

        # Calculate trend strength (ADX)
        adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
        df['adx'] = adx.adx()

        # Get latest values
        latest = df.iloc[-1]
        return {
            'rsi': latest['rsi'],
            'macd': latest['macd'],
            'macd_signal': latest['macd_signal'],
            'macd_hist': latest['macd_hist'],
            'bb_upper': latest['bb_high'],
            'bb_middle': latest['bb_mid'],
            'bb_lower': latest['bb_low'],
            'momentum': latest['momentum'],
            'adx': latest['adx'],
            'close': latest['close'],
            'prev_close': df.iloc[-2]['close']
        }
    except Exception as e:
        logger.error(f"Error calculating indicators: {str(e)}")
        return None

def get_market_sentiment(indicators):
    """Analyze market sentiment based on technical indicators"""
    sentiment = 0

    # RSI analysis
    if indicators['rsi'] > 70:
        sentiment -= 1  # Overbought
    elif indicators['rsi'] < 30:
        sentiment += 1  # Oversold

    # MACD analysis
    if indicators['macd'] > indicators['macd_signal']:
        sentiment += 1
    else:
        sentiment -= 1

    # Bollinger Bands analysis
    if indicators['close'] > indicators['bb_upper']:
        sentiment -= 1  # Overbought
    elif indicators['close'] < indicators['bb_lower']:
        sentiment += 1  # Oversold

    # Momentum
    if indicators['momentum'] > 0:
        sentiment += 1
    else:
        sentiment -= 1

    # ADX for trend strength
    trend_strength = "weak" if indicators['adx'] < 25 else "strong"

    # Price change
    price_change = ((indicators['close'] - indicators['prev_close']) / indicators['prev_close']) * 100

    return {
        'sentiment': sentiment,
        'trend_strength': trend_strength,
        'price_change': price_change,
        'rsi_condition': "overbought" if indicators['rsi'] > 70 else "oversold" if indicators['rsi'] < 30 else "neutral"
    }

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
            'pyth': 'HZ1JovNiVvGrGNiiYvEozEVgZ88xvNbhB9bmHhxQuTc9'
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

def get_random_response():
    """Get a random non-analysis response"""
    responses = [
        "ready to analyze any coin you throw at me!",
        "feed me some tickers! let's see those charts!",
        "crypto analysis mode activated! what are we checking?",
        "charts are my specialty! give me a symbol!",
        "ready to dive into some price action!",
        "scanning markets! which coin should we analyze?",
        "neural networks primed for chart analysis!",
        "show me a ticker and watch the magic happen!",
        "market scanner online! what's our target?",
        "ready to crunch those numbers! what are we looking at?"
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
