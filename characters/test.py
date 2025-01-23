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
    """Get token data with technical analysis"""
    try:
        if token_id.upper() in ['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'DOGE', 'ADA', 'DOT', 'MATIC', 'LINK']:
            # Get current price data
            symbol = token_id.upper()
            price_resp = requests.get(f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}USDT")
            ticker_resp = requests.get(f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}USDT")

            if price_resp.status_code == 200 and ticker_resp.status_code == 200:
                price_data = price_resp.json()
                ticker_data = ticker_resp.json()

                # Get technical analysis data
                kline_data = get_kline_data(symbol)
                if kline_data is not None:
                    indicators = calculate_indicators(kline_data)
                    sentiment = get_market_sentiment(indicators)

                    return {
                        'price': float(price_data['price']),
                        'change_24h': float(ticker_data['priceChangePercent']),
                        'volume_24h': float(ticker_data['volume']) * float(price_data['price']),
                        'source': 'binance',
                        'technical': {
                            'rsi': indicators['rsi'],
                            'sentiment': sentiment['sentiment'],
                            'trend_strength': sentiment['trend_strength'],
                            'short_term_change': sentiment['price_change'],
                            'rsi_condition': sentiment['rsi_condition']
                        }
                    }
    except Exception as e:
        logger.error(f"Error in get_token_data: {str(e)}")
    return None

def format_analysis(analysis_data):
    """Format market analysis with technical insights"""
    try:
        price = analysis_data.get('price', 0)
        change = analysis_data.get('change_24h', 0)
        volume = analysis_data.get('volume_24h', 0)
        technical = analysis_data.get('technical', {})

        # Prediction phrases based on technical analysis
        sentiment = technical.get('sentiment', 0)
        trend_strength = technical.get('trend_strength', 'neutral')
        rsi_condition = technical.get('rsi_condition', 'neutral')

        prediction_phrases = [
            "looking like a breakout incoming" if sentiment > 1 else "might see a pullback soon" if sentiment < -1 else "consolidating for next move",
            f"momentum is {trend_strength} on the 30s chart",
            f"market is {rsi_condition} right now",
            "buy pressure building up" if sentiment > 0 else "sellers taking control" if sentiment < 0 else "traders in standoff"
        ]

        # Market action phrases
        action_phrases = [
            f"price action at ${price:.8f}",
            f"volume pumping at ${volume:,.2f}",
            f"{'climbing' if change > 0 else 'dropping'} {abs(change):.2f}% on the daily",
            random.choice(prediction_phrases)
        ]

        # Randomize order and select 3 phrases
        random.shuffle(action_phrases)
        selected_phrases = action_phrases[:3]

        # Join with varied connectors
        connectors = [' ! ', ' - ', ' && ', ' >> ']
        return random.choice(connectors).join(selected_phrases)

    except Exception as e:
        logger.error(f"Error formatting analysis: {str(e)}")
        return "analyzing market conditions! stand by for update!"

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
                    return jsonify({"response": "token not found in my database! try another one!"})

            return jsonify({"response": "couldn't find that coin! give me a valid ticker or address!"})

        # For non-crypto queries, get a random response
        return jsonify({"response": get_random_response()})

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({"response": "system hiccup! give me a moment to recalibrate!"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
