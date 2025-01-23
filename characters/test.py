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
    """Get real-time token data with accurate prices"""
    try:
        # Normalize token input
        token_id = token_id.lower().strip()

        # Handle major coins with Binance API first (most real-time prices)
        major_coins = {
            'sol': 'SOLUSDT',
            'btc': 'BTCUSDT',
            'eth': 'ETHUSDT',
            'bnb': 'BNBUSDT',
            'xrp': 'XRPUSDT',
            'doge': 'DOGEUSDT',
            'ada': 'ADAUSDT',
            'matic': 'MATICUSDT',
            'link': 'LINKUSDT'
        }

        if token_id in major_coins:
            symbol = major_coins[token_id]
            # Get real-time price from Binance
            price_url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
            ticker_url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"

            price_resp = requests.get(price_url)
            ticker_resp = requests.get(ticker_url)

            if price_resp.status_code == 200 and ticker_resp.status_code == 200:
                price_data = price_resp.json()
                ticker_data = ticker_resp.json()

                # Get market cap from CoinGecko for completeness
                cg_ids = {
                    'sol': 'solana',
                    'btc': 'bitcoin',
                    'eth': 'ethereum',
                    'bnb': 'binancecoin',
                    'xrp': 'ripple',
                    'doge': 'dogecoin',
                    'ada': 'cardano',
                    'matic': 'matic-network',
                    'link': 'chainlink'
                }

                mcap = 0
                try:
                    cg_url = f"https://api.coingecko.com/api/v3/simple/price?ids={cg_ids[token_id]}&vs_currencies=usd&include_market_cap=true"
                    cg_resp = requests.get(cg_url)
                    if cg_resp.status_code == 200:
                        cg_data = cg_resp.json()
                        mcap = cg_data[cg_ids[token_id]]['usd_market_cap']
                except:
                    pass

                return {
                    'price': float(price_data['price']),
                    'market_cap': mcap,
                    'change_24h': float(ticker_data['priceChangePercent']),
                    'volume_24h': float(ticker_data['volume']) * float(price_data['price']),
                    'source': 'binance',
                    'extra': {
                        'high_24h': float(ticker_data['highPrice']),
                        'low_24h': float(ticker_data['lowPrice'])
                    }
                }

        # For other tokens, try DexScreener
        dex_resp = requests.get(f"https://api.dexscreener.com/latest/dex/search?q={token_id}")
        if dex_resp.status_code == 200:
            data = dex_resp.json()
            if data.get('pairs') and len(data['pairs']) > 0:
                pair = data['pairs'][0]  # Get most liquid pair
                mcap = float(pair['priceUsd']) * float(pair.get('baseToken', {}).get('totalSupply', 0))
                return {
                    'price': float(pair['priceUsd']),
                    'market_cap': mcap,
                    'change_24h': float(pair.get('priceChange', {}).get('h24', 0)),
                    'volume_24h': float(pair.get('volume', {}).get('h24', 0)),
                    'source': 'dexscreener',
                    'extra': {
                        'liquidity': pair.get('liquidity', {}).get('usd', 0),
                        'dex': pair.get('dexId', 'unknown'),
                        'name': pair.get('baseToken', {}).get('name', 'unknown')
                    }
                }

    except Exception as e:
        logger.error(f"Error in get_token_data: {str(e)}")
    return None

def format_analysis(analysis_data):
    """Format market analysis with accurate numbers"""
    try:
        price = analysis_data.get('price', 0)
        mcap = analysis_data.get('market_cap', 0)
        change = analysis_data.get('change_24h', 0)
        volume = analysis_data.get('volume_24h', 0)
        source = analysis_data.get('source', 'unknown')
        extra = analysis_data.get('extra', {})

        # Format market cap
        if mcap >= 1_000_000_000:
            mcap_str = f"${mcap/1_000_000_000:.2f}B"
        elif mcap >= 1_000_000:
            mcap_str = f"${mcap/1_000_000:.2f}M"
        else:
            mcap_str = f"${mcap:,.0f}"

        # Format price based on its magnitude
        if price < 0.01:
            price_str = f"${price:.8f}"
        elif price < 1:
            price_str = f"${price:.4f}"
        else:
            price_str = f"${price:,.2f}"

        # Base response
        base_info = f"Price: {price_str} - MCap: {mcap_str}"

        # Additional info
        extra_parts = [
            f"24h Change: {change:+.2f}%",
            f"Volume: ${volume:,.0f}"
        ]

        if extra.get('liquidity'):
            extra_parts.append(f"Liquidity: ${extra['liquidity']:,.0f}")
        if extra.get('dex'):
            extra_parts.append(f"DEX: {extra['dex']}")
        if extra.get('high_24h'):
            extra_parts.append(f"24h High: ${float(extra['high_24h']):,.2f}")
        if extra.get('low_24h'):
            extra_parts.append(f"24h Low: ${float(extra['low_24h']):,.2f}")

        return f"{base_info} ! {' - '.join(extra_parts)}"

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
        logger.info(f"Received message: {message}")

        # Extract token - support various formats
        token = None

        # Remove common prefixes and get the token
        message = message.replace('price of ', '').replace('check ', '')
        message = message.replace('how is ', '').replace('what is ', '')
        message = message.replace('$', '').replace('#', '')

        words = message.split()
        if words:
            token = words[0]  # Take first word as token

            # If it's a contract address, use as is
            if token.startswith('0x'):
                token = token
            else:
                # Try to find the token/coin name
                token = token.strip().lower()

        if token:
            analysis_data = get_token_data(token)
            if analysis_data:
                response = format_analysis(analysis_data)
                return jsonify({"response": response})
            else:
                return jsonify({"response": "can't find that token! double check the name or address!"})

        return jsonify({"response": "what token should i look up? give me a name or address!"})

    except Exception as e:
        logger.error(f"Error in ask route: {str(e)}")
        return jsonify({"response": "system hiccup! give me a moment to recalibrate!"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
