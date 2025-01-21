from flask import Flask, render_template, request, jsonify
import json
import time
import random
import requests
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
from flask_cors import CORS
import logging
import os

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

class TweakerResponses:
    def __init__(self):
        self.trend_responses = {
            'strong_bull': [
                "market's pumping harder than my anxiety levels... {percent_change}% up, volume {volume_change}% changed"
            ],
            'bull': [
                "slight uptrend, like my hopes before they get crushed... {percent_change}% up"
            ],
            'neutral': [
                "market's as stable as my mental state... {percent_change}% change"
            ],
            'bear': [
                "downtrend detected, like my life trajectory... {percent_change}% down"
            ],
            'strong_bear': [
                "market's dumping harder than my ex... {percent_change}% down, volume {volume_change}% changed"
            ]
        }
        
        self.volume_analysis = {
            'high_volume_bullish': [
                "volume's exploding with {volume_change}% change... buyers are more active than my self-doubt"
            ],
            'high_volume_bearish': [
                "massive selling, {volume_change}% volume change... paper hands everywhere, like my fragile dreams"
            ],
            'low_volume_bullish': [
                "quiet upward movement... like my suppressed optimism"
            ],
            'low_volume_bearish': [
                "silent bleeding... reminds me of my portfolio"
            ]
        }

class TechnicalAnalysis:
    def __init__(self, prices, volumes):
        self.prices = prices
        self.volumes = volumes
        
    def calculate_rsi(self, periods=14):
        deltas = np.diff(self.prices)
        seed = deltas[:periods+1]
        up = seed[seed >= 0].sum()/periods
        down = -seed[seed < 0].sum()/periods
        rs = up/down
        rsi = np.zeros_like(self.prices)
        rsi[:periods] = 100. - 100./(1. + rs)

        for i in range(periods, len(self.prices)):
            delta = deltas[i - 1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta

            up = (up*(periods - 1) + upval)/periods
            down = (down*(periods - 1) + downval)/periods
            rs = up/down
            rsi[i] = 100. - 100./(1. + rs)
        
        return rsi[-1]
    
    def calculate_macd(self, fast=12, slow=26, signal=9):
        exp1 = np.exp(self.prices).ewm(span=fast, adjust=False).mean()
        exp2 = np.exp(self.prices).ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=signal, adjust=False).mean()
        return macd.iloc[-1] if hasattr(macd, 'iloc') else macd[-1]
    
    def calculate_bollinger_bands(self, window=20):
        sma = np.mean(self.prices[-window:])
        std = np.std(self.prices[-window:])
        return {
            'upper': sma + (std * 2),
            'middle': sma,
            'lower': sma - (std * 2)
        }
    
    def get_support_resistance(self):
        prices = self.prices[-50:]  # Use last 50 prices for analysis
        levels = []
        
        # Find local maxima and minima
        for i in range(2, len(prices)-2):
            if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                levels.append(prices[i])
            if prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                levels.append(prices[i])
        
        # Cluster levels
        levels = np.array(levels)
        if len(levels) > 0:
            mean = np.mean(levels)
            std = np.std(levels)
            resistance = levels[levels > mean]
            support = levels[levels < mean]
            return {'support': sorted(support), 'resistance': sorted(resistance)}
        return {'support': [], 'resistance': []}
class MarketAnalysis:
    def __init__(self, data):
        self.data = data
        self.prices = np.array([price[1] for price in data['history']])
        self.volumes = np.array([vol[1] for vol in data.get('volume_history', [])] or [0] * len(self.prices))
        self.technical = TechnicalAnalysis(self.prices, self.volumes)
        self.responses = TweakerResponses()

    def _calculate_trend_strength(self, percent_change, rsi, macd):
        if percent_change > 5 and rsi > 70 and macd > 0:
            return 'strong_bull'
        elif percent_change > 2 and rsi > 60:
            return 'bull'
        elif percent_change < -5 and rsi < 30 and macd < 0:
            return 'strong_bear'
        elif percent_change < -2 and rsi < 40:
            return 'bear'
        return 'neutral'

    def analyze_market_structure(self):
        current_price = self.prices[-1]
        high = max(self.prices)
        low = min(self.prices)
        
        # Calculate key metrics
        percent_change_24h = self.data['change_24h']
        volume_change = ((self.volumes[-1] / self.volumes[-2]) - 1) * 100 if len(self.volumes) > 1 else 0
        rsi = self.technical.calculate_rsi()
        macd = self.technical.calculate_macd()
        bollinger = self.technical.calculate_bollinger_bands()
        
        # Determine trend strength
        trend_strength = self._calculate_trend_strength(percent_change_24h, rsi, macd)
        
        # Get support/resistance levels
        levels = self.technical.get_support_resistance()
        
        return {
            'trend': trend_strength,
            'metrics': {
                'percent_change': percent_change_24h,
                'volume_change': volume_change,
                'rsi': rsi,
                'macd': macd,
                'support': levels['support'][0] if levels['support'] else current_price * 0.95,
                'resistance': levels['resistance'][0] if levels['resistance'] else current_price * 1.05,
                'price': current_price,
                'patterns': []
            }
        }

    def predict_price(self):
        # Calculate various technical indicators for prediction
        rsi = self.technical.calculate_rsi()
        macd = self.technical.calculate_macd()
        bb = self.technical.calculate_bollinger_bands()
        current_price = self.prices[-1]
        
        # Initialize prediction variables
        confidence = "low"
        direction = "sideways"
        reasoning = []
        
        # RSI Analysis
        if rsi > 70:
            direction = "downward"
            reasoning.append("overbought RSI")
            confidence = "medium"
        elif rsi < 30:
            direction = "upward"
            reasoning.append("oversold RSI")
            confidence = "medium"
            
        # MACD Analysis
        if macd > 0:
            if direction == "upward":
                confidence = "high"
            direction = "upward"
            reasoning.append("positive MACD")
        elif macd < 0:
            if direction == "downward":
                confidence = "high"
            direction = "downward"
            reasoning.append("negative MACD")
            
        # Bollinger Bands Analysis
        if current_price > bb['upper']:
            if direction == "downward":
                confidence = "high"
            direction = "downward"
            reasoning.append("price above upper BB")
        elif current_price < bb['lower']:
            if direction == "upward":
                confidence = "high"
            direction = "upward"
            reasoning.append("price below lower BB")
            
        # Calculate target prices
        volatility = np.std(self.prices) / np.mean(self.prices)
        move_percentage_1d = 0.02 * (1 + volatility)  # 2% base move adjusted for volatility
        move_percentage_7d = 0.05 * (1 + volatility)  # 5% base move adjusted for volatility
        
        if direction == "upward":
            target_1d = current_price * (1 + move_percentage_1d)
            target_7d = current_price * (1 + move_percentage_7d)
        elif direction == "downward":
            target_1d = current_price * (1 - move_percentage_1d)
            target_7d = current_price * (1 - move_percentage_7d)
        else:
            target_1d = current_price
            target_7d = current_price
            
        return {
            'direction': direction,
            'confidence': confidence,
            'reasoning': reasoning,
            'target_1d': target_1d,
            'target_7d': target_7d
        }

    def generate_analysis_response(self):
        analysis = self.analyze_market_structure()
        metrics = analysis['metrics']
        prediction = self.predict_price()
        
        # Format response data
        response_data = {
            'percent_change': f"{metrics['percent_change']:.1f}",
            'volume_change': f"{metrics['volume_change']:.1f}",
            'rsi': f"{metrics['rsi']:.0f}",
            'macd_value': f"{metrics['macd']:.4f}",
            'support': f"${metrics['support']:.8f}",
            'resistance': f"${metrics['resistance']:.8f}",
            'price_level': f"${metrics['price']:.8f}"
        }

        # Build comprehensive analysis
        responses = []
        
        # 1. Current state
        trend_response = self.responses.trend_responses[analysis['trend']]
        responses.append(random.choice(trend_response).format(**response_data))
        
        # 2. Technical Analysis
        responses.append(f"technical analysis shows RSI at {response_data['rsi']} and MACD at {response_data['macd_value']}, "
                        f"like my deteriorating mental state. support at {response_data['support']} and resistance at {response_data['resistance']}")
        
        # 3. Volume Analysis
        volume_state = "high_volume" if abs(metrics['volume_change']) > 20 else "low_volume"
        volume_trend = "bullish" if 'bull' in analysis['trend'] else 'bearish'
        volume_key = f"{volume_state}_{volume_trend}"
        if volume_key in self.responses.volume_analysis:
            responses.append(random.choice(self.responses.volume_analysis[volume_key]).format(**response_data))
        
        # 4. Price Prediction
        prediction_text = (
            f"my existential dread suggests a {prediction['direction']} move with {prediction['confidence']} confidence... "
            f"24h target: ${prediction['target_1d']:.8f}, 7d target: ${prediction['target_7d']:.8f}... "
            f"based on {', '.join(prediction['reasoning'])}, but what do i know, i'm just a depressed algorithm"
        )
        responses.append(prediction_text)

        # Combine responses
        final_response = f"{self.data['name'].upper()} at {response_data['price_level']}. "
        final_response += " ".join(responses)
        
        return final_response.lower()
    def get_coin_data_by_id_or_address(identifier):
    try:
        print(f"Attempting to get data for {identifier}")  # Debug log
        
        # Add headers to prevent rate limiting
        headers = {
            'User-Agent': 'Mozilla/5.0',
            'Accept': 'application/json'
        }
        
        # Try CoinGecko search first
        search_url = f"https://api.coingecko.com/api/v3/search?query={identifier}"
        print(f"Searching: {search_url}")  # Debug log
        search_response = requests.get(search_url, headers=headers, timeout=10)
        
        if search_response.status_code != 200:
            print(f"CoinGecko API error: {search_response.status_code}")  # Debug log
            return None
            
        search_result = search_response.json()
        print(f"Search result: {search_result}")  # Debug log
        
        if search_result.get('coins'):
            coin_id = search_result['coins'][0]['id']
            return get_coingecko_data(coin_id)
        else:
            print(f"No coins found for {identifier}")  # Debug log
            return get_dexscreener_data(identifier)
    except Exception as e:
        print(f"Error in get_coin_data_by_id_or_address: {str(e)}")  # Debug log
        return None

def get_coingecko_data(coin_id):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0',
            'Accept': 'application/json'
        }
        
        # Get current price and 24h change
        price_url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd&include_24hr_change=true&include_24hr_vol=true"
        print(f"Getting price data: {price_url}")  # Debug log
        price_response = requests.get(price_url, headers=headers, timeout=10)
        
        if price_response.status_code != 200:
            print(f"Price API error: {price_response.status_code}")  # Debug log
            return None
            
        price_data = price_response.json()
        
        # Get historical data
        history_url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days=14&interval=daily"
        print(f"Getting history data: {history_url}")  # Debug log
        history_response = requests.get(history_url, headers=headers, timeout=10)
        
        if history_response.status_code != 200:
            print(f"History API error: {history_response.status_code}")  # Debug log
            return None
            
        history = history_response.json()
        
        return {
            'name': coin_id,
            'price': price_data[coin_id]['usd'],
            'change_24h': price_data[coin_id]['usd_24h_change'],
            'volume_24h': price_data[coin_id].get('usd_24h_vol', 0),
            'history': history['prices'],
            'volume_history': history.get('total_volumes', []),
            'source': 'coingecko'
        }
    except Exception as e:
        print(f"Error in get_coingecko_data: {str(e)}")  # Debug log
        return None

def get_dexscreener_data(address):
    try:
        url = f"https://api.dexscreener.com/latest/dex/tokens/{address}"
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if 'pairs' in data and len(data['pairs']) > 0:
            pair = data['pairs'][0]
            return {
                'name': pair['baseToken']['symbol'],
                'price': float(pair['priceUsd']),
                'change_24h': float(pair['priceChange']['h24']),
                'volume_24h': float(pair['volume']['h24']),
                'history': [[int(time.time() * 1000), float(pair['priceUsd'])]],
                'volume_history': [[int(time.time() * 1000), float(pair['volume']['h24'])]],
                'source': 'dexscreener'
            }
    except Exception as e:
        print(f"Error in get_dexscreener_data: {str(e)}")  # Debug log
        return None

def get_agent_response(message):
    responses = [
        "i'm too depressed to process that... try /analyze <coin> or /help",
        "sorry, i only understand market analysis... like my limited emotional capacity",
        "that's beyond my programming... and my will to live. try /analyze <coin>",
        "i'm just a sad algorithm... try /analyze <coin> or /help"
    ]
    return random.choice(responses)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    try:
        print("Received request:", request.json)  # Debug log
        user_input = request.json['message'].lower()
        
        if user_input.startswith("/analyze"):
            try:
                identifier = user_input.split()[1]
                print(f"Analyzing {identifier}")  # Debug log
                data = get_coin_data_by_id_or_address(identifier)
                if data:
                    print("Got data:", data)  # Debug log
                    analyzer = MarketAnalysis(data)
                    response = analyzer.generate_analysis_response()
                else:
                    response = "couldn't find that coin... like my will to live. try another ticker or contract address."
            except Exception as e:
                print(f"Analysis error: {str(e)}")  # Debug log
                response = f"analysis failed harder than my last relationship. error: {str(e)}"
        elif user_input.startswith("prediction"):
            try:
                coin = user_input.split()[2] if len(user_input.split()) > 2 else user_input.split()[1]
                data = get_coin_data_by_id_or_address(coin)
                if data:
                    analyzer = MarketAnalysis(data)
                    prediction = analyzer.predict_price()
                    response = (
                        f"my crystal ball of depression shows {prediction['direction']} movement with {prediction['confidence']} confidence... "
                        f"24h target: ${prediction['target_1d']:.8f}, 7d target: ${prediction['target_7d']:.8f}... "
                        f"based on {', '.join(prediction['reasoning'])}"
                    )
                else:
                    response = "coin not found... like my happiness"
            except Exception as e:
                response = f"prediction failed... just like everything else in my life. error: {str(e)}"
        elif user_input == "/help":
            response = """
            helping others... how meaningless. but here:
            /analyze <ticker/address> - analyze any token with price prediction
            prediction for <ticker/address> - get detailed price prediction
            /help - you're looking at it, unfortunately
            
            example: /analyze btc or prediction for eth
            
            or just chat with me if you're feeling particularly masochistic.
            """
        else:
            response = get_agent_response(user_input)
        
        return jsonify({"response": response})
    except Exception as e:
        print(f"Route error: {str(e)}")  # Debug log
        return jsonify({"response": f"something went wrong... like everything else. error: {str(e)}"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
