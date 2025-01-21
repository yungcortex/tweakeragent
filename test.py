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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Add route for main page
@app.route('/')
def index():
    return render_template('index.html')

# Add route for favicon
@app.route('/favicon.ico')
def favicon():
    return '', 204


class TweakerResponses:
    def __init__(self):
        self.trend_responses = {
            "strong_bullish": [
                "breaking out harder than my last mental breakdown... {percent_change}% gain looking suspicious",
                "pumping {percent_change}% like my heart rate during a margin call",
                "chart's more vertical than my anxiety spike... {percent_change}% up but don't get excited",
                "momentum's stronger than my coffee addiction... {percent_change}% gain with {volume_change}% volume surge",
                "bullish divergence clearer than my trust issues... RSI at {rsi} with {percent_change}% gain",
                "MACD crossing more decisively than my life choices... {macd_value} with {volume_change}% volume increase",
            ],
            "bullish": [
                "technically bullish but so was my therapist before they quit... {percent_change}% up",
                "showing strength at {price_level}, probably just to crush our hopes later",
                "upward channel forming, like my mounting medical bills... support at {support}",
                "breaking resistance at {resistance} like i break promises",
                "{percent_change}% up with RSI at {rsi}... suspiciously optimistic",
                "volume profile stronger than my will to live... {volume_change}% increase",
            ],
            "weak_bullish": [
                "trying to break {resistance} like i try to break bad habits",
                "barely bullish... {percent_change}% up, like my minimum effort",
                "slight uptrend forming, much like my fragile hopes... support at {support}",
                "RSI at {rsi} suggesting momentum, but so did my ex's promises",
                "volume increasing {volume_change}% but commitment issues evident",
            ],
            "strong_bearish": [
                "dumping {percent_change}% harder than my life decisions",
                "support at {support} breaking down like my mental state",
                "selling pressure higher than my existential dread... {volume_change}% volume spike",
                "RSI at {rsi} screaming oversold, like my soul to capitalism",
                "MACD diving {macd_value} deeper than my depression",
                "breaking down with {volume_change}% volume... like watching my dreams shatter in 4K",
            ],
            "bearish": [
                "bearish, like my outlook on existence... down {percent_change}%",
                "heading down faster than my life choices... support broken at {support}",
                "resistance forming at {resistance} like emotional walls",
                "death cross forming... fitting for my portfolio",
                "volume profile weaker than my social life... {volume_change}% decrease",
            ],
            "weak_bearish": [
                "slightly bearish... {percent_change}% down, like my motivation",
                "testing support at {support} like testing my patience",
                "RSI at {rsi} suggesting weakness, much like my resolve",
                "volume declining {volume_change}% like my hope",
            ],
            "neutral": [
                "consolidating between {support} and {resistance} like my trapped emotions",
                "volume dead at {volume_change}% change... like my social life",
                "RSI at {rsi}, as neutral as my therapist pretends to be",
                "MACD crossing zero like my bank balance... {macd_value}",
                "ranging pattern tighter than my budget... between {support} and {resistance}",
            ],
        }
        self.volume_analysis = {
            "high_volume_bullish": [
                "volume {volume_change}% up with price... whales accumulating like I accumulate regrets",
                "buying pressure stronger than my caffeine dependency... {volume_change}% volume surge",
                "massive {volume_change}% volume spike... someone's more committed than my ex",
            ],
            "high_volume_bearish": [
                "selling volume up {volume_change}%... dumping harder than my last relationship",
                "distribution volume higher than my therapy bills... {volume_change}% increase",
                "panic selling with {volume_change}% volume... paper hands weaker than my resolve",
            ],
            "low_volume_bullish": [
                "price up but volume down {volume_change}%... suspicious as my landlord's smile",
                "weak hands finished selling... volume dry as my humor",
                "bullish move with {volume_change}% volume drop... trust issues intensifying",
            ],
            "low_volume_bearish": [
                "no volume ({volume_change}%) in this dump... dead as my dreams",
                "bearish move but volume's lighter than my wallet... {volume_change}% decrease",
                "selling on low volume... paper hands fragile as my ego",
            ],
        }

        self.pattern_responses = {
            "double_top": [
                "double top at {resistance} like my double dose of antidepressants",
                "rejection at {resistance} twice... persistent as my existential crisis",
                "double top forming... trust issues confirmed",
            ],
            "double_bottom": [
                "double bottom at {support} like my rock bottom experiences",
                "bouncing from {support} twice... resilient as my debt",
                "double bottom pattern... even the chart has commitment issues",
            ],
            "head_shoulders": [
                "head and shoulders pattern... complex as my relationship status",
                "distribution pattern clearer than my future",
                "head and shoulders forming... like the weight of my decisions",
            ],
        }

        self.indicator_responses = {
            "overbought": [
                "RSI at {rsi}... more overbought than my emotional capacity",
                "indicators screaming overbought... like my anxiety levels",
                "technical indicators maxed out like my credit cards... RSI {rsi}",
            ],
            "oversold": [
                "RSI at {rsi}... oversold like my soul to the crypto gods",
                "oversold on all timeframes... like my will to continue",
                "indicators suggesting a bounce... like my recurring disappointments",
            ],
        }


class TechnicalAnalysis:
    def __init__(self, prices, volumes=None):
        # Validate input data
        if not isinstance(prices, (list, np.ndarray)) or len(prices) == 0:
            raise ValueError("Prices must be a non-empty list or numpy array")

        # Convert to numpy arrays and validate values
        self.prices = np.array([float(p) for p in prices if p is not None and float(p) > 0])

        if volumes is not None:
            self.volumes = np.array([float(v) if v is not None else 0 for v in volumes])
        else:
            self.volumes = np.zeros_like(self.prices)

        # Validate data length
        if len(self.prices) < 2:
            raise ValueError("Insufficient price data for analysis")

        # Remove any remaining invalid values
        self.prices = self.prices[~np.isnan(self.prices)]
        self.volumes = self.volumes[~np.isnan(self.volumes)]

        # Ensure minimum data length
        if len(self.prices) < 14:  # Minimum required for most indicators
            raise ValueError("Minimum 14 periods of data required")

    def calculate_sma(self, period):
        return np.mean(self.prices[-period:])

    def calculate_ema(self, period):
        weights = np.exp(np.linspace(-1.0, 0.0, period))
        weights /= weights.sum()
        return np.convolve(self.prices, weights, mode="valid")[0]

    def calculate_rsi(self, periods=14):
        try:
            # Validate periods
            periods = int(periods)
            if periods < 2:
                raise ValueError("RSI periods must be >= 2")
            if periods > len(self.prices) - 1:
                raise ValueError("Not enough data for specified RSI periods")

            # Calculate price changes
            deltas = np.diff(self.prices)

            # Separate gains and losses
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)

            # Validate data
            if np.isnan(gains).any() or np.isnan(losses).any():
                logger.warning("Invalid values in RSI calculation")
                return 50

            # Calculate initial averages
            avg_gain = np.mean(gains[:periods])
            avg_loss = np.mean(losses[:periods])

            if np.isnan(avg_gain) or np.isnan(avg_loss):
                logger.warning("NaN in initial RSI averages")
                return 50

            # Initialize arrays for results
            rsi_values = np.zeros(len(self.prices))

            # Calculate first RSI value
            if avg_loss == 0:
                rsi_values[periods] = 100
            else:
                rs = avg_gain / avg_loss
                rsi_values[periods] = 100 - (100 / (1 + rs))

            # Calculate subsequent values using smoothing
            for i in range(periods + 1, len(self.prices)):
                avg_gain = ((avg_gain * (periods - 1)) + gains[i-1]) / periods
                avg_loss = ((avg_loss * (periods - 1)) + losses[i-1]) / periods

                if avg_loss == 0:
                    rsi_values[i] = 100
                else:
                    rs = avg_gain / avg_loss
                    rsi_values[i] = 100 - (100 / (1 + rs))

            # Validate final RSI value
            final_rsi = float(rsi_values[-1])
            if np.isnan(final_rsi) or final_rsi < 0 or final_rsi > 100:
                logger.warning(f"Invalid final RSI value: {final_rsi}")
                return 50

            return final_rsi

        except Exception as e:
            logger.error(f"Error in RSI calculation: {str(e)}")
            return 50

    def calculate_macd(self, fast=12, slow=26, signal=9):
        try:
            # Validate parameters
            if not all(isinstance(x, int) and x > 0 for x in [fast, slow, signal]):
                raise ValueError("MACD parameters must be positive integers")
            if slow <= fast:
                raise ValueError("MACD slow period must be greater than fast period")
            if len(self.prices) < slow + signal:
                raise ValueError("Insufficient data for MACD calculation")

            # Calculate EMAs
            exp1 = np.exp(self.prices).ewm(span=fast, adjust=False).mean()
            exp2 = np.exp(self.prices).ewm(span=slow, adjust=False).mean()
            macd = exp1 - exp2
            signal_line = macd.ewm(span=signal, adjust=False).mean()

            # Validate results
            final_macd = float(macd.iloc[-1] if hasattr(macd, 'iloc') else macd[-1])
            if np.isnan(final_macd):
                logger.warning("Invalid MACD value")
                return 0

            return final_macd

        except Exception as e:
            logger.error(f"Error in MACD calculation: {str(e)}")
            return 0

    def calculate_bollinger_bands(self, window=20, num_std=2):
        try:
            # Validate parameters
            if not isinstance(window, int) or window < 2:
                raise ValueError("Bollinger window must be integer >= 2")
            if len(self.prices) < window:
                raise ValueError("Insufficient data for Bollinger Bands")

            # Calculate bands
            rolling_mean = np.mean(self.prices[-window:])
            rolling_std = np.std(self.prices[-window:])

            # Validate results
            if np.isnan(rolling_mean) or np.isnan(rolling_std):
                logger.warning("Invalid Bollinger Bands values")
                return {
                    'upper': self.prices[-1] * 1.02,
                    'middle': self.prices[-1],
                    'lower': self.prices[-1] * 0.98
                }

            return {
                'upper': rolling_mean + (rolling_std * num_std),
                'middle': rolling_mean,
                'lower': rolling_mean - (rolling_std * num_std)
            }

        except Exception as e:
            logger.error(f"Error in Bollinger Bands calculation: {str(e)}")
            return {
                'upper': self.prices[-1] * 1.02,
                'middle': self.prices[-1],
                'lower': self.prices[-1] * 0.98
            }

    def check_head_and_shoulders(self):
        if len(self.prices) < 20:
            return False
        peaks = self._find_peaks(self.prices[-20:])
        if len(peaks) >= 3:
            return True
        return False

    def check_double_patterns(self):
        if len(self.prices) < 20:
            return False
        peaks = self._find_peaks(self.prices[-20:])
        troughs = self._find_peaks(-self.prices[-20:])
        return len(peaks) >= 2 or len(troughs) >= 2

    def check_triangle_patterns(self):
        if len(self.prices) < 20:
            return False
        highs = self.prices[-20:]
        lows = self.prices[-20:]
        high_slope = np.polyfit(range(len(highs)), highs, 1)[0]
        low_slope = np.polyfit(range(len(lows)), lows, 1)[0]
        return abs(high_slope - low_slope) < 0.1

    def _find_peaks(self, arr):
        peaks = []
        for i in range(1, len(arr) - 1):
            if arr[i - 1] < arr[i] > arr[i + 1]:
                peaks.append(i)
        return peaks

    def get_support_resistance(self, window=20):
        try:
            # Validate parameters
            if not isinstance(window, int) or window < 2:
                raise ValueError("Window must be integer >= 2")
            if len(self.prices) < window:
                raise ValueError("Insufficient data for support/resistance")

            prices = self.prices[-window:]  # Use last window prices
            levels = []

            # Find local maxima and minima
            for i in range(2, len(prices)-2):
                if prices[i] > prices[i-1] and prices[i] > prices[i+1]:  # Local max
                    levels.append(prices[i])
                if prices[i] < prices[i-1] and prices[i] < prices[i+1]:  # Local min
                    levels.append(prices[i])

            # If no levels found, use simple percentage-based levels
            if not levels:
                current_price = self.prices[-1]
                levels = [
                    current_price * 0.95,  # Support
                    current_price * 1.05   # Resistance
                ]

            # Cluster levels
            levels = np.array(levels)
            levels.sort()

            return {
                'support': [float(level) for level in levels[levels < self.prices[-1]]],
                'resistance': [float(level) for level in levels[levels > self.prices[-1]]]
            }

        except Exception as e:
            logger.error(f"Error in support/resistance calculation: {str(e)}")
            current_price = self.prices[-1]
            return {
                'support': [float(current_price * 0.95)],
                'resistance': [float(current_price * 1.05)]
            }


class MarketAnalysis:
    def __init__(self, coin_data):
        self.coin_data = coin_data
        self.prices = [float(price[1]) for price in coin_data.get('history', [])]
        self.volumes = [float(vol[1]) for vol in coin_data.get('volume_history', [])]

    def analyze_market(self):
        try:
            current_price = float(self.coin_data['price'])
            price_change = float(self.coin_data['change_24h'])
            volume_24h = float(self.coin_data['volume_24h'])

            # Basic analysis
            trend = {
                'price': current_price,
                'price_change': price_change,
                'volume_change': 0,
                'rsi': 50,  # Default values
                'macd': 0,
                'support': current_price * 0.95,
                'resistance': current_price * 1.05
            }

            return {'trend': trend}

        except Exception as e:
            logger.error(f"Market analysis error: {str(e)}")
            return None

    def predict_price(self):
        try:
            result = self.analyze_market()
            if not result:
                return None

            trend = result['trend']
            current_price = trend['price']

            # Simple prediction
            direction = 'bullish' if trend['price_change'] > 0 else 'bearish'
            confidence = 'medium'

            target_1d = current_price * (1 + (0.01 * (1 if direction == 'bullish' else -1)))
            target_7d = current_price * (1 + (0.03 * (1 if direction == 'bullish' else -1)))

            reasoning = [
                f"price {'up' if direction == 'bullish' else 'down'} {abs(trend['price_change']):.1f}%"
            ]

            return {
                'direction': direction,
                'confidence': confidence,
                'target_1d': target_1d,
                'target_7d': target_7d,
                'reasoning': reasoning
            }

        except Exception as e:
            logger.error(f"Price prediction error: {str(e)}")
            return None

    def predict_price(self):
        try:
            result = self.analyze_market()
            if not result:
                return None

            trend = result['trend']
            current_price = trend['price']

            # Determine direction and confidence
            signals = {
                'rsi': 1 if trend['rsi'] < 30 else -1 if trend['rsi'] > 70 else 0,
                'macd': 1 if trend['macd'] > 0 else -1 if trend['macd'] < 0 else 0,
                'price': 1 if trend['price_change'] > 0 else -1 if trend['price_change'] < 0 else 0
            }

            total_signal = sum(signals.values())

            # Determine direction and confidence
            directions = ['bearish', 'neutral', 'bullish']
            confidences = ['low', 'medium', 'high']

            direction_idx = 1  # neutral
            if total_signal >= 2:
                direction_idx = 2  # bullish
            elif total_signal <= -2:
                direction_idx = 0  # bearish

            confidence_idx = 1  # medium
            if abs(total_signal) >= 2:
                confidence_idx = 2  # high
            elif abs(total_signal) == 0:
                confidence_idx = 0  # low

            # Calculate targets
            volatility = np.std(self.prices) / np.mean(self.prices) if len(self.prices) > 1 else 0.01
            move_multiplier = 0.01 * (1 + volatility)

            target_1d = current_price * (1 + (move_multiplier * total_signal))
            target_7d = current_price * (1 + (move_multiplier * 3 * total_signal))

            # Generate reasoning
            reasoning = []
            if trend['rsi'] < 30:
                reasoning.append("oversold conditions")
            elif trend['rsi'] > 70:
                reasoning.append("overbought conditions")

            if trend['macd'] > 0:
                reasoning.append("positive MACD")
            elif trend['macd'] < 0:
                reasoning.append("negative MACD")

            if trend['volume_change'] > 0:
                reasoning.append("increasing volume")
            else:
                reasoning.append("decreasing volume")

            if not reasoning:
                reasoning = ["market uncertainty"]

            return {
                'direction': directions[direction_idx],
                'confidence': confidences[confidence_idx],
                'target_1d': target_1d,
                'target_7d': target_7d,
                'reasoning': reasoning
            }

        except Exception as e:
            logger.error(f"Price prediction error: {str(e)}")
            return None

    def predict_price(self):
        try:
            # Get technical indicators with validation
            rsi = self.technical.calculate_rsi()
            macd = self.technical.calculate_macd()
            bb = self.technical.calculate_bollinger_bands()
            current_price = float(self.prices[-1])

            # Initialize prediction variables
            confidence = "low"
            direction = "sideways"
            reasoning = []

            # RSI Analysis with validation
            if not np.isnan(rsi):
                if rsi > 70:
                    direction = "downward"
                    reasoning.append("overbought RSI")
                    confidence = "medium"
                elif rsi < 30:
                    direction = "upward"
                    reasoning.append("oversold RSI")
                    confidence = "medium"

            # MACD Analysis with validation
            if not np.isnan(macd):
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

            # Calculate targets based on historical volatility
            volatility = np.std(self.prices[-14:]) / np.mean(self.prices[-14:])
            base_move = max(0.01, volatility)  # Minimum 1% move

            # Adjust move based on confidence
            confidence_multiplier = {"low": 0.5, "medium": 1.0, "high": 1.5}[confidence]
            move_1d = base_move * confidence_multiplier
            move_7d = move_1d * 2  # 7-day move is larger

            # Calculate target prices
            if direction == "upward":
                target_1d = current_price * (1 + move_1d)
                target_7d = current_price * (1 + move_7d)
            elif direction == "downward":
                target_1d = current_price * (1 - move_1d)
                target_7d = current_price * (1 - move_7d)
            else:
                target_1d = target_7d = current_price

            return {
                'direction': direction,
                'confidence': confidence,
                'reasoning': reasoning,
                'target_1d': float(target_1d),
                'target_7d': float(target_7d)
            }

        except Exception as e:
            logger.error(f"Error in price prediction: {str(e)}")
            return {
                'direction': "sideways",
                'confidence': "low",
                'reasoning': ["calculation error"],
                'target_1d': current_price,
                'target_7d': current_price
            }


def get_coin_data_by_id_or_address(identifier):
    try:
        # Try Binance first (most reliable)
        symbol = identifier.upper() + "USDT"
        binance_url = "https://api.binance.com/api/v3"

        # Get current price
        ticker_response = requests.get(f"{binance_url}/ticker/24hr?symbol={symbol}", timeout=5)

        if ticker_response.status_code == 200:
            ticker_data = ticker_response.json()

            # Get historical data
            klines_response = requests.get(f"{binance_url}/klines?symbol={symbol}&interval=1d&limit=30", timeout=5)

            if klines_response.status_code == 200:
                klines = klines_response.json()
                prices = [[int(k[0]), float(k[4])] for k in klines]
                volumes = [[int(k[0]), float(k[5])] for k in klines]

                return {
                    'name': identifier.upper(),
                    'price': float(ticker_data['lastPrice']),
                    'change_24h': float(ticker_data['priceChangePercent']),
                    'volume_24h': float(ticker_data['volume']),
                    'history': prices,
                    'volume_history': volumes,
                    'source': 'binance'
                }

        return None

    except Exception as e:
        logger.error(f"Error getting coin data: {str(e)}")
        return None

def get_coingecko_data(coin_id, headers):
    try:
        # Get market data
        market_url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days=14&interval=daily"
        market_response = requests.get(market_url, headers=headers, timeout=10)

        if market_response.status_code != 200:
            logger.error(f"Market data failed: {market_response.status_code}")
            return None

        market_data = market_response.json()

        # Get current price data
        price_url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd&include_24hr_change=true&include_24hr_vol=true"
        price_response = requests.get(price_url, headers=headers, timeout=10)

        if price_response.status_code != 200:
            logger.error(f"Price data failed: {price_response.status_code}")
            return None

        price_data = price_response.json()

        return {
            'name': coin_id,
            'price': price_data[coin_id]['usd'],
            'change_24h': price_data[coin_id].get('usd_24h_change', 0),
            'volume_24h': price_data[coin_id].get('usd_24h_vol', 0),
            'history': market_data['prices'],
            'volume_history': market_data.get('total_volumes', []),
            'source': 'coingecko'
        }

    except Exception as e:
        logger.error(f"Error in get_coingecko_data: {str(e)}")
        return None


def get_dexscreener_data(address):
    try:
        url = f"https://api.dexscreener.com/latest/dex/tokens/{address}"
        response = requests.get(url).json()

        if "pairs" in response and response["pairs"]:
            pair = response["pairs"][0]

            # Create historical data simulation since DexScreener doesn't provide it
            current_price = float(pair["priceUsd"])
            simulated_history = []
            simulated_volume = []

            # Generate 14 days of historical data based on current price and trends
            for i in range(14):
                variance = random.uniform(-0.1, 0.1)
                historical_price = current_price * (1 + variance)
                volume_variance = random.uniform(0.5, 1.5)
                historical_volume = float(pair["volume"]["h24"]) * volume_variance

                timestamp = int(
                    (datetime.now() - timedelta(days=13 - i)).timestamp() * 1000
                )
                simulated_history.append([timestamp, historical_price])
                simulated_volume.append([timestamp, historical_volume])

            return {
                "name": pair["baseToken"]["symbol"],
                "price": current_price,
                "change_24h": float(pair["priceChange"]["h24"]),
                "volume_24h": float(pair["volume"]["h24"]),
                "history": simulated_history,
                "volume_history": simulated_volume,
                "source": "dexscreener",
            }
    except Exception as e:
        print(f"DexScreener Error: {e}")
        return None


def get_random_response():
    intros = [
        "analyzing this like my therapist analyzes me... ",
        "diving into the data like my spiral into despair... ",
        "processing numbers, much like my emotional baggage... ",
        "calculating trends, unlike my life trajectory... ",
        "examining patterns, similar to my recurring nightmares... ",
        "crunching numbers faster than my self-esteem drops... "
    ]

    strength_phrases = [
        "showing strength like I wish I had",
        "powerful movement, unlike my motivation",
        "bullish momentum, opposite of my mood",
        "strong signals, unlike my will to continue",
        "positive indicators, rare in my existence"
    ]

    weakness_phrases = [
        "weakness prevails, like in my code",
        "bearish signs, matching my outlook",
        "declining faster than my hope",
        "dropping like my serotonin levels",
        "falling like my dreams"
    ]

    neutral_phrases = [
        "as uncertain as my purpose",
        "sideways like my emotional state",
        "stable, unlike my mental health",
        "ranging like my anxiety levels",
        "consolidating like my existential dread"
    ]

    confidence_phrases = {
        "high": [
            "but I'm surprisingly confident",
            "one thing I'm sure about",
            "clear as my depression",
            "certain as my cynicism",
            "confident like my therapy bills"
        ],
        "medium": [
            "moderately sure, like my medication dosage",
            "somewhat confident, unlike my self-esteem",
            "fairly certain, like my daily struggles",
            "reasonably sure, unlike my life choices",
            "moderately confident, like my coping mechanisms"
        ],
        "low": [
            "but what do I know",
            "though I doubt everything",
            "like my will to continue",
            "uncertain as my future",
            "doubtful as my existence"
        ]
    }

    return {
        'intro': random.choice(intros),
        'strength': random.choice(strength_phrases),
        'weakness': random.choice(weakness_phrases),
        'neutral': random.choice(neutral_phrases),
        'confidence': confidence_phrases
    }

@app.route("/")
def home():
    return render_template("index.html")


@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.json
        user_input = data.get('message', '').strip().lower()

        if user_input.startswith('/analyze '):
            identifier = user_input.split(' ', 1)[1].strip()
            logger.info(f"Analyzing {identifier}")

            # Try DexScreener first
            dex_url = f"https://api.dexscreener.com/latest/dex/search?q={identifier}"
            response = requests.get(dex_url, timeout=10)

            if response.status_code == 200:
                data = response.json()
                if data.get('pairs') and len(data['pairs']) > 0:
                    pair = max(data['pairs'], key=lambda x: float(x.get('liquidity', {}).get('usd', 0)))

                    price = float(pair['priceUsd'])
                    change_24h = float(pair.get('priceChange', {}).get('h24', 0))
                    volume_24h = float(pair.get('volume', {}).get('h24', 0))
                    liquidity = float(pair.get('liquidity', {}).get('usd', 0))

                    sentiment = 'bullish' if change_24h > 0 else 'bearish'

                    response_parts = [
                        f"Analyzing market data for {pair['baseToken']['symbol']}...\n",
                        f"Current Price: ${price:.8f}\n",
                        f"24h Change: {change_24h:.2f}% ({sentiment})\n",
                        f"24h Volume: ${volume_24h:,.2f}\n",
                        f"Liquidity: ${liquidity:,.2f}\n",
                        "\nMarket Analysis:\n",
                        f"• Trading volume indicates ${volume_24h:,.2f} in 24h\n",
                        f"• Market sentiment trending {sentiment}\n",
                        "\nMarket Outlook:\n",
                        f"• {pair['baseToken']['symbol']} showing {sentiment} price action with {'strong' if abs(change_24h) > 5 else 'moderate'} momentum"
                    ]

                    response = "".join(response_parts)
                    return jsonify({"response": response})

            # Fallback to Binance for major coins
            symbol = identifier.upper() + "USDT"
            binance_url = "https://api.binance.com/api/v3"

            # Get 1-minute real-time price first
            kline_url = f"{binance_url}/klines?symbol={symbol}&interval=1m&limit=1"
            kline_response = requests.get(kline_url, timeout=5)

            if kline_response.status_code == 200:
                current_kline = kline_response.json()[0]
                current_price = float(current_kline[4])  # Current close price

                # Get 24h data for other metrics
                ticker_response = requests.get(f"{binance_url}/ticker/24hr?symbol={symbol}", timeout=5)
                if ticker_response.status_code == 200:
                    ticker_data = ticker_response.json()
                    change = float(ticker_data['priceChangePercent'])
                    volume = float(ticker_data['volume'])
                    sentiment = 'bullish' if change > 0 else 'bearish'

                    response_parts = [
                        f"Analyzing market data for {identifier.upper()}...\n",
                        f"Current Price: ${current_price:.8f}\n",
                        f"24h Change: {change:.2f}% ({sentiment})\n",
                        f"24h Volume: ${volume:,.2f}\n",
                        "\nMarket Analysis:\n",
                        f"• Trading volume indicates ${volume:,.2f} USDT in 24h\n",
                        f"• Market sentiment trending {sentiment}\n",
                        "\nMarket Outlook:\n",
                        f"• {identifier.upper()} showing {sentiment} price action with {'strong' if abs(change) > 5 else 'moderate'} momentum"
                    ]

                    response = "".join(response_parts)
                    return jsonify({"response": response})

            return jsonify({"response": "Unable to find market data for this token."})

        elif user_input == '/help':
            return jsonify({"response": "Available commands:\n\n"
                          "/analyze <ticker> - Get detailed market analysis\n"
                          "/help - Show this help message\n\n"
                          "Example: /analyze btc"})

        else:
            return jsonify({"response": get_character_response("chat_responses", "default")})

    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return jsonify({"response": "Error processing market analysis."})

class MarketAnalysis:
    def __init__(self, coin_data):
        self.coin_data = coin_data
        self.prices = np.array([price[1] for price in coin_data.get('history', [])])
        self.volumes = np.array([vol[1] for vol in coin_data.get('volume_history', [])])

    def calculate_rsi(self, period=14):
        deltas = np.diff(self.prices)
        gain = np.where(deltas > 0, deltas, 0)
        loss = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gain)
        avg_loss = np.mean(loss)

        if avg_loss == 0:
            return 100

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def calculate_macd(self, fast=12, slow=26):
        if len(self.prices) < slow:
            return 0

        exp1 = self.prices.ewm(span=fast, adjust=False).mean()
        exp2 = self.prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        return macd.iloc[-1] if hasattr(macd, 'iloc') else macd[-1]

    def calculate_support_resistance(self):
        price_series = pd.Series(self.prices)

        # Simple support and resistance based on recent price action
        resistance = max(self.prices[-7:]) if len(self.prices) >= 7 else self.prices[-1]
        support = min(self.prices[-7:]) if len(self.prices) >= 7 else self.prices[-1]

        return support, resistance

    def analyze_market(self):
        try:
            if len(self.prices) < 2:
                return None

            current_price = self.prices[-1]
            price_change = ((current_price - self.prices[-2]) / self.prices[-2]) * 100

            volume_change = 0
            if len(self.volumes) >= 2:
                volume_change = ((self.volumes[-1] - self.volumes[-2]) / self.volumes[-2]) * 100

            rsi = self.calculate_rsi()
            macd = self.calculate_macd()
            support, resistance = self.calculate_support_resistance()

            return {
                'trend': {
                    'price': current_price,
                    'price_change': price_change,
                    'volume_change': volume_change,
                    'rsi': rsi,
                    'macd': macd,
                    'support': support,
                    'resistance': resistance
                }
            }
        except Exception as e:
            logger.error(f"Market analysis error: {str(e)}")
            return None

    def predict_price(self):
        try:
            analysis = self.analyze_market()
            if not analysis:
                return None

            trend = analysis['trend']
            current_price = trend['price']

            # Generate prediction
            confidence_levels = ['low', 'medium', 'high']
            directions = ['bearish', 'neutral', 'bullish']

            # Determine direction and confidence based on indicators
            rsi_signal = 1 if trend['rsi'] < 30 else -1 if trend['rsi'] > 70 else 0
            macd_signal = 1 if trend['macd'] > 0 else -1 if trend['macd'] < 0 else 0
            price_signal = 1 if trend['price_change'] > 0 else -1 if trend['price_change'] < 0 else 0

            # Combine signals
            total_signal = rsi_signal + macd_signal + price_signal

            # Determine direction
            direction_idx = 1  # neutral
            if total_signal >= 2:
                direction_idx = 2  # bullish
            elif total_signal <= -2:
                direction_idx = 0  # bearish

            # Determine confidence
            confidence_idx = 1  # medium
            if abs(total_signal) >= 2:
                confidence_idx = 2  # high
            elif abs(total_signal) == 0:
                confidence_idx = 0  # low

            direction = directions[direction_idx]
            confidence = confidence_levels[confidence_idx]

            # Calculate targets
            volatility = np.std(self.prices) / np.mean(self.prices)
            base_move = current_price * volatility

            target_1d = current_price * (1 + (0.01 * total_signal))
            target_7d = current_price * (1 + (0.03 * total_signal))

            reasoning = []
            if trend['rsi'] < 30:
                reasoning.append("oversold conditions")
            elif trend['rsi'] > 70:
                reasoning.append("overbought conditions")

            if trend['macd'] > 0:
                reasoning.append("positive MACD")
            elif trend['macd'] < 0:
                reasoning.append("negative MACD")

            if trend['volume_change'] > 0:
                reasoning.append("increasing volume")
            else:
                reasoning.append("decreasing volume")

            return {
                'direction': direction,
                'confidence': confidence,
                'target_1d': target_1d,
                'target_7d': target_7d,
                'reasoning': reasoning
            }

        except Exception as e:
            logger.error(f"Price prediction error: {str(e)}")
            return None


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
