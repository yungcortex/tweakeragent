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
    def __init__(self, data):
        try:
            # Validate input data
            if not isinstance(data, dict):
                raise ValueError("Data must be a dictionary")
            required_fields = ['name', 'price', 'history', 'volume_history']
            if not all(field in data for field in required_fields):
                raise ValueError(f"Missing required fields: {required_fields}")

            self.data = data

            # Convert price history to numpy arrays with validation
            self.prices = np.array([float(price[1]) for price in data['history'] if price[1] is not None])
            self.volumes = np.array([float(vol[1]) for vol in data.get('volume_history', []) if vol[1] is not None])

            # Ensure minimum data length
            if len(self.prices) < 14:
                raise ValueError("Insufficient price history")

            # Initialize technical analysis with validated data
            self.technical = TechnicalAnalysis(self.prices, self.volumes)

        except Exception as e:
            logger.error(f"Error initializing MarketAnalysis: {str(e)}")
            raise

    def analyze_market(self):
        try:
            # Get current price and calculate changes
            current_price = float(self.data['price'])
            price_change = float(self.data.get('change_24h', 0))

            # Calculate volume change
            if len(self.volumes) >= 2:
                volume_change = ((self.volumes[-1] - self.volumes[-2]) / self.volumes[-2] * 100
                               if self.volumes[-2] != 0 else 0)
            else:
                volume_change = 0

            # Calculate technical indicators with error handling
            try:
                rsi = self.technical.calculate_rsi()
                macd = self.technical.calculate_macd()
                levels = self.technical.get_support_resistance()
            except Exception as e:
                logger.error(f"Error calculating indicators: {str(e)}")
                rsi = 50
                macd = 0
                levels = {'support': [], 'resistance': []}

            return {
                'trend': {
                    'price_change': price_change,
                    'volume_change': volume_change,
                    'rsi': rsi,
                    'macd': macd,
                    'support': levels['support'][0] if levels['support'] else current_price * 0.95,
                    'resistance': levels['resistance'][0] if levels['resistance'] else current_price * 1.05,
                    'price': current_price,
                    'patterns': []
                }
            }

        except Exception as e:
            logger.error(f"Error in market analysis: {str(e)}")
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
        logger.info(f"Getting data for {identifier}")

        # Add retry mechanism
        max_retries = 3
        retry_delay = 1  # seconds

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json'
        }

        # Try multiple data sources in sequence
        for attempt in range(max_retries):
            try:
                # 1. Try CoinGecko direct mapping
                coin_mapping = {
                    'btc': 'bitcoin',
                    'eth': 'ethereum',
                    'bnb': 'binancecoin',
                    'sol': 'solana',
                    'xrp': 'ripple',
                    'doge': 'dogecoin',
                    'ada': 'cardano',
                    'dot': 'polkadot',
                    'matic': 'polygon',
                    'link': 'chainlink'
                }

                coin_id = coin_mapping.get(identifier.lower())
                if coin_id:
                    data = get_coingecko_data(coin_id, headers)
                    if data:
                        return data

                # 2. Try CoinGecko search with rate limit handling
                search_url = f"https://api.coingecko.com/api/v3/search?query={identifier}"
                search_response = requests.get(search_url, headers=headers, timeout=10)

                if search_response.status_code == 429:  # Rate limit hit
                    logger.warning("Rate limit hit, waiting before retry...")
                    time.sleep(retry_delay * (attempt + 1))
                    continue

                if search_response.status_code == 200:
                    search_result = search_response.json()
                    if search_result.get('coins'):
                        coin_id = search_result['coins'][0]['id']
                        data = get_coingecko_data(coin_id, headers)
                        if data:
                            return data

                # 3. Try DexScreener for contract addresses
                if len(identifier) > 30:
                    data = get_dexscreener_data(identifier)
                    if data:
                        return data

                # 4. Try alternative API (Binance)
                try:
                    binance_symbol = identifier.upper()
                    if not binance_symbol.endswith('USDT'):
                        binance_symbol += 'USDT'
                    binance_url = f"https://api.binance.com/api/v3/klines?symbol={binance_symbol}&interval=1d&limit=14"
                    binance_response = requests.get(binance_url, timeout=10)

                    if binance_response.status_code == 200:
                        klines = binance_response.json()
                        if len(klines) > 0:
                            prices = [[int(k[0]), float(k[4])] for k in klines]  # Use closing prices
                            volumes = [[int(k[0]), float(k[5])] for k in klines]
                            current_price = float(klines[-1][4])

                            return {
                                'name': identifier.upper(),
                                'price': current_price,
                                'change_24h': ((current_price - float(klines[-2][4])) / float(klines[-2][4])) * 100,
                                'volume_24h': float(klines[-1][5]),
                                'history': prices,
                                'volume_history': volumes,
                                'source': 'binance'
                            }
                except Exception as e:
                    logger.error(f"Binance API error: {str(e)}")

            except requests.exceptions.RequestException as e:
                logger.error(f"Request error (attempt {attempt + 1}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                continue

        logger.error("All data source attempts failed")
        return None

    except Exception as e:
        logger.error(f"Error in get_coin_data_by_id_or_address: {str(e)}")
        return None

def get_coingecko_data(coin_id, headers):
    try:
        # Add exponential backoff
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Check CoinGecko API status first
                status_url = "https://api.coingecko.com/api/v3/ping"
                status_response = requests.get(status_url, headers=headers, timeout=5)

                if status_response.status_code != 200:
                    logger.warning(f"CoinGecko API might be down: {status_response.status_code}")
                    time.sleep(1 * (attempt + 1))
                    continue

                # Get market data
                market_url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days=14&interval=daily"
                market_response = requests.get(market_url, headers=headers, timeout=10)

                if market_response.status_code == 429:  # Rate limit
                    logger.warning("Rate limit hit, waiting...")
                    time.sleep(1 * (attempt + 1))
                    continue

                if market_response.status_code != 200:
                    logger.error(f"Market data failed: {market_response.status_code}")
                    continue

                market_data = market_response.json()

                # Validate market data
                if not market_data.get('prices') or len(market_data['prices']) < 2:
                    logger.error("Insufficient market data")
                    continue

                # Get current price data with retry
                price_url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd&include_24hr_change=true&include_24hr_vol=true"
                price_response = requests.get(price_url, headers=headers, timeout=10)

                if price_response.status_code != 200:
                    logger.error(f"Price data failed: {price_response.status_code}")
                    continue

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

            except requests.exceptions.RequestException as e:
                logger.error(f"Request error (attempt {attempt + 1}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(1 * (attempt + 1))
                continue

        return None

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


@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.json
        user_input = data.get('message', '').strip().lower()

        if not user_input:
            return jsonify({"response": "i may be depressed, but i still need input"})

        if user_input.startswith('/analyze ') or user_input.startswith('prediction for '):
            # Extract coin identifier
            identifier = user_input.split(' ', 1)[1].strip()

            try:
                # Get coin data
                coin_data = get_coin_data_by_id_or_address(identifier)
                if not coin_data:
                    return jsonify({"response": "couldn't find that coin... like my will to live"})

                # Perform analysis
                analysis = MarketAnalysis(coin_data)

                if user_input.startswith('/analyze'):
                    result = analysis.analyze_market()
                    prediction = analysis.predict_price()

                    if result and prediction:
                        trend = result['trend']
                        responses = get_random_response()

                        response = (
                            f"{responses['intro']}{coin_data['name']} at ${trend['price']:.8f}. "
                            f"rsi at {trend['rsi']:.4f} suggesting {responses['weakness'] if trend['rsi'] > 70 else responses['strength'] if trend['rsi'] < 30 else responses['neutral']}, "
                            f"technical analysis shows rsi at {trend['rsi']:.4f} and macd at {trend['macd']:.4f}, "
                            f"like my deteriorating mental state. support at ${trend['support']:.8f} and resistance at ${trend['resistance']:.8f} "
                            f"{'bearish' if trend['price_change'] < 0 else 'bullish'} move but volume's lighter than my wallet... "
                            f"{trend['volume_change']:.1f}% {'decrease' if trend['volume_change'] < 0 else 'increase'} "
                            f"suggests a {prediction['direction']} move with {prediction['confidence']} confidence... "
                            f"24h target: ${prediction['target_1d']:.8f}, 7d target: ${prediction['target_7d']:.8f}... "
                            f"based on {', '.join(prediction['reasoning'])}, {random.choice(responses['confidence'][prediction['confidence']])}"
                        )
                    else:
                        response = "analysis failed... like everything else in my life"
                else:
                    prediction = analysis.predict_price()
                    if prediction:
                        response = (
                            f"prediction for {coin_data['name']}: "
                            f"{prediction['direction']} move with {prediction['confidence']} confidence... "
                            f"24h target: ${prediction['target_1d']:.8f}, 7d target: ${prediction['target_7d']:.8f}... "
                            f"based on {', '.join(prediction['reasoning'])}, but what do i know, i'm just a sad bot"
                        )
                    else:
                        response = "prediction failed... just like my dreams"

                return jsonify({"response": response})

            except Exception as e:
                logger.error(f"Analysis error: {str(e)}")
                return jsonify({"response": f"analysis failed... like my life. error: {str(e)}"})

        elif user_input == '/help':
            return jsonify({"response": """
            helping others... how meaningless. but here:
            /analyze <ticker/address> - analyze any token with price prediction
            prediction for <ticker/address> - get detailed price prediction
            /help - you're looking at it, unfortunately

            example: /analyze btc or prediction for eth

            or just chat with me if you're feeling particularly masochistic.
            """})
        else:
            return jsonify({"response": "i'm too depressed to understand that command"})

    except Exception as e:
        logger.error(f"Route error: {str(e)}")
        return jsonify({"response": f"something went wrong... like everything else. error: {str(e)}"})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
