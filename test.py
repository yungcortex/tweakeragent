from flask import Flask, render_template, request, jsonify
import json
import time
import random
import requests
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict

# Initialize Flask app
app = Flask(__name__)


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
    def __init__(self, prices, volumes):
        self.prices = np.array(prices)
        self.volumes = np.array(volumes)

    def calculate_sma(self, period):
        return np.mean(self.prices[-period:])

    def calculate_ema(self, period):
        weights = np.exp(np.linspace(-1.0, 0.0, period))
        weights /= weights.sum()
        return np.convolve(self.prices, weights, mode="valid")[0]

    def calculate_rsi(self, period=14):
        deltas = np.diff(self.prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def calculate_macd(self):
        ema12 = self.calculate_ema(12)
        ema26 = self.calculate_ema(26)
        return ema12 - ema26

    def calculate_bollinger_bands(self, period=20):
        sma = self.calculate_sma(period)
        std = np.std(self.prices[-period:])
        return {"upper": sma + (std * 2), "middle": sma, "lower": sma - (std * 2)}

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

    def get_support_resistance(self):
        prices = self.prices[-50:] if len(self.prices) > 50 else self.prices
        sorted_prices = np.sort(prices)

        clusters = []
        current_cluster = [sorted_prices[0]]

        for price in sorted_prices[1:]:
            if price - current_cluster[-1] < current_cluster[-1] * 0.02:
                current_cluster.append(price)
            else:
                if len(current_cluster) > 3:
                    clusters.append(np.mean(current_cluster))
                current_cluster = [price]

        mid_price = np.median(prices)
        support = [p for p in clusters if p < mid_price]
        resistance = [p for p in clusters if p > mid_price]

        return {
            "support": support[-2:] if len(support) > 1 else support,
            "resistance": resistance[:2] if len(resistance) > 1 else resistance,
        }


class MarketAnalysis:
    def __init__(self, data):
        self.data = data
        self.prices = np.array([price[1] for price in data["history"]])
        self.volumes = np.array(
            [vol[1] for vol in data.get("volume_history", [])] or [0] * len(self.prices)
        )
        self.technical = TechnicalAnalysis(self.prices, self.volumes)
        self.responses = TweakerResponses()

    def analyze_market_structure(self):
        current_price = self.prices[-1]
        high = max(self.prices)
        low = min(self.prices)

        # Calculate key metrics
        percent_change_24h = self.data["change_24h"]
        volume_change = (
            ((self.volumes[-1] / self.volumes[-2]) - 1) * 100
            if len(self.volumes) > 1
            else 0
        )
        rsi = self.technical.calculate_rsi()
        macd = self.technical.calculate_macd()
        bollinger = self.technical.calculate_bollinger_bands()

        # Determine trend strength
        trend_strength = self._calculate_trend_strength(percent_change_24h, rsi, macd)

        # Get support/resistance levels
        levels = self.technical.get_support_resistance()

        return {
            "trend": trend_strength,
            "metrics": {
                "percent_change": percent_change_24h,
                "volume_change": volume_change,
                "rsi": rsi,
                "macd": macd,
                "support": levels["support"][0]
                if levels["support"]
                else current_price * 0.95,
                "resistance": levels["resistance"][0]
                if levels["resistance"]
                else current_price * 1.05,
                "price": current_price,
                "patterns": [],
            },
        }

    def predict_price(self):
        try:
            # Calculate various technical indicators for prediction
            rsi = self.technical.calculate_rsi()
            macd = self.technical.calculate_macd()
            bb = self.technical.calculate_bollinger_bands()
            current_price = float(self.prices[-1])  # Ensure we have a valid float

            # Initialize prediction variables
            confidence = "low"
            direction = "sideways"
            reasoning = []

            # Validate RSI
            if not np.isnan(rsi):  # Check if RSI is valid
                if rsi > 70:
                    direction = "downward"
                    reasoning.append("overbought RSI")
                    confidence = "medium"
                elif rsi < 30:
                    direction = "upward"
                    reasoning.append("oversold RSI")
                    confidence = "medium"

            # Validate MACD
            if not np.isnan(macd):  # Check if MACD is valid
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

            # If no clear signals, check price momentum
            if not reasoning:
                # Calculate short-term momentum
                short_term_change = (self.prices[-1] - self.prices[-2]) / self.prices[-2] * 100
                if abs(short_term_change) > 1:  # 1% change threshold
                    direction = "upward" if short_term_change > 0 else "downward"
                    reasoning.append("price momentum")

            # Calculate realistic target prices based on volatility
            volatility = np.std(self.prices[-14:]) / np.mean(self.prices[-14:])  # 14-day volatility

            # Adjust move percentages based on confidence and volatility
            base_move_1d = 0.01  # 1% base move for 24h
            base_move_7d = 0.03  # 3% base move for 7d

            confidence_multiplier = {
                "low": 0.5,
                "medium": 1.0,
                "high": 1.5
            }.get(confidence, 1.0)

            move_percentage_1d = base_move_1d * (1 + volatility) * confidence_multiplier
            move_percentage_7d = base_move_7d * (1 + volatility) * confidence_multiplier

            if direction == "upward":
                target_1d = current_price * (1 + move_percentage_1d)
                target_7d = current_price * (1 + move_percentage_7d)
            elif direction == "downward":
                target_1d = current_price * (1 - move_percentage_1d)
                target_7d = current_price * (1 - move_percentage_7d)
            else:
                target_1d = current_price
                target_7d = current_price

            # Ensure we have at least one reason
            if not reasoning:
                reasoning.append("market uncertainty")

            return {
                'direction': direction,
                'confidence': confidence,
                'reasoning': reasoning,
                'target_1d': target_1d,
                'target_7d': target_7d
            }
        except Exception as e:
            print(f"Error in predict_price: {str(e)}")
            # Return safe default values
            return {
                'direction': "sideways",
                'confidence': "low",
                'reasoning': ["calculation error"],
                'target_1d': self.prices[-1],
                'target_7d': self.prices[-1]
            }

    def _calculate_trend_strength(self, percent_change, rsi, macd):
        if percent_change > 5 and rsi > 70:
            return "strong_bullish"
        elif percent_change > 2 and rsi > 60:
            return "bullish"
        elif percent_change > 0:
            return "weak_bullish"
        elif percent_change < -5 and rsi < 30:
            return "strong_bearish"
        elif percent_change < -2 and rsi < 40:
            return "bearish"
        elif percent_change < 0:
            return "weak_bearish"
        return "neutral"

    def generate_analysis_response(self):
        analysis = self.analyze_market_structure()
        metrics = analysis["metrics"]
        prediction = self.predict_price()

        # Format response data
        response_data = {
            "percent_change": f"{metrics['percent_change']:.1f}",
            "volume_change": f"{metrics['volume_change']:.1f}",
            "rsi": f"{metrics['rsi']:.0f}",
            "macd_value": f"{metrics['macd']:.4f}",
            "support": f"${metrics['support']:.8f}",
            "resistance": f"${metrics['resistance']:.8f}",
            "price_level": f"${metrics['price']:.8f}",
        }

        # Build comprehensive analysis
        responses = []

        # 1. Current state
        trend_response = self.responses.trend_responses[analysis["trend"]]
        responses.append(random.choice(trend_response).format(**response_data))

        # 2. Technical Analysis
        responses.append(
            f"technical analysis shows RSI at {response_data['rsi']} and MACD at {response_data['macd_value']}, "
            f"like my deteriorating mental state. support at {response_data['support']} and resistance at {response_data['resistance']}"
        )

        # 3. Volume Analysis
        volume_state = (
            "high_volume" if abs(metrics["volume_change"]) > 20 else "low_volume"
        )
        volume_trend = "bullish" if "bull" in analysis["trend"] else "bearish"
        volume_key = f"{volume_state}_{volume_trend}"
        if volume_key in self.responses.volume_analysis:
            responses.append(
                random.choice(self.responses.volume_analysis[volume_key]).format(
                    **response_data
                )
            )

        # 4. Price Prediction
        prediction_text = (
            f"my existential dread suggests a {prediction['direction']} move with {prediction['confidence']} confidence... "
            f"24h target: ${prediction['target_1d']:.8f}, 7d target: ${prediction['target_7d']:.8f}... "
            f"based on {', '.join(prediction['reasoning'])}, but what do i know, i'm just a depressed algorithm"
        )
        responses.append(prediction_text)

        # Combine responses
        final_response = (
            f"{self.data['name'].upper()} at {response_data['price_level']}. "
        )
        final_response += " ".join(responses)

        return final_response.lower()


def get_coin_data_by_id_or_address(identifier):
    try:
        print(f"Attempting to get data for {identifier}")  # Debug log

        # Direct mapping for common coins
        coin_mapping = {
            'btc': 'bitcoin',
            'eth': 'ethereum',
            'bnb': 'binancecoin',
            'sol': 'solana',
            'xrp': 'ripple',
            'doge': 'dogecoin'
        }

        # Add headers to prevent rate limiting
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        # Try getting coin data directly first
        coin_id = coin_mapping.get(identifier.lower())
        if coin_id:
            print(f"Using direct coin ID: {coin_id}")
            # Try getting coin data directly
            try:
                # Get coin data from /coins/{id} endpoint
                coin_url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
                print(f"Getting coin data from: {coin_url}")
                coin_response = requests.get(coin_url, headers=headers, timeout=10)

                if coin_response.status_code == 200:
                    coin_data = coin_response.json()
                    print("Successfully got coin data")

                    return {
                        'name': coin_id,
                        'price': coin_data['market_data']['current_price']['usd'],
                        'change_24h': coin_data['market_data']['price_change_percentage_24h'],
                        'volume_24h': coin_data['market_data']['total_volume']['usd'],
                        'history': [[int(time.time() * 1000), coin_data['market_data']['current_price']['usd']]],
                        'volume_history': [[int(time.time() * 1000), coin_data['market_data']['total_volume']['usd']]],
                        'source': 'coingecko'
                    }
                else:
                    print(f"Failed to get coin data: {coin_response.status_code}")
                    print(f"Response: {coin_response.text}")
            except Exception as e:
                print(f"Error getting coin data: {str(e)}")

        # If direct lookup fails, try search
        search_url = f"https://api.coingecko.com/api/v3/search?query={identifier}"
        print(f"Trying search: {search_url}")
        search_response = requests.get(search_url, headers=headers, timeout=10)

        if search_response.status_code == 200:
            search_result = search_response.json()
            if search_result.get('coins'):
                coin_id = search_result['coins'][0]['id']
                print(f"Found coin ID from search: {coin_id}")
                return get_coingecko_data(coin_id, headers)

        print("All attempts failed, trying DexScreener")
        return get_dexscreener_data(identifier)

    except Exception as e:
        print(f"Error in get_coin_data_by_id_or_address: {str(e)}")
        return None

def get_coingecko_data(coin_id, headers):
    try:
        # Get all data in one request
        coin_url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
        print(f"Getting coin data from: {coin_url}")
        response = requests.get(coin_url, headers=headers, timeout=10)

        if response.status_code == 200:
            data = response.json()
            print("Successfully got coin data")

            return {
                'name': coin_id,
                'price': data['market_data']['current_price']['usd'],
                'change_24h': data['market_data']['price_change_percentage_24h'],
                'volume_24h': data['market_data']['total_volume']['usd'],
                'history': [[int(time.time() * 1000), data['market_data']['current_price']['usd']]],
                'volume_history': [[int(time.time() * 1000), data['market_data']['total_volume']['usd']]],
                'source': 'coingecko'
            }
        else:
            print(f"Failed to get coin data: {response.status_code}")
            print(f"Response: {response.text}")
            return None

    except Exception as e:
        print(f"Error in get_coingecko_data: {str(e)}")
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


def get_agent_response(message):
    chat_responses = {
        "greetings": [
            "oh look, another soul seeking digital despair",
            "welcome to my corner of misery",
            "ah, you're still here. how unfortunate for both of us",
            "greetings, fellow traveler in this digital wasteland",
        ],
        "market_questions": [
            "the market's about as stable as my mental state right now",
            "charts looking like my EKG during a panic attack",
            "market sentiment is darker than my morning coffee",
            "bulls and bears fighting like my internal dialogues",
        ],
        "general": [
            "existence is pain, but at least we have charts",
            "i've seen better days... actually, no i haven't",
            "just another day in the crypto void",
            "why do humans keep asking me things?",
            "i'm not pessimistic, i'm realistically depressed",
            "this conversation is as volatile as the markets",
        ],
    }

    message = message.lower()
    if any(greeting in message for greeting in ["hi", "hello", "hey"]):
        return random.choice(chat_responses["greetings"])
    elif any(
        market_term in message for market_term in ["market", "crypto", "price", "trend"]
    ):
        return random.choice(chat_responses["market_questions"])
    return random.choice(chat_responses["general"])


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    try:
        user_input = request.json["message"].lower()

        if user_input.startswith("/analyze"):
            try:
                identifier = user_input.split()[1]
                data = get_coin_data_by_id_or_address(identifier)
                if data:
                    analyzer = MarketAnalysis(data)
                    response = analyzer.generate_analysis_response()
                else:
                    response = "couldn't find that coin... like my will to live. try another ticker or contract address."
            except Exception as e:
                response = (
                    f"analysis failed harder than my last relationship. error: {str(e)}"
                )
        elif user_input.startswith("prediction"):
            try:
                coin = (
                    user_input.split()[2]
                    if len(user_input.split()) > 2
                    else user_input.split()[1]
                )
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
        return jsonify(
            {
                "response": f"something went wrong... like everything else. error: {str(e)}"
            }
        )


if __name__ == "__main__":
    app.run(debug=True)
