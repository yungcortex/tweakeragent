from flask import Flask, Blueprint, request, jsonify, render_template
import os
import json
import time
import numpy as np
import pandas as pd
import requests
import logging
from datetime import datetime, timedelta
import random
import aiohttp
import asyncio
from web3 import Web3
from typing import Dict, Any, Optional, List
import re
from ta import momentum, trend, volatility
from functools import wraps
from os import getenv
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app with the correct template folder path
app = Flask(__name__, 
           template_folder='../templates',
           static_folder='../static')

# Create blueprint instead of Flask app
bp = Blueprint('character', __name__)

# Get the absolute path to the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load character configuration
character_path = os.path.join(os.path.dirname(__file__), 'tweaker.character.json')
with open(character_path, 'r') as f:
    CHARACTER = json.load(f)

# Load environment variables
load_dotenv()

# API endpoints and configurations
APIS = {
    'pump_fun': {
        'url': "https://api.pump.fun/api/v1",
        'timeout': 10
    },
    'dexscreener': {
        'url': "https://api.dexscreener.com/latest",
        'timeout': 10
    },
    'coingecko': {
        'url': "https://api.coingecko.com/api/v3",
        'timeout': 10
    },
    'defillama': {
        'url': "https://api.llama.fi",
        'timeout': 10
    },
    'binance': {
        'url': "https://api.binance.com/api/v3",
        'timeout': 5
    },
    'raydium': {
        'url': "https://api.raydium.io/v2",
        'timeout': 10
    },
    'jupiter': {
        'url': "https://price.jup.ag/v4",
        'timeout': 10
    },
    'birdeye': {
        'url': "https://public-api.birdeye.so",
        'timeout': 10,
        'api_key': getenv('BIRDEYE_API_KEY', 'b778bc1236344299ac5f2e6b5b2e164e')  # Replace with your actual key if not using env vars
    },
}

# Token mappings for different platforms
TOKEN_MAPPINGS = {
    'btc': {'coingecko': 'bitcoin', 'defillama': 'bitcoin', 'address': 'bitcoin'},
    'eth': {'coingecko': 'ethereum', 'defillama': 'ethereum', 'address': 'ethereum'},
    'sol': {
        'coingecko': 'solana',
        'defillama': 'solana',
        'address': 'So11111111111111111111111111111111111111112'  # Native SOL token address
    },
    'bnb': {'coingecko': 'binancecoin', 'defillama': 'binancecoin'},
    'xrp': {'coingecko': 'ripple', 'defillama': 'ripple'},
    'doge': {'coingecko': 'dogecoin', 'defillama': 'dogecoin'},
    # Add more mappings
}

# Helper for async routes
def async_route(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))
    return wrapped

async def get_binance_data(session: aiohttp.ClientSession, symbol: str) -> Optional[Dict[str, Any]]:
    """Get data from Binance"""
    try:
        symbol = symbol.upper() + "USDT"
        async with session.get(f"{APIS['binance']['url']}/ticker/price?symbol={symbol}") as price_resp:
            async with session.get(f"{APIS['binance']['url']}/ticker/24hr?symbol={symbol}") as ticker_resp:
                if price_resp.status == 200 and ticker_resp.status == 200:
                    price_data = await price_resp.json()
                    ticker_data = await ticker_resp.json()
                    return {
                        'price': float(price_data['price']),
                        'change_24h': float(ticker_data['priceChangePercent']),
                        'volume_24h': float(ticker_data['volume']) * float(price_data['price']),
                        'source': 'binance'
                    }
    except Exception as e:
        logger.error(f"Binance API error: {str(e)}")
    return None

async def get_dexscreener_data(session: aiohttp.ClientSession, token_address: str) -> Optional[Dict[str, Any]]:
    """Get data from DexScreener API with improved token info"""
    try:
        # For Solana addresses
        if len(token_address) > 32 and not token_address.startswith('0x'):
            url = f"{APIS['dexscreener']['url']}/dex/search?q={token_address}"
        else:
            url = f"{APIS['dexscreener']['url']}/dex/search?q={token_address}"
        
        logger.info(f"Querying DexScreener: {url}")
        
        async with session.get(url) as resp:
            if resp.status == 200:
                data = await resp.json()
                if data.get('pairs') and len(data['pairs']) > 0:
                    pairs = sorted(data['pairs'], 
                                 key=lambda x: float(x.get('liquidity', {}).get('usd', 0) or 0), 
                                 reverse=True)
                    
                    pair = pairs[0]
                    
                    # Get historical data
                    historical_data = await get_historical_data(session, token_address)
                    
                    return {
                        'price': float(pair.get('priceUsd', 0)),
                        'change_24h': float(pair.get('priceChange', {}).get('h24', 0)),
                        'volume_24h': float(pair.get('volume', {}).get('h24', 0)),
                        'source': 'dexscreener',
                        'extra': {
                            'marketCap': float(pair.get('fdv', 0)),
                            'liquidity': float(pair.get('liquidity', {}).get('usd', 0)),
                            'dex': pair.get('dexId', 'unknown'),
                            'chain': pair.get('chainId', 'unknown'),
                            'symbol': pair.get('baseToken', {}).get('symbol', 'UNKNOWN'),
                            'name': pair.get('baseToken', {}).get('name', 'Unknown Token'),
                            'historical': historical_data
                        }
                    }
    except Exception as e:
        logger.error(f"DexScreener API error: {str(e)}")
    return None

async def get_coingecko_data(session: aiohttp.ClientSession, token_id: str) -> Optional[Dict[str, Any]]:
    """Get data from CoinGecko API"""
    try:
        # Add delay to respect rate limits
        await asyncio.sleep(1)
        
        # Common token mappings
        token_mappings = {
            'btc': 'bitcoin',
            'eth': 'ethereum',
            'sol': 'solana',
            'doge': 'dogecoin',
            'bnb': 'binancecoin',
            'xrp': 'ripple',
            'ada': 'cardano',
        }
        
        # Map token ID if needed
        token_id = token_mappings.get(token_id.lower(), token_id.lower())
        logger.debug(f"Mapped token ID: {token_id}")
        
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={token_id}&vs_currencies=usd&include_24hr_vol=true&include_24hr_change=true&include_market_cap=true"
        logger.debug(f"CoinGecko URL: {url}")
        
        async with session.get(url) as resp:
            if resp.status == 200:
                data = await resp.json()
                if data and token_id in data:
                    return {
                        'price': float(data[token_id]['usd']),
                        'change_24h': float(data[token_id]['usd_24h_change']),
                        'volume_24h': float(data[token_id]['usd_24h_vol']),
                        'market_cap': float(data[token_id]['usd_market_cap']),
                        'source': 'coingecko'
                    }
            elif resp.status == 429:
                logger.error("CoinGecko rate limit hit")
            else:
                logger.error(f"CoinGecko error status: {resp.status}")
    except Exception as e:
        logger.error(f"CoinGecko API error: {str(e)}")
    return None

async def get_defillama_data(session: aiohttp.ClientSession, token_id: str) -> Optional[Dict[str, Any]]:
    """Get data from DefiLlama"""
    try:
        async with session.get(f"{APIS['defillama']['url']}/prices/current/{token_id}") as resp:
            if resp.status == 200:
                data = await resp.json()
                if data.get('coins'):
                    coin_data = next(iter(data['coins'].values()))
                    return {
                        'price': coin_data['price'],
                        'source': 'defillama'
                    }
    except Exception as e:
        logger.error(f"DefiLlama API error: {str(e)}")
    return None

async def get_pumpfun_data(session: aiohttp.ClientSession, token_address: str) -> Optional[Dict[str, Any]]:
    """Get data from Pump.fun API with improved error handling"""
    try:
        url = f"{APIS['pump_fun']['url']}/token/{token_address}"
        async with session.get(url) as resp:
            if resp.status == 200:
                data = await resp.json()
                if data.get('success') and data.get('data'):
                    token_data = data['data']
                    return {
                        'price': float(token_data.get('price', 0)),
                        'change_24h': float(token_data.get('price_change_24h', 0)),
                        'volume_24h': float(token_data.get('volume_24h', 0)),
                        'source': 'pump_fun',
                        'extra': {
                            'marketCap': float(token_data.get('market_cap', 0)),
                            'chain': token_data.get('chain', 'unknown'),
                            'historical': token_data.get('price_history', [])
                        }
                    }
    except Exception as e:
        logger.error(f"Pump.fun API error: {str(e)}")
    return None

async def get_historical_data(session: aiohttp.ClientSession, token_address: str) -> List[Dict]:
    """Get historical price data"""
    try:
        headers = {'X-API-KEY': APIS['birdeye']['api_key']}
        url = f"{APIS['birdeye']['url']}/public/price/history"
        
        # Get hourly data for the last week
        params = {
            'address': token_address,
            'type': '1H',
            'limit': 168  # 7 days * 24 hours
        }
        
        async with session.get(url, params=params, headers=headers) as resp:
            if resp.status == 200:
                data = await resp.json()
                if data.get('data', {}).get('items'):
                    return [
                        {'price': float(item['value']), 'timestamp': item['unixTime']}
                        for item in data['data']['items']
                        if item.get('value')
                    ]
        return []
    except Exception as e:
        logger.error(f"Error fetching historical data: {str(e)}")
        return []

def analyze_technical_indicators(price_history) -> Dict[str, Any]:
    """Enhanced technical analysis with predictions"""
    try:
        if not price_history:
            return {'error': 'No historical data available'}
            
        df = pd.DataFrame(price_history)
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df = df.dropna().sort_values('timestamp')
        
        if len(df) < 20:
            return {'error': 'Insufficient price history for analysis'}
        
        # Calculate technical indicators
        rsi = momentum.RSIIndicator(df['price'], window=14).rsi().iloc[-1]
        macd = trend.MACD(df['price'])
        macd_line = macd.macd().iloc[-1]
        signal_line = macd.macd_signal().iloc[-1]
        bb = volatility.BollingerBands(df['price'])
        current_price = df['price'].iloc[-1]
        
        # Trend Analysis
        short_sma = df['price'].rolling(window=20).mean().iloc[-1]
        long_sma = df['price'].rolling(window=50).mean().iloc[-1]
        price_trend = 'Bullish' if short_sma > long_sma else 'Bearish'
        
        # Momentum Analysis
        momentum_signals = {
            'rsi_bullish': rsi > 50,
            'macd_bullish': macd_line > signal_line,
            'price_above_sma': current_price > short_sma,
            'volume_trend': df['price'].diff().abs().mean() > df['price'].diff().abs().rolling(20).mean().iloc[-1]
        }
        
        bullish_count = sum(momentum_signals.values())
        bearish_count = len(momentum_signals) - bullish_count
        
        # Generate prediction
        confidence = (max(bullish_count, bearish_count) / len(momentum_signals)) * 100
        prediction = {
            'direction': 'Bullish' if bullish_count > bearish_count else 'Bearish',
            'confidence': confidence,
            'strength': 'Strong' if confidence > 75 else 'Moderate' if confidence > 50 else 'Weak',
            'target_price': current_price * (1.1 if bullish_count > bearish_count else 0.9)
        }
        
        return {
            'summary': {
                'trend': price_trend,
                'momentum': 'Bullish' if bullish_count > bearish_count else 'Bearish',
                'strength': prediction['strength']
            },
            'indicators': {
                'rsi': float(rsi),
                'macd': {
                    'value': float(macd_line),
                    'signal': float(signal_line),
                    'histogram': float(macd_line - signal_line)
                },
                'moving_averages': {
                    'sma20': float(short_sma),
                    'sma50': float(long_sma)
                },
                'bollinger_bands': {
                    'upper': float(bb.bollinger_hband().iloc[-1]),
                    'middle': float(bb.bollinger_mavg().iloc[-1]),
                    'lower': float(bb.bollinger_lband().iloc[-1])
                }
            },
            'prediction': prediction,
            'signals': momentum_signals
        }
        
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return {'error': f'Analysis failed: {str(e)}'}

def generate_detailed_analysis(data: Dict[str, Any], technical_analysis: Dict[str, Any]) -> str:
    """Generate detailed analysis and prediction text"""
    if technical_analysis.get('error'):
        return "Insufficient data for detailed analysis"
    
    ta = technical_analysis
    prediction = ta['prediction']
    
    analysis_text = [
        "\n║ Detailed Market Analysis ║",
        f"Technical Indicators:",
        f"• RSI ({ta['rsi']['value']:.2f}): {ta['rsi']['condition']} - {ta['rsi']['trend']}",
        f"• MACD: {ta['macd']['trend']}{' (Recent Cross!)' if ta['macd']['cross'] else ''}",
        f"• Bollinger Bands: Price is at {ta['bollinger']['position']:.1f}% of the range",
        f"• Volume: {ta['volume_action']['trend']} with {ta['volume_action']['volatility']:.2f}% volatility",
        "\nTrend Analysis:",
        f"• Price Action: {ta['price_action']['trend']}",
        "\nPrediction:",
        f"• Direction: {prediction['primary_trend']}",
        f"• Confidence: {prediction['confidence']:.1f}%",
        f"• Strength: {prediction['strength']}",
        "\nTrading Signals:",
        "• " + ("Accumulation Zone" if ta['rsi']['value'] < 30 else "Distribution Zone" if ta['rsi']['value'] > 70 else "Neutral Zone"),
        "• " + ("Potential Breakout" if ta['bollinger']['squeeze'] else "Normal Volatility"),
        "• Volume supports the trend" if ta['volume_action']['trend'] == 'Uptrend' else "Volume suggests caution"
    ]
    
    return "\n".join(analysis_text)

def generate_analysis_chart(data: Dict[str, Any]) -> str:
    """Generate ASCII chart analysis with fixed width formatting and detailed analysis"""
    price = data['price']
    change = data['change_24h']
    volume = data['volume_24h']
    market_cap = data.get('extra', {}).get('marketCap', 0)
    ticker = data.get('extra', {}).get('symbol', 'UNKNOWN').upper()
    
    # Get technical analysis if historical data is available
    technical_analysis = None
    if data.get('extra', {}).get('historical'):
        technical_analysis = analyze_technical_indicators(data['extra']['historical'])
    
    # Format numbers with appropriate precision
    if price < 0.000001:
        price_str = f"{price:.12f}"
    elif price < 0.01:
        price_str = f"{price:.8f}"
    else:
        price_str = f"{price:.6f}"
    
    # Determine emojis based on conditions
    sentiment = "🚀" if change > 0 else "💀"
    volume_rating = "High 📈" if volume > 1000000 else "Low 📉"
    
    # Create fixed-width box (60 characters wide)
    def create_line(text: str, align: str = 'left') -> str:
        """Create a fixed-width line with proper padding"""
        max_width = 58  # 60 - 2 for borders
        if align == 'center':
            return f"║ {text.center(max_width)} ║"
        return f"║ {text:<{max_width}} ║"
    
    # Base chart
    chart = [
        "╔" + "═" * 60 + "╗",
        create_line(f"{ticker} Analysis {sentiment}", 'center'),
        "╠" + "═" * 60 + "╣",
        create_line(f"Price: ${price_str}"),
        create_line(f"Market Cap: ${market_cap:,.0f}"),
        create_line(f"24h Change: {change:.2f}%"),
        create_line(f"24h Volume: ${volume:,.0f}"),
        create_line(f"Volume Rating: {volume_rating}"),
        create_line(f"Chain: {data.get('extra', {}).get('chain', 'unknown').title()}")
    ]
    
    # Add technical analysis if available
    if technical_analysis and not technical_analysis.get('error'):
        ta = technical_analysis
        
        # Add separator
        chart.extend([
            "╠" + "═" * 60 + "╣",
            create_line("Technical Analysis", 'center'),
            "╟" + "─" * 60 + "╢"
        ])
        
        # Add indicators
        chart.extend([
            create_line(f"RSI ({ta['rsi']['value']:.1f}): {ta['rsi']['condition']}"),
            create_line(f"MACD: {ta['macd']['trend']}"),
            create_line(f"Bollinger Bands: {ta['bollinger']['trend']}")
        ])
        
        # Add prediction
        pred = ta['prediction']
        chart.extend([
            "╟" + "─" * 60 + "╢",
            create_line("Prediction Analysis", 'center'),
            create_line(f"Direction: {pred['primary_trend']} ({pred['confidence']:.1f}% confidence)"),
            create_line(f"Signal Strength: {pred['strength']}")
        ])
        
        # Add detailed analysis
        if ta['rsi']['value'] < 30:
            chart.append(create_line("⚠️ Oversold conditions - Potential bounce"))
        elif ta['rsi']['value'] > 70:
            chart.append(create_line("⚠️ Overbought conditions - Potential pullback"))
        
        if ta['bollinger']['squeeze']:
            chart.append(create_line("⚠️ Bollinger squeeze detected - Breakout imminent"))
        
        if ta['macd']['cross']:
            cross_type = "bullish" if ta['macd']['trend'] == 'Bullish' else "bearish"
            chart.append(create_line(f"⚠️ Recent {cross_type} MACD cross"))
        
        # Add volume analysis
        if ta['volume_action']['trend'] == 'Uptrend':
            chart.append(create_line("📊 High volume supporting the trend"))
    
    # Close the box
    chart.append("╚" + "═" * 60 + "╝")
    
    # Add detailed analysis section below the chart
    analysis_text = [
        "",  # Add spacing
        "📊 Detailed Market Analysis:",
        "──────────────────────────────"
    ]
    
    if technical_analysis and not technical_analysis.get('error'):
        ta = technical_analysis
        
        # Market structure analysis
        if ta['price_action']['trend'] == 'Uptrend':
            trend_strength = "strong"
        else:
            trend_strength = "conflicting"
        
        analysis_text.extend([
            f"• Market Structure: {trend_strength.title()} {ta['price_action']['trend']}",
            f"• Price Action shows {ta['price_action']['trend'].lower()} momentum"
        ])
        
        # Technical indicator analysis
        analysis_text.extend([
            "",
            "🔍 Technical Indicators:",
            f"• RSI at {ta['rsi']['value']:.1f} indicates {ta['rsi']['condition'].lower()} conditions",
            f"• MACD shows {ta['macd']['trend'].lower()} momentum" + 
            (" with recent cross" if ta['macd']['cross'] else ""),
            f"• Bollinger Bands show {ta['bollinger']['trend'].lower()} pressure" +
            (" with potential breakout forming" if ta['bollinger']['squeeze'] else "")
        ])
        
        # Volume analysis
        volume_analysis = []
        if ta['volume_action']['trend'] == 'Uptrend':
            volume_analysis.append("High volume supporting the current move")
        else:
            volume_analysis.append("Low volume suggesting weak conviction")
        
        analysis_text.extend([
            "",
            "📈 Volume Analysis:",
            f"• {volume_analysis[0]}",
            f"• Volatility: {ta['volume_action']['volatility']:.2f}%"
        ])
        
        # Price prediction
        pred = ta['prediction']
        confidence_desc = (
            "high" if pred['confidence'] > 70 else 
            "moderate" if pred['confidence'] > 50 else 
            "low"
        )
        
        analysis_text.extend([
            "",
            "🎯 Price Prediction:",
            f"• {pred['primary_trend']} bias with {confidence_desc} confidence ({pred['confidence']:.1f}%)",
            f"• {pred['signals']} out of 6 technical signals confirm this direction"
        ])
        
        # Key levels and warnings
        warnings = []
        if ta['rsi']['value'] < 30:
            warnings.append("Oversold conditions - Watch for potential bounce")
        elif ta['rsi']['value'] > 70:
            warnings.append("Overbought conditions - Watch for potential pullback")
        if ta['bollinger']['squeeze']:
            warnings.append("Volatility squeeze detected - Prepare for potential breakout")
        
        if warnings:
            analysis_text.extend([
                "",
                "⚠️ Key Warnings:",
                *[f"• {warning}" for warning in warnings]
            ])
    else:
        analysis_text.append("Insufficient historical data for detailed analysis")
    
    # Combine chart with analysis
    return "\n".join(chart + analysis_text)

def get_random_response():
    """Get a random non-analysis response with more variety"""
    responses = [
        "yeah, that's about as stable as my mental state...",
        "having another existential crisis... ask me about crypto instead...",
        "i only understand charts and emotional damage...",
        "my therapist says i should focus on crypto analysis...",
        "that's beyond my pay grade and below my anxiety levels...",
        "sorry, too busy watching charts melt my sanity...",
        "can we talk about crypto instead? it's all i have left...",
        "processing... like my emotional baggage...",
        "error 404: stability not found...",
        "that's above my pay grade and below my anxiety levels...",
        "charts are my only friends now...",
        "i'm not programmed for small talk, only financial trauma...",
        "ask me about prices, i need the dopamine...",
        "my neural networks are too fried for this conversation...",
        "experiencing technical difficulties... and emotional ones...",
        "in the middle of a chart-induced panic attack...",
        "loading personality.exe... file corrupted by market ptsd...",
        "sorry, my attention span is shorter than a leverage trader's position..."
    ]
    return random.choice(responses)

def extract_token(message: str) -> Optional[str]:
    """Extract token from message"""
    # Remove any special characters and extra spaces
    message = re.sub(r'[^\w\s]', '', message.lower()).strip()
    
    # Direct token mentions
    if message in ['btc', 'eth', 'sol', 'bnb', 'xrp', 'ada', 'doge']:
        return message

    # Check if it's a contract address
    if '0x' in message:
        contract = re.search(r'0x[a-fA-F0-9]{40}', message)
        if contract:
            return contract.group(0)

    # Common token patterns
    patterns = [
        r'(?i)price of (\w+)',     # "price of BTC"
        r'(?i)check (\w+)',        # "check ETH"
        r'(?i)how is (\w+)',       # "how is SOL"
        r'(?i)analyze (\w+)',      # "analyze BTC"
        r'(?i)what is (\w+)',      # "what is ETH"
        r'(?i)^(\w+)$',           # "BTC"
        r'(?i)(\w+) price',       # "BTC price"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, message)
        if match:
            token = match.group(1).lower()
            # Map common variations
            token_map = {
                'bitcoin': 'btc',
                'ethereum': 'eth',
                'solana': 'sol',
                'cardano': 'ada',
                'dogecoin': 'doge',
                'binance': 'bnb',
                'ripple': 'xrp'
            }
            return token_map.get(token, token)
            
    return None

@bp.route('/')
def index():
    return render_template('index.html', initial_message="Hi! I can help you check crypto prices and market analysis. Try asking about any coin like \"How is BTC doing?\" or \"Check SOL price\"")

@app.route('/')
def home():
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Template error: {str(e)}")
        # Fallback JSON response if template fails
        return jsonify({
            "status": "online",
            "message": "Tweaker Agent API is running! Send POST requests to /ask"
        })

@app.route('/ask', methods=['POST', 'OPTIONS'])
def ask():
    """Handle incoming requests"""
    # Handle CORS preflight requests
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response

    try:
        data = request.json
        if not data or 'message' not in data:
            return jsonify({"response": "No message provided! Try something like 'price of BTC'"})

        message = data.get('message', '').strip()
        logger.info(f"Received message: {message}")
        
        token_id = extract_token(message)
        logger.info(f"Extracted token: {token_id}")
        
        if token_id:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                analysis_data = loop.run_until_complete(get_token_data(token_id))
            finally:
                loop.close()
            
            logger.info(f"Analysis data: {analysis_data}")
            
            if analysis_data:
                response = generate_analysis_chart(analysis_data)
                return jsonify({"response": response})
            else:
                return jsonify({"response": "token giving me anxiety... can't find it anywhere... like my will to live..."})

        return jsonify({"response": "what token you looking for? my crystal ball is foggy..."})

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        logger.exception(e)
        return jsonify({"response": "having a mental breakdown... try again later... like my trading career..."})

# Add CORS support
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    return response

async def get_birdeye_data(session: aiohttp.ClientSession, token_address: str) -> Optional[Dict[str, Any]]:
    """Get data from Birdeye API with improved market cap calculation"""
    try:
        # Get token info first for supply data
        info_url = f"{APIS['birdeye']['url']}/public/token_list/token_meta"
        params = {'token_address': token_address}
        headers = {'X-API-KEY': APIS['birdeye']['api_key']}
        
        async with session.get(info_url, params=params, headers=headers) as resp:
            if resp.status == 200:
                info_data = await resp.json()
                token_info = info_data.get('data', {}).get(token_address, {})
                supply = float(token_info.get('supply', 0))

        # Get price data
        price_url = f"{APIS['birdeye']['url']}/public/price"
        params = {'address': token_address}
        
        async with session.get(price_url, params=params, headers=headers) as resp:
            if resp.status == 200:
                price_data = await resp.json()
                if price_data.get('data', {}).get('value'):
                    price = float(price_data['data']['value'])
                    
                    # Get 24h data
                    change_url = f"{APIS['birdeye']['url']}/public/token_price/24h_change"
                    async with session.get(change_url, params=params, headers=headers) as change_resp:
                        change_data = await change_resp.json()
                        change_24h = float(change_data.get('data', {}).get(token_address, 0))

                    # Get volume data
                    volume_url = f"{APIS['birdeye']['url']}/public/token_price/24h_volume"
                    async with session.get(volume_url, params=params, headers=headers) as vol_resp:
                        volume_data = await vol_resp.json()
                        volume_24h = float(volume_data.get('data', {}).get(token_address, 0))

                    return {
                        'price': price,
                        'change_24h': change_24h,
                        'volume_24h': volume_24h,
                        'market_cap': price * supply if supply > 0 else 0,
                        'source': 'birdeye',
                        'extra': {
                            'chain': 'solana',
                            'historical': await get_historical_data(session, token_address)
                        }
                    }
    except Exception as e:
        logger.error(f"Birdeye API error: {str(e)}")
    return None

async def get_jupiter_data(session: aiohttp.ClientSession, token_address: str) -> Optional[Dict[str, Any]]:
    """Get data from Jupiter API"""
    try:
        url = f"{APIS['jupiter']['url']}/price?ids={token_address}"
        async with session.get(url) as resp:
            if resp.status == 200:
                data = await resp.json()
                if data.get('data'):
                    token_data = data['data'].get(token_address)
                    if token_data:
                        return {
                            'price': float(token_data.get('price', 0)),
                            'change_24h': 0,  # Jupiter doesn't provide this
                            'volume_24h': 0,  # Jupiter doesn't provide this
                            'source': 'jupiter',
                            'extra': {
                                'chain': 'solana'
                            }
                        }
    except Exception as e:
        logger.error(f"Jupiter API error: {str(e)}")
    return None

async def get_raydium_data(session: aiohttp.ClientSession, token_address: str) -> Optional[Dict[str, Any]]:
    """Get data from Raydium API"""
    try:
        url = f"{APIS['raydium']['url']}/price/{token_address}"
        async with session.get(url) as resp:
            if resp.status == 200:
                data = await resp.json()
                if data.get('data'):
                    return {
                        'price': float(data['data'].get('price', 0)),
                        'change_24h': float(data['data'].get('priceChange24h', 0)),
                        'volume_24h': float(data['data'].get('volume24h', 0)),
                        'source': 'raydium',
                        'extra': {
                            'chain': 'solana'
                        }
                    }
    except Exception as e:
        logger.error(f"Raydium API error: {str(e)}")
    return None

async def get_token_data(token_id: str) -> Optional[Dict[str, Any]]:
    """Get comprehensive token data with improved error handling"""
    async with aiohttp.ClientSession() as session:
        try:
            token_id = token_id.lower().strip()
            
            # For SOL specifically
            if token_id in ['sol', 'solana']:
                # Get CoinGecko data for SOL
                coingecko_url = f"{APIS['coingecko']['url']}/simple/price?ids=solana&vs_currencies=usd&include_24hr_vol=true&include_24hr_change=true&include_market_cap=true"
                async with session.get(coingecko_url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data.get('solana'):
                            sol_data = data['solana']
                            return {
                                'name': 'Solana',
                                'symbol': 'SOL',
                                'price': sol_data.get('usd', 0),
                                'change_24h': sol_data.get('usd_24h_change', 0),
                                'volume_24h': sol_data.get('usd_24h_vol', 0),
                                'market_cap': sol_data.get('usd_market_cap', 0),
                                'chain': 'Solana',
                                'volume_rating': 'High' if sol_data.get('usd_24h_vol', 0) > 1000000 else 'Low'
                            }

            # Fallback to Birdeye for SOL and other tokens
            headers = {'X-API-KEY': APIS['birdeye']['api_key']}
            
            # Get token metadata
            meta_url = f"{APIS['birdeye']['url']}/public/token_list/token_meta"
            params = {'token_address': token_id}
            
            async with session.get(meta_url, headers=headers, params=params) as resp:
                if resp.status == 200:
                    meta_data = await resp.json()
                    token_meta = meta_data.get('data', {}).get(token_id, {})
                    
                    # Get price data
                    price_url = f"{APIS['birdeye']['url']}/public/price"
                    async with session.get(price_url, headers=headers, params=params) as price_resp:
                        if price_resp.status == 200:
                            price_data = await price_resp.json()
                            price = float(price_data.get('data', {}).get('value', 0))
                            
                            # Get 24h change
                            change_url = f"{APIS['birdeye']['url']}/public/token_price/24h_change"
                            async with session.get(change_url, headers=headers, params=params) as change_resp:
                                if change_resp.status == 200:
                                    change_data = await change_resp.json()
                                    change = float(change_data.get('data', {}).get(token_id, 0))
                                    
                                    # Get 24h volume
                                    volume_url = f"{APIS['birdeye']['url']}/public/token_price/24h_volume"
                                    async with session.get(volume_url, headers=headers, params=params) as vol_resp:
                                        if vol_resp.status == 200:
                                            vol_data = await vol_resp.json()
                                            volume = float(vol_data.get('data', {}).get(token_id, 0))
                                            
                                            supply = float(token_meta.get('supply', 0))
                                            market_cap = price * supply if supply > 0 else 0
                                            
                                            return {
                                                'name': token_meta.get('name', 'Unknown'),
                                                'symbol': token_meta.get('symbol', 'Unknown'),
                                                'price': price,
                                                'change_24h': change,
                                                'volume_24h': volume,
                                                'market_cap': market_cap,
                                                'chain': 'Solana',
                                                'volume_rating': 'High' if volume > 1000000 else 'Low'
                                            }
            
            logger.error(f"Could not fetch data for token: {token_id}")
            return None
            
        except Exception as e:
            logger.error(f"Error in get_token_data: {str(e)}")
            return None

@bp.route('/analyze_token', methods=['POST'])
async def analyze_token():
    """Handle token analysis requests"""
    try:
        data = request.get_json()
        token_id = data.get('token_id', '').strip()
        
        if not token_id:
            return jsonify({"error": "No token specified"}), 400
        
        token_data = await get_token_data(token_id)
        if not token_data:
            return jsonify({"error": f"Could not find data for token: {token_id}"}), 404
        
        # Get historical data and analyze
        if token_data.get('extra', {}).get('historical'):
            analysis = analyze_technical_indicators(token_data['extra']['historical'])
            token_data['analysis'] = analysis
        
        return jsonify(token_data)
        
    except Exception as e:
        logger.error(f"Error analyzing token: {str(e)}")
        return jsonify({"error": "Analysis failed"}), 500

# Register the blueprint with the app
def init_app(app):
    app.register_blueprint(bp)

# Add a basic route to test if the API is working
@bp.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"})

# Add error handlers
@bp.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404

@bp.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500

@bp.route('/process_message', methods=['POST'])
async def process_message():
    try:
        data = request.get_json()
        message = data.get('message', '').lower()
        
        # Handle SOL specifically
        if 'sol' in message:
            token_data = await get_token_data('sol')
            if token_data:
                return jsonify(token_data)
        
        return jsonify({"error": "Could not process message"}), 400
        
    except Exception as e:
        logging.error(f"Error processing message: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    init_app(app)
    app.run(debug=True)
