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
from typing import Dict, Any, Optional
import re

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
    }
}

# Token mappings for different platforms
TOKEN_MAPPINGS = {
    'btc': {'coingecko': 'bitcoin', 'defillama': 'bitcoin'},
    'eth': {'coingecko': 'ethereum', 'defillama': 'ethereum'},
    'sol': {'coingecko': 'solana', 'defillama': 'solana'},
    'bnb': {'coingecko': 'binancecoin', 'defillama': 'binancecoin'},
    'xrp': {'coingecko': 'ripple', 'defillama': 'ripple'},
    'doge': {'coingecko': 'dogecoin', 'defillama': 'dogecoin'},
    # Add more mappings
}

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

async def get_dexscreener_data(session: aiohttp.ClientSession, token_id: str) -> Optional[Dict[str, Any]]:
    """Get data from DexScreener API"""
    try:
        # Clean the input
        token_id = token_id.strip()
        logger.info(f"Querying DexScreener for: {token_id}")
        
        # Try search endpoint first
        search_url = f"{APIS['dexscreener']['url']}/pairs/search?q={token_id}"
        async with session.get(search_url) as resp:
            if resp.status == 200:
                data = await resp.json()
                pairs = data.get('pairs', [])
                if pairs and len(pairs) > 0:
                    pair = pairs[0]
                    return {
                        'price': float(pair.get('priceUsd', 0)),
                        'change_24h': float(pair.get('priceChange', {}).get('h24', 0)),
                        'volume_24h': float(pair.get('volume', {}).get('h24', 0)),
                        'source': 'dexscreener',
                        'extra': {
                            'liquidity': pair.get('liquidity', {}).get('usd', 0),
                            'dex': pair.get('dexId', 'unknown'),
                            'chain': pair.get('chainId', 'unknown'),
                            'address': pair.get('baseToken', {}).get('address', '')
                        }
                    }
    except Exception as e:
        logger.error(f"DexScreener API error: {str(e)}")
    return None

async def get_coingecko_data(session: aiohttp.ClientSession, token_id: str) -> Optional[Dict[str, Any]]:
    """Get data from CoinGecko API"""
    try:
        # Common token mappings
        token_mappings = {
            'btc': 'bitcoin',
            'eth': 'ethereum',
            'sol': 'solana',
            'doge': 'dogecoin',
            'bnb': 'binancecoin',
            'xrp': 'ripple',
            'ada': 'cardano',
            'dot': 'polkadot',
            'link': 'chainlink',
            'uni': 'uniswap',
            'matic': 'polygon'
        }
        
        # Clean and standardize token ID
        token_id = token_id.lower().strip()
        if token_id in token_mappings:
            token_id = token_mappings[token_id]

        url = f"{APIS['coingecko']['url']}/simple/price?ids={token_id}&vs_currencies=usd&include_24hr_vol=true&include_24hr_change=true&include_market_cap=true"
        logger.info(f"Querying CoinGecko: {url}")
        
        async with session.get(url) as resp:
            if resp.status == 200:
                data = await resp.json()
                if data and token_id in data:
                    token_data = data[token_id]
                    return {
                        'price': float(token_data.get('usd', 0)),
                        'change_24h': float(token_data.get('usd_24h_change', 0)),
                        'volume_24h': float(token_data.get('usd_24h_vol', 0)),
                        'source': 'coingecko',
                        'extra': {
                            'marketCap': float(token_data.get('usd_market_cap', 0)),
                            'liquidity': 0,  # CoinGecko doesn't provide liquidity
                            'holders': 'N/A'  # CoinGecko doesn't provide holder count
                        }
                    }

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

async def get_pumpfun_data(session: aiohttp.ClientSession, token_id: str) -> Optional[Dict[str, Any]]:
    """Get data from Pump.fun API"""
    try:
        # Remove any $ prefix and spaces
        token_id = token_id.replace('$', '').strip()
        
        # Try both with and without /tokens/ endpoint
        urls = [
            f"{APIS['pump_fun']['url']}/tokens/{token_id}",
            f"{APIS['pump_fun']['url']}/{token_id}"
        ]
        
        for url in urls:
            logger.info(f"Trying PumpFun URL: {url}")
            try:
                async with session.get(url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        logger.info(f"PumpFun response: {data}")
                        
                        if data.get('data'):
                            token_data = data['data']
                            return {
                                'price': float(token_data.get('price', 0)),
                                'change_24h': float(token_data.get('price_change_24h', 0)),
                                'volume_24h': float(token_data.get('volume_24h', 0)),
                                'source': 'pump_fun',
                                'extra': {
                                    'liquidity': token_data.get('liquidity', 0),
                                    'dex': 'pump.fun'
                                }
                            }
            except Exception as e:
                logger.error(f"Error with URL {url}: {str(e)}")
                continue
                
    except Exception as e:
        logger.error(f"Pump.fun API error for {token_id}: {str(e)}")
    return None

async def get_token_data(token_id: str) -> Optional[Dict[str, Any]]:
    """Get token data from multiple sources"""
    try:
        async with aiohttp.ClientSession() as session:
            # Try DexScreener first
            dex_result = await get_dexscreener_data(session, token_id)
            if dex_result:
                return dex_result
            
            # Fallback to CoinGecko
            cg_result = await get_coingecko_data(session, token_id)
            if cg_result:
                return cg_result
                
    except Exception as e:
        logger.error(f"Error getting token data: {str(e)}")
    return None

def format_analysis(data: Dict[str, Any]) -> str:
    """Format token analysis data into a chart display"""
    try:
        # Get basic price data
        price = data.get('price', 0)
        change_24h = data.get('change_24h', 0)
        volume_24h = data.get('volume_24h', 0)
        
        # Get additional metrics from extra data
        extra = data.get('extra', {})
        mcap = extra.get('marketCap', 0)
        holders = extra.get('holders', 'N/A')
        liquidity = extra.get('liquidity', 0)
        
        # Generate prediction based on 24h change
        if change_24h > 5:
            prediction = "bullish af... like my hopium addiction"
        elif change_24h > 0:
            prediction = "slightly bullish... like my morning coffee"
        elif change_24h < -5:
            prediction = "bearish af... like my life savings"
        else:
            prediction = "crabbing... like my trading strategy"

        # Format numbers with commas and limit decimal places
        def format_number(num):
            if isinstance(num, (int, float)):
                if num > 1:
                    return f"${num:,.2f}"
                else:
                    return f"${num:.8f}"
            return str(num)

        # Format with exact spacing and layout
        sources = data.get('sources', [data.get('source', 'unknown')])
        analysis = (
            f"📊 CHART ANALYSIS 📊\n"
            f"------------------------\n"
            f"💰Price: {format_number(price)}\n\n"
            f"📈 24h Change: {change_24h:+.2f}%\n"
            f"💎 Market Cap: {format_number(mcap)}\n\n"
            f"🏊 Liquidity: {format_number(liquidity)}\n"
            f"👥 Holders: {holders}\n\n"
            f"📊 Volume 24h: {format_number(volume_24h)}\n\n"
            f"------------------------\n"
            f"🔮 Prediction: {prediction}\n\n"
            f"------------------------\n\n"
            f"📡 Data: {', '.join(sources)}"
        )
        
        return analysis

    except Exception as e:
        logger.error(f"Error formatting analysis: {str(e)}")
        return "chart machine broke... like my portfolio..."

def get_random_response():
    """Get a random non-analysis response with more variety"""
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

def extract_token(msg: str) -> Optional[str]:
    """Extract token from message"""
    try:
        if not msg:
            return None
            
        msg = msg.lower().strip()
        logger.info(f"Extracting token from: {msg}")
        
        # Handle "price of" format
        if 'price of' in msg:
            return msg.split('price of')[1].strip()
            
        # Handle "check" format
        if 'check' in msg:
            parts = msg.split('check')
            if len(parts) > 1:
                return parts[1].strip()
        
        # Handle direct token input
        return msg.strip()

    except Exception as e:
        logger.error(f"Error in extract_token: {str(e)}")
        return None

@bp.route('/')
def index():
    return render_template('index.html')

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
        message = data.get('message', '').strip()
        logger.info(f"Received message: {message}")
        
        # Extract token from message
        token_id = extract_token(message)
        logger.info(f"Extracted token: {token_id}")
        
        if token_id:
            # Create new event loop for async operations
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                analysis_data = loop.run_until_complete(get_token_data(token_id))
            finally:
                loop.close()
            
            logger.info(f"Analysis data: {analysis_data}")
            
            if analysis_data:
                response = format_analysis(analysis_data)
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

if __name__ == '__main__':
    app.run(debug=True)
