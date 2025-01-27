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
    },
    'jupiter': {
        'url': "https://price.jup.ag/v4",
        'timeout': 10
    },
    'birdeye': {
        'url': "https://public-api.birdeye.so",
        'timeout': 10,
        'api_key': 'YOUR_BIRDEYE_API_KEY'  # Replace with your API key
    },
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

async def get_dexscreener_data(session: aiohttp.ClientSession, token_address: str) -> Optional[Dict[str, Any]]:
    """Get data from DexScreener API with improved error handling"""
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
                    # Sort pairs by liquidity
                    pairs = sorted(data['pairs'], 
                                 key=lambda x: float(x.get('liquidity', {}).get('usd', 0) or 0), 
                                 reverse=True)
                    
                    pair = pairs[0]  # Get the most liquid pair
                    
                    return {
                        'price': float(pair.get('priceUsd', 0)),
                        'change_24h': float(pair.get('priceChange', {}).get('h24', 0)),
                        'volume_24h': float(pair.get('volume', {}).get('h24', 0)),
                        'source': 'dexscreener',
                        'extra': {
                            'marketCap': float(pair.get('fdv', 0)),
                            'liquidity': float(pair.get('liquidity', {}).get('usd', 0)),
                            'dex': pair.get('dexId', 'unknown'),
                            'chain': pair.get('chainId', 'unknown')
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

def generate_analysis_chart(data: Dict[str, Any]) -> str:
    """Generate ASCII chart analysis with improved formatting"""
    price = data['price']
    change = data['change_24h']
    volume = data['volume_24h']
    market_cap = data.get('extra', {}).get('marketCap', 0)
    
    # Determine market sentiment indicators
    sentiment = "🚀" if change > 0 else "💀"
    volume_rating = "High 📈" if volume > 1000000 else "Low 📉"
    
    # Format numbers with appropriate precision
    if price < 0.000001:
        price_str = f"{price:.12f}"
    elif price < 0.01:
        price_str = f"{price:.8f}"
    else:
        price_str = f"{price:.6f}"
    
    return "\n".join([
        f"║ Market Analysis {sentiment} ║",
        f"║ Price: ${price_str} ║",
        f"║ Market Cap: ${market_cap:,.0f} ║",
        f"║ 24h Change: {change:.2f}% ║",
        f"║ 24h Volume: ${volume:,.0f} ║",
        f"║ Volume Rating: {volume_rating} ║",
        f"║ Source: {data['source'].title()} ║",
        f"║ Chain: {data.get('extra', {}).get('chain', 'unknown').title()} ║"
    ])

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
    """Get data from Birdeye API"""
    try:
        url = f"{APIS['birdeye']['url']}/public/token_list?chain=solana"
        headers = {'X-API-KEY': APIS['birdeye']['api_key']}
        
        async with session.get(url, headers=headers) as resp:
            if resp.status == 200:
                data = await resp.json()
                if data.get('success') and data.get('data', {}).get('tokens'):
                    token = next((t for t in data['data']['tokens'] 
                                if t.get('address') == token_address), None)
                    if token:
                        price_url = f"{APIS['birdeye']['url']}/public/price?address={token_address}"
                        async with session.get(price_url, headers=headers) as price_resp:
                            if price_resp.status == 200:
                                price_data = await price_resp.json()
                                if price_data.get('data'):
                                    return {
                                        'price': float(price_data['data'].get('value', 0)),
                                        'change_24h': float(price_data['data'].get('priceChange24h', 0)),
                                        'volume_24h': float(price_data['data'].get('volume24h', 0)),
                                        'source': 'birdeye',
                                        'extra': {
                                            'chain': 'solana',
                                            'symbol': token.get('symbol', ''),
                                            'name': token.get('name', '')
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
    """Get token data from multiple sources"""
    async with aiohttp.ClientSession() as session:
        results = []
        
        # Try DexScreener first
        dex_data = await get_dexscreener_data(session, token_id)
        if dex_data:
            results.append(dex_data)
        
        # If it's a Solana address, try Birdeye
        if len(token_id) > 32 and not token_id.startswith('0x'):
            birdeye_data = await get_birdeye_data(session, token_id)
            if birdeye_data:
                results.append(birdeye_data)
        
        # If we got any results, return the one with highest volume
        if results:
            return max(results, 
                      key=lambda x: float(x.get('volume_24h', 0)) if x.get('volume_24h') else 0)
        
        # If no results, try CoinGecko as fallback
        return await get_coingecko_data(session, token_id)

if __name__ == '__main__':
    app.run(debug=True)
