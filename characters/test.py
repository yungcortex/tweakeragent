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
import aiohttp
import asyncio
from web3 import Web3
from typing import Dict, Any, Optional

# Setup logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get the absolute path to the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Initialize Flask with the correct paths
app = Flask(__name__,
           static_url_path='/static',
           static_folder=os.path.join(project_root, 'static'),
           template_folder=os.path.join(project_root, 'templates'))

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
    """Get data from DexScreener with additional token info"""
    try:
        # Remove any $ prefix and spaces
        token_id = token_id.replace('$', '').strip()
        
        # Try multiple endpoints for DexScreener
        urls = [
            f"{APIS['dexscreener']['url']}/pairs/solana/{token_id}",
            f"{APIS['dexscreener']['url']}/pairs/search?q={token_id}",
            f"{APIS['dexscreener']['url']}/pairs/ethereum/{token_id}"
        ]
        
        for url in urls:
            try:
                async with session.get(url) as resp:
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
                                    'address': pair.get('baseToken', {}).get('address', ''),
                                }
                            }
            except Exception as e:
                logger.error(f"Error with URL {url}: {str(e)}")
                continue
                
    except Exception as e:
        logger.error(f"DexScreener API error for {token_id}: {str(e)}")
    return None

async def get_coingecko_data(session: aiohttp.ClientSession, token_id: str) -> Optional[Dict[str, Any]]:
    """Get data from CoinGecko"""
    try:
        coin_id = TOKEN_MAPPINGS.get(token_id.lower(), {}).get('coingecko', token_id.lower())
        async with session.get(
            f"{APIS['coingecko']['url']}/simple/price?ids={coin_id}&vs_currencies=usd&include_24hr_vol=true&include_24hr_change=true"
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                if data.get(coin_id):
                    return {
                        'price': data[coin_id]['usd'],
                        'change_24h': data[coin_id]['usd_24h_change'],
                        'volume_24h': data[coin_id]['usd_24h_vol'],
                        'source': 'coingecko'
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
    """Get token data from multiple sources with fallbacks"""
    try:
        # Clean the token_id
        token_id = token_id.strip().replace('$', '')
        logger.info(f"Searching for token: {token_id}")
        
        async with aiohttp.ClientSession() as session:
            # Try all sources concurrently
            tasks = [
                get_dexscreener_data(session, token_id),
                get_pumpfun_data(session, token_id)
            ]
            
            # Add other sources for non-contract tokens
            if len(token_id) < 30:
                tasks.extend([
                    get_coingecko_data(session, token_id),
                    get_defillama_data(session, token_id)
                ])
                
                if token_id.upper() in ['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'DOGE']:
                    tasks.append(get_binance_data(session, token_id))

            # Wait for all API calls to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out errors and None results
            valid_results = [r for r in results if isinstance(r, dict)]
            logger.info(f"Valid results found: {len(valid_results)}")
            
            if valid_results:
                # For contract addresses, prioritize DexScreener and PumpFun
                if len(token_id) > 30:
                    for source in ['dexscreener', 'pump_fun']:
                        result = next((r for r in valid_results if r['source'] == source), None)
                        if result:
                            result['sources'] = [r['source'] for r in valid_results]
                            return result

                # For other tokens, use any available source
                result = valid_results[0]
                result['sources'] = [r['source'] for r in valid_results]
                return result

    except Exception as e:
        logger.error(f"Error in get_token_data: {str(e)}")
        logger.exception(e)
    return None

def format_analysis(data: Dict[str, Any]) -> str:
    """Format token analysis data into a chart display with DexScreener chart link"""
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
        
        # Get token address and chain for DexScreener link
        token_address = extra.get('address', '')
        chain = extra.get('chain', 'solana').lower()
        
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

        # Format with exact line breaks as specified
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
            f"📡 Data: {', '.join(sources)}\n"
        )

        # Add DexScreener chart link if we have a token address
        if token_address:
            dex_url = f"https://dexscreener.com/{chain}/{token_address}"
            analysis += f"\n📈 Live Chart: {dex_url}\n"
            
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
    """Enhanced token detection"""
    try:
        msg = msg.lower().strip()
        
        # Handle contract addresses (including Solana format)
        words = msg.split()
        for word in words:
            # Clean the word
            word = word.strip('?!.,')
            
            # Check for long alphanumeric strings (likely contract addresses)
            if len(word) > 30 and word.isalnum():
                logger.info(f"Found contract address: {word}")
                return word
            
            # Check for 0x addresses
            if word.startswith('0x'):
                logger.info(f"Found 0x address: {word}")
                return word
        
        # Handle $ prefixed tokens
        if '$' in msg:
            token = msg[msg.find('$')+1:].split()[0].strip()
            logger.info(f"Found $ prefixed token: {token}")
            return token
        
        # Handle "price of" format
        if 'price of' in msg:
            token = msg.split('price of')[1].strip()
            logger.info(f"Found 'price of' format: {token}")
            return token
            
        # Handle "price for" format
        if 'price for' in msg:
            token = msg.split('price for')[1].strip()
            logger.info(f"Found 'price for' format: {token}")
            return token

        # Handle general case
        for word in words:
            word = word.strip('?!.,')
            if word.isalnum() and word not in ['price', 'of', 'for', 'the']:
                logger.info(f"Found token from general text: {word}")
                return word
                
        return None
    except Exception as e:
        logger.error(f"Error in extract_token: {str(e)}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.json
        message = data.get('message', '').strip()
        print(f"Received message: {message}")  # Debug print
        logger.info(f"Received message: {message}")
        
        # Extract token from message
        token_id = extract_token(message)
        print(f"Extracted token ID: {token_id}")  # Debug print
        logger.info(f"Extracted token: {token_id}")
        
        if token_id:
            # Get token data asynchronously
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            analysis_data = loop.run_until_complete(get_token_data(token_id))
            print(f"Analysis data: {analysis_data}")  # Debug print
            loop.close()
            
            logger.info(f"Analysis data: {analysis_data}")

            if analysis_data:
                response = format_analysis(analysis_data)
                print(f"Formatted response: {response}")  # Debug print
                return jsonify({"response": response})
            else:
                print("No analysis data found")  # Debug print
                return jsonify({"response": "token giving me anxiety... can't find it anywhere... like my will to live..."})

        return jsonify({"response": get_random_response()})

    except Exception as e:
        print(f"Error in ask route: {str(e)}")  # Debug print
        logger.error(f"Error processing request: {str(e)}")
        logger.exception(e)  # This will print the full stack trace
        return jsonify({"response": "having a mental breakdown... try again later... like my trading career..."})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
