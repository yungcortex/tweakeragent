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
    """Get data from DexScreener"""
    try:
        # Handle both contract addresses and token symbols
        search_param = token_id if token_id.startswith('0x') else f"search?q={token_id}"
        async with session.get(f"{APIS['dexscreener']['url']}/pairs/{search_param}") as resp:
            if resp.status == 200:
                data = await resp.json()
                if data.get('pairs') and len(data['pairs']) > 0:
                    pair = data['pairs'][0]
                    return {
                        'price': float(pair['priceUsd']),
                        'change_24h': float(pair['priceChange']['h24']),
                        'volume_24h': float(pair['volume']['h24']),
                        'source': 'dexscreener',
                        'extra': {
                            'liquidity': pair.get('liquidity', {}).get('usd', 0),
                            'dex': pair.get('dexId', 'unknown')
                        }
                    }
    except Exception as e:
        logger.error(f"DexScreener API error: {str(e)}")
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
        async with session.get(f"{APIS['pump_fun']['url']}/tokens/{token_id}") as resp:
            if resp.status == 200:
                data = await resp.json()
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
        logger.error(f"Pump.fun API error: {str(e)}")
    return None

async def get_token_data(token_id: str) -> Optional[Dict[str, Any]]:
    """Get token data from multiple sources with fallbacks"""
    async with aiohttp.ClientSession() as session:
        tasks = []
        
        # Always try PumpFun and DexScreener first for any token
        tasks.extend([
            get_pumpfun_data(session, token_id),
            get_dexscreener_data(session, token_id)
        ])

        # For major tokens, also try Binance
        if token_id.upper() in ['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'DOGE']:
            tasks.append(get_binance_data(session, token_id))

        # Try CoinGecko and DefiLlama as fallbacks
        tasks.extend([
            get_coingecko_data(session, token_id),
            get_defillama_data(session, token_id)
        ])

        # Wait for all API calls to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out errors and None results
        valid_results = [r for r in results if isinstance(r, dict)]

        if valid_results:
            # Prioritize sources in this order: PumpFun > DexScreener > Others
            for source in ['pump_fun', 'dexscreener']:
                source_data = next((r for r in valid_results if r['source'] == source), None)
                if source_data:
                    source_data['sources'] = [r['source'] for r in valid_results]
                    return source_data

            # If no preferred source, use the first available with combined sources
            combined_data = valid_results[0]
            combined_data['sources'] = [r['source'] for r in valid_results]
            
            # Average the prices if we have multiple sources
            if len(valid_results) > 1:
                prices = [r['price'] for r in valid_results if 'price' in r]
                combined_data['price'] = sum(prices) / len(prices)

            return combined_data

        return None

def format_analysis(analysis_data):
    """Format market analysis according to character style"""
    try:
        # Extract basic data
        price = analysis_data.get('price', 0)
        change = analysis_data.get('change_24h', 0)
        volume = analysis_data.get('volume_24h', 0)
        sources = analysis_data.get('sources', [analysis_data.get('source', 'unknown')])
        extra = analysis_data.get('extra', {})

        # Enhanced random phrases
        manic_states = [
            "manic episode", "psychological breakdown", "existential crisis",
            "therapy session", "mental relapse", "dissociative state",
            "panic attack", "dopamine rush", "trading ptsd", "chart anxiety",
            "market trauma", "hopium overdose"
        ]

        addiction_comparisons = [
            "caffeine addiction", "doomscrolling habit", "trading addiction",
            "chart obsession", "hopium dependency", "fomo syndrome",
            "leverage addiction", "rug pull ptsd", "degen gambling",
            "yield farming obsession", "airdrop addiction"
        ]

        relationship_metaphors = [
            "my relationships", "my commitment issues", "my trust issues",
            "my emotional stability", "my social life", "my portfolio",
            "my trading history", "my liquidation history", "my leverage habits",
            "my wallet balance", "my trading psychology"
        ]

        market_conditions = [
            "paper hands crying", "diamond hands forming", "bears in therapy",
            "bulls on steroids", "whales having mood swings", "paper hands folding",
            "market makers in rehab", "liquidity pools evaporating"
        ]

        # Create dynamic response with more variety
        state = random.choice(manic_states)
        addiction = random.choice(addiction_comparisons)
        relationship = random.choice(relationship_metaphors)
        condition = random.choice(market_conditions)

        # Format with random variations and more technical details
        response_parts = [
            f"having another {state}... {'up' if change > 0 else 'down'} {abs(change):.2f}% like my serotonin levels",
            f"volume at ${volume:,.2f} - more unstable than my {addiction}",
            f"price ${price:.8f} giving me flashbacks",
            f"market sentiment more volatile than {relationship}",
            f"{condition} while {sources[0]} shows signs of {random.choice(manic_states)}"
        ]

        # Add extra insights if available
        if extra.get('liquidity'):
            response_parts.append(f"liquidity pool deeper than my emotional issues: ${extra['liquidity']:,.2f}")
        if extra.get('dex'):
            response_parts.append(f"trading on {extra['dex']} like my life depends on it")

        # Add multi-source commentary
        if len(sources) > 1:
            response_parts.append(f"cross-referencing {', '.join(sources)} like my multiple personalities")

        # Randomize order and select 3-4 parts
        random.shuffle(response_parts)
        selected_parts = response_parts[:random.randint(3, 4)]

        return " ... ".join(selected_parts).lower()

    except Exception as e:
        logger.error(f"Error formatting analysis: {str(e)}")
        return "having a breakdown... give me a minute..."

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
    msg = msg.lower().strip()
    words = msg.split()
    
    for word in words:
        # Clean the word
        word = word.strip('?!.,')
        
        # Contract address
        if word.startswith('0x'):
            return word
        
        # $ symbol
        if word.startswith('$'):
            return word[1:]
        
        # Remove common prefixes
        if word.startswith('price'):
            continue
        if word in ['of', 'for', 'the']:
            continue
            
        # Check for token symbols
        if word.isalnum():  # Basic check for token symbols
            return word
            
    # Check for specific patterns
    if 'price of' in msg:
        parts = msg.split('price of')
        if len(parts) > 1:
            return parts[1].strip()
            
    if '/analyze' in msg:
        parts = msg.split('/analyze')
        if len(parts) > 1:
            return parts[1].strip()
            
    return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.json
        message = data.get('message', '').strip().lower()
        
        # Extract token from message
        token_id = extract_token(message)
        
        if token_id:
            # Get token data asynchronously
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            analysis_data = loop.run_until_complete(get_token_data(token_id))
            loop.close()

            if analysis_data:
                response = format_analysis(analysis_data)
                return jsonify({"response": response})
            else:
                return jsonify({"response": "token giving me anxiety... can't find it anywhere... like my will to live..."})

        return jsonify({"response": get_random_response()})

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({"response": "having a mental breakdown... try again later... like my trading career..."})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
