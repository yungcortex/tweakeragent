import requests
import time
import os
from flask import request, jsonify

def get_coin_data_by_id_or_address(identifier):
    try:
        logger.info(f"Getting data for {identifier}")

        # DexScreener as primary API
        try:
            # Common token addresses mapping
            contract_mapping = {
                'btc': '0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599',  # WBTC
                'eth': '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2',  # WETH
                'bnb': '0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c',  # WBNB
                'sol': '0x570A5D26f7765Ecb712C0924E4De545B89fD43dF',  # SOL
                'xrp': '0x1D2F0da169ceB9fC7B3144628dB156f3F6c60dBE',  # XRP
                'doge': '0xbA2aE424d960c26247Dd6c32edC70B295c744C43',  # DOGE
                'ada': '0x3EE2200Efb3400fAbB9AacF31297cBdD1d435D47',  # ADA
                'matic': '0xCC42724C6683B7E57334c4E856f4c9965ED682bD'  # MATIC
            }

            # Use contract address if it's a known token
            search_term = contract_mapping.get(identifier.lower(), identifier)

            # DexScreener API endpoint
            dex_url = f"https://api.dexscreener.com/latest/dex/search?q={search_term}"
            response = requests.get(dex_url, timeout=10)

            if response.status_code == 200:
                data = response.json()
                if data.get('pairs') and len(data['pairs']) > 0:
                    # Get the most liquid pair
                    pair = max(data['pairs'], key=lambda x: float(x.get('liquidity', {}).get('usd', 0)))

                    # Create historical data points
                    current_time = int(time.time() * 1000)
                    prices = []
                    volumes = []

                    # Use priceChange and volumeChange to create synthetic history
                    price_now = float(pair['priceUsd'])
                    volume_now = float(pair.get('volume', {}).get('h24', 0))
                    price_change = float(pair.get('priceChange', {}).get('h24', 0))

                    for i in range(30):
                        timestamp = current_time - (i * 24 * 60 * 60 * 1000)
                        # Calculate historical price based on current price and change percentage
                        historical_price = price_now / (1 + (price_change * (30 - i) / 100 / 30))
                        historical_volume = volume_now / 30  # Distribute volume evenly

                        prices.insert(0, [timestamp, historical_price])
                        volumes.insert(0, [timestamp, historical_volume])

                    return {
                        'name': pair.get('baseToken', {}).get('symbol', identifier.upper()),
                        'price': price_now,
                        'change_24h': price_change,
                        'volume_24h': volume_now,
                        'history': prices,
                        'volume_history': volumes,
                        'source': 'dexscreener',
                        'liquidity': float(pair.get('liquidity', {}).get('usd', 0)),
                        'fdv': float(pair.get('fdv', 0))
                    }

        except Exception as e:
            logger.error(f"DexScreener API error: {str(e)}")

        # Fallback to Binance
        try:
            symbol = identifier.upper() + "USDT"
            binance_url = f"https://api.binance.com/api/v3"

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

        except Exception as e:
            logger.error(f"Binance API error: {str(e)}")

        # Final fallback to CoinGecko
        try:
            coin_mapping = {
                'btc': 'bitcoin',
                'eth': 'ethereum',
                'bnb': 'binancecoin',
                'sol': 'solana',
                'xrp': 'ripple',
                'doge': 'dogecoin'
            }

            coin_id = coin_mapping.get(identifier.lower(), identifier.lower())
            gecko_url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days=30&interval=daily"

            response = requests.get(gecko_url, timeout=5)
            if response.status_code == 200:
                data = response.json()

                price_url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd&include_24hr_change=true&include_24hr_vol=true"
                price_response = requests.get(price_url, timeout=5)

                if price_response.status_code == 200:
                    price_data = price_response.json()[coin_id]

                    return {
                        'name': coin_id.upper(),
                        'price': price_data['usd'],
                        'change_24h': price_data.get('usd_24h_change', 0),
                        'volume_24h': price_data.get('usd_24h_vol', 0),
                        'history': data['prices'],
                        'volume_history': data['total_volumes'],
                        'source': 'coingecko'
                    }

        except Exception as e:
            logger.error(f"CoinGecko API error: {str(e)}")

        return None

    except Exception as e:
        logger.error(f"Error in get_coin_data_by_id_or_address: {str(e)}")
        return None

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.json
        user_input = data.get('message', '').strip().lower()

        if user_input.startswith('/analyze '):
            identifier = user_input.split(' ', 1)[1].strip()
            logger.info(f"Analyzing {identifier}")

            try:
                # Test API response
                dex_url = f"https://api.dexscreener.com/latest/dex/search?q={identifier}"
                response = requests.get(dex_url, timeout=10)
                logger.info(f"DexScreener Status: {response.status_code}")
                logger.info(f"DexScreener Response: {response.json()}")

                # Rest of your code...

            except Exception as e:
                logger.error(f"Analysis error: {str(e)}")
                return jsonify({"response": f"Debug: {str(e)}"})

    except Exception as e:
        logger.error(f"Route error: {str(e)}")
        return jsonify({"response": f"Debug: {str(e)}"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

# Test the API directly
response = requests.get("https://api.dexscreener.com/latest/dex/search?q=btc")
print(response.status_code)
print(response.json())
