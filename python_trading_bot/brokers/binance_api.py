#!/usr/bin/env python3
"""
Binance API Integration - Complete implementation
"""

import requests
import hmac
import hashlib
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import logging
from urllib.parse import urlencode
import websocket
import threading

logger = logging.getLogger(__name__)

class BinanceAPI:
    """Complete Binance API implementation"""
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        
        if testnet:
            self.base_url = "https://testnet.binance.vision/api"
            self.ws_base_url = "wss://testnet.binance.vision/ws"
        else:
            self.base_url = "https://api.binance.com/api"
            self.ws_base_url = "wss://stream.binance.com:9443/ws"
        
        self.session = requests.Session()
        self.session.headers.update({
            'X-MBX-APIKEY': api_key,
            'Content-Type': 'application/json'
        })
        
        self.ws_connections = {}
        self.is_connected = False
        
    def _generate_signature(self, params: Dict) -> str:
        """Generate signature for authenticated requests"""
        query_string = urlencode(params)
        return hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def _make_request(self, method: str, endpoint: str, params: Dict = None, signed: bool = False) -> Dict:
        """Make API request"""
        if params is None:
            params = {}
        
        if signed:
            params['timestamp'] = int(time.time() * 1000)
            params['signature'] = self._generate_signature(params)
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method == 'GET':
                response = self.session.get(url, params=params)
            elif method == 'POST':
                response = self.session.post(url, params=params)
            elif method == 'DELETE':
                response = self.session.delete(url, params=params)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            raise
    
    def get_server_time(self) -> Dict:
        """Get server time"""
        return self._make_request('GET', '/v3/time')
    
    def get_exchange_info(self) -> Dict:
        """Get exchange information"""
        return self._make_request('GET', '/v3/exchangeInfo')
    
    def get_account_info(self) -> Dict:
        """Get account information"""
        return self._make_request('GET', '/v3/account', signed=True)
    
    def get_balance(self, asset: str = None) -> Dict:
        """Get account balance"""
        account_info = self.get_account_info()
        balances = account_info['balances']
        
        if asset:
            for balance in balances:
                if balance['asset'] == asset:
                    return {
                        'asset': balance['asset'],
                        'free': float(balance['free']),
                        'locked': float(balance['locked']),
                        'total': float(balance['free']) + float(balance['locked'])
                    }
            return None
        
        return [
            {
                'asset': b['asset'],
                'free': float(b['free']),
                'locked': float(b['locked']),
                'total': float(b['free']) + float(b['locked'])
            }
            for b in balances if float(b['free']) > 0 or float(b['locked']) > 0
        ]
    
    def get_ticker_price(self, symbol: str = None) -> Dict:
        """Get ticker price"""
        params = {}
        if symbol:
            params['symbol'] = symbol
        
        return self._make_request('GET', '/v3/ticker/price', params)
    
    def get_ticker_24hr(self, symbol: str = None) -> Dict:
        """Get 24hr ticker statistics"""
        params = {}
        if symbol:
            params['symbol'] = symbol
        
        return self._make_request('GET', '/v3/ticker/24hr', params)
    
    def get_order_book(self, symbol: str, limit: int = 100) -> Dict:
        """Get order book"""
        params = {
            'symbol': symbol,
            'limit': limit
        }
        return self._make_request('GET', '/v3/depth', params)
    
    def get_recent_trades(self, symbol: str, limit: int = 500) -> List[Dict]:
        """Get recent trades"""
        params = {
            'symbol': symbol,
            'limit': limit
        }
        return self._make_request('GET', '/v3/trades', params)
    
    def get_klines(self, symbol: str, interval: str, start_time: datetime = None, 
                   end_time: datetime = None, limit: int = 500) -> pd.DataFrame:
        """Get kline/candlestick data"""
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        
        if start_time:
            params['startTime'] = int(start_time.timestamp() * 1000)
        if end_time:
            params['endTime'] = int(end_time.timestamp() * 1000)
        
        data = self._make_request('GET', '/v3/klines', params)
        
        # Convert to DataFrame
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convert data types
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_columns] = df[numeric_columns].astype(float)
        
        df.set_index('timestamp', inplace=True)
        return df[numeric_columns]
    
    def place_order(self, symbol: str, side: str, order_type: str, quantity: float,
                   price: float = None, stop_price: float = None, 
                   time_in_force: str = 'GTC', new_client_order_id: str = None) -> Dict:
        """Place a new order"""
        params = {
            'symbol': symbol,
            'side': side.upper(),
            'type': order_type.upper(),
            'quantity': quantity
        }
        
        if price:
            params['price'] = price
        if stop_price:
            params['stopPrice'] = stop_price
        if time_in_force:
            params['timeInForce'] = time_in_force
        if new_client_order_id:
            params['newClientOrderId'] = new_client_order_id
        
        return self._make_request('POST', '/v3/order', params, signed=True)
    
    def place_market_order(self, symbol: str, side: str, quantity: float) -> Dict:
        """Place a market order"""
        return self.place_order(symbol, side, 'MARKET', quantity)
    
    def place_limit_order(self, symbol: str, side: str, quantity: float, price: float) -> Dict:
        """Place a limit order"""
        return self.place_order(symbol, side, 'LIMIT', quantity, price=price)
    
    def place_stop_loss_order(self, symbol: str, side: str, quantity: float, stop_price: float) -> Dict:
        """Place a stop loss order"""
        return self.place_order(symbol, side, 'STOP_LOSS_LIMIT', quantity, 
                               price=stop_price, stop_price=stop_price)
    
    def cancel_order(self, symbol: str, order_id: int = None, orig_client_order_id: str = None) -> Dict:
        """Cancel an order"""
        params = {'symbol': symbol}
        
        if order_id:
            params['orderId'] = order_id
        elif orig_client_order_id:
            params['origClientOrderId'] = orig_client_order_id
        else:
            raise ValueError("Either order_id or orig_client_order_id must be provided")
        
        return self._make_request('DELETE', '/v3/order', params, signed=True)
    
    def get_order(self, symbol: str, order_id: int = None, orig_client_order_id: str = None) -> Dict:
        """Get order status"""
        params = {'symbol': symbol}
        
        if order_id:
            params['orderId'] = order_id
        elif orig_client_order_id:
            params['origClientOrderId'] = orig_client_order_id
        else:
            raise ValueError("Either order_id or orig_client_order_id must be provided")
        
        return self._make_request('GET', '/v3/order', params, signed=True)
    
    def get_open_orders(self, symbol: str = None) -> List[Dict]:
        """Get all open orders"""
        params = {}
        if symbol:
            params['symbol'] = symbol
        
        return self._make_request('GET', '/v3/openOrders', params, signed=True)
    
    def get_all_orders(self, symbol: str, order_id: int = None, start_time: datetime = None,
                      end_time: datetime = None, limit: int = 500) -> List[Dict]:
        """Get all orders for a symbol"""
        params = {
            'symbol': symbol,
            'limit': limit
        }
        
        if order_id:
            params['orderId'] = order_id
        if start_time:
            params['startTime'] = int(start_time.timestamp() * 1000)
        if end_time:
            params['endTime'] = int(end_time.timestamp() * 1000)
        
        return self._make_request('GET', '/v3/allOrders', params, signed=True)
    
    def get_my_trades(self, symbol: str, start_time: datetime = None, 
                     end_time: datetime = None, limit: int = 500) -> List[Dict]:
        """Get account trade list"""
        params = {
            'symbol': symbol,
            'limit': limit
        }
        
        if start_time:
            params['startTime'] = int(start_time.timestamp() * 1000)
        if end_time:
            params['endTime'] = int(end_time.timestamp() * 1000)
        
        return self._make_request('GET', '/v3/myTrades', params, signed=True)
    
    def start_user_data_stream(self) -> Dict:
        """Start a new user data stream"""
        return self._make_request('POST', '/v3/userDataStream')
    
    def keepalive_user_data_stream(self, listen_key: str) -> Dict:
        """Keepalive a user data stream"""
        params = {'listenKey': listen_key}
        return self._make_request('PUT', '/v3/userDataStream', params)
    
    def close_user_data_stream(self, listen_key: str) -> Dict:
        """Close a user data stream"""
        params = {'listenKey': listen_key}
        return self._make_request('DELETE', '/v3/userDataStream', params)
    
    def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol"""
        try:
            ticker = self.get_ticker_price(symbol)
            return float(ticker['price'])
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return 0.0
    
    def supports_symbol(self, symbol: str) -> bool:
        """Check if broker supports a symbol"""
        try:
            exchange_info = self.get_exchange_info()
            symbols = [s['symbol'] for s in exchange_info['symbols']]
            return symbol.upper() in symbols
        except Exception as e:
            logger.error(f"Error checking symbol support: {e}")
            return False
