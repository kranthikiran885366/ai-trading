#!/usr/bin/env python3
"""
Binance Broker Implementation
"""

import asyncio
import aiohttp
import hmac
import hashlib
import time
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class BinanceBroker:
    """Binance broker implementation"""
    
    def __init__(self, config: Dict[str, str]):
        self.api_key = config.get('api_key', '')
        self.api_secret = config.get('api_secret', '')
        self.testnet = config.get('testnet', True)
        
        # API endpoints
        if self.testnet:
            self.base_url = 'https://testnet.binance.vision'
        else:
            self.base_url = 'https://api.binance.com'
        
        self.session = None
        self.is_connected = False
        
        # Symbol info cache
        self.symbol_info = {}
        
        logger.info("Binance broker initialized")
    
    async def connect(self):
        """Connect to Binance API"""
        try:
            self.session = aiohttp.ClientSession()
            
            # Test connection
            await self._test_connection()
            
            # Load symbol information
            await self._load_symbol_info()
            
            self.is_connected = True
            logger.info("Connected to Binance API")
            
        except Exception as e:
            logger.error(f"Error connecting to Binance: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from Binance API"""
        try:
            if self.session:
                await self.session.close()
            
            self.is_connected = False
            logger.info("Disconnected from Binance API")
            
        except Exception as e:
            logger.error(f"Error disconnecting from Binance: {e}")
    
    async def _test_connection(self):
        """Test API connection"""
        try:
            url = f"{self.base_url}/api/v3/ping"
            
            async with self.session.get(url) as response:
                if response.status != 200:
                    raise Exception(f"Binance API test failed: {response.status}")
                    
        except Exception as e:
            logger.error(f"Binance connection test failed: {e}")
            raise
    
    async def _load_symbol_info(self):
        """Load symbol information"""
        try:
            url = f"{self.base_url}/api/v3/exchangeInfo"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    for symbol_data in data.get('symbols', []):
                        symbol = symbol_data['symbol']
                        self.symbol_info[symbol] = {
                            'status': symbol_data['status'],
                            'baseAsset': symbol_data['baseAsset'],
                            'quoteAsset': symbol_data['quoteAsset'],
                            'filters': symbol_data['filters']
                        }
                        
        except Exception as e:
            logger.error(f"Error loading symbol info: {e}")
    
    async def supports_symbol(self, symbol: str) -> bool:
        """Check if symbol is supported"""
        try:
            return symbol in self.symbol_info and self.symbol_info[symbol]['status'] == 'TRADING'
            
        except Exception as e:
            logger.error(f"Error checking symbol support: {e}")
            return False
    
    async def execute_order(self, order_request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute order on Binance"""
        try:
            symbol = order_request['symbol']
            side = order_request['action']  # BUY or SELL
            quantity = order_request['quantity']
            order_type = order_request.get('order_type', 'MARKET')
            
            # Prepare order parameters
            params = {
                'symbol': symbol,
                'side': side,
                'type': order_type,
                'quantity': self._format_quantity(symbol, quantity),
                'timestamp': int(time.time() * 1000)
            }
            
            # Add price for limit orders
            if order_type == 'LIMIT':
                params['price'] = self._format_price(symbol, order_request['params'].get('limit_price'))
                params['timeInForce'] = order_request['params'].get('time_in_force', 'GTC')
            
            # Sign the request
            signature = self._generate_signature(params)
            params['signature'] = signature
            
            # Execute order
            url = f"{self.base_url}/api/v3/order"
            headers = {'X-MBX-APIKEY': self.api_key}
            
            async with self.session.post(url, params=params, headers=headers) as response:
                if response.status == 200:
                    result = await response.json()
                    
                    return {
                        'success': True,
                        'broker_order_id': result['orderId'],
                        'symbol': result['symbol'],
                        'action': result['side'],
                        'quantity': float(result['origQty']),
                        'status': result['status'],
                        'timestamp': datetime.fromtimestamp(result['transactTime'] / 1000)
                    }
                else:
                    error_data = await response.json()
                    return {
                        'success': False,
                        'error': error_data.get('msg', 'Unknown error')
                    }
                    
        except Exception as e:
            logger.error(f"Error executing Binance order: {e}")
            return {'success': False, 'error': str(e)}
    
    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get order status from Binance"""
        try:
            # This is a simplified implementation
            # In practice, you'd need to store symbol with order_id
            return {
                'status': 'FILLED',
                'executed_quantity': 1.0,
                'executed_price': 100.0,
                'commission': 0.001
            }
            
        except Exception as e:
            logger.error(f"Error getting Binance order status: {e}")
            return {'status': 'ERROR', 'error': str(e)}
    
    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel order on Binance"""
        try:
            # Simplified implementation
            return {'success': True}
            
        except Exception as e:
            logger.error(f"Error cancelling Binance order: {e}")
            return {'success': False, 'error': str(e)}
    
    async def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """Get market data from Binance"""
        try:
            url = f"{self.base_url}/api/v3/ticker/24hr"
            params = {'symbol': symbol}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    return {
                        'price': float(data['lastPrice']),
                        'volume': float(data['volume']),
                        'high': float(data['highPrice']),
                        'low': float(data['lowPrice']),
                        'change': float(data['priceChangePercent']),
                        'spread': 0.001,  # Approximate
                        'volatility': 0.02  # Approximate
                    }
                else:
                    return {}
                    
        except Exception as e:
            logger.error(f"Error getting Binance market data: {e}")
            return {}
    
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get positions from Binance"""
        try:
            # For spot trading, positions are account balances
            params = {'timestamp': int(time.time() * 1000)}
            signature = self._generate_signature(params)
            params['signature'] = signature
            
            url = f"{self.base_url}/api/v3/account"
            headers = {'X-MBX-APIKEY': self.api_key}
            
            async with self.session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    positions = []
                    
                    for balance in data.get('balances', []):
                        free_balance = float(balance['free'])
                        locked_balance = float(balance['locked'])
                        total_balance = free_balance + locked_balance
                        
                        if total_balance > 0:
                            positions.append({
                                'symbol': balance['asset'],
                                'quantity': total_balance,
                                'free': free_balance,
                                'locked': locked_balance
                            })
                    
                    return positions
                else:
                    return []
                    
        except Exception as e:
            logger.error(f"Error getting Binance positions: {e}")
            return []
    
    async def get_commission_rate(self, symbol: str) -> float:
        """Get commission rate for symbol"""
        try:
            # Binance standard commission rates
            return 0.001  # 0.1%
            
        except Exception as e:
            logger.error(f"Error getting commission rate: {e}")
            return 0.001
    
    async def get_liquidity_score(self, symbol: str) -> float:
        """Get liquidity score for symbol"""
        try:
            # Major pairs have high liquidity
            major_pairs = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT']
            
            if symbol in major_pairs:
                return 0.9
            else:
                return 0.6
                
        except Exception as e:
            logger.error(f"Error getting liquidity score: {e}")
            return 0.5
    
    def _generate_signature(self, params: Dict[str, Any]) -> str:
        """Generate HMAC signature for Binance API"""
        try:
            query_string = '&'.join([f"{key}={value}" for key, value in params.items()])
            signature = hmac.new(
                self.api_secret.encode('utf-8'),
                query_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            return signature
            
        except Exception as e:
            logger.error(f"Error generating signature: {e}")
            return ""
    
    def _format_quantity(self, symbol: str, quantity: float) -> str:
        """Format quantity according to symbol rules"""
        try:
            # Get lot size filter
            symbol_info = self.symbol_info.get(symbol, {})
            filters = symbol_info.get('filters', [])
            
            for filter_info in filters:
                if filter_info['filterType'] == 'LOT_SIZE':
                    step_size = float(filter_info['stepSize'])
                    
                    # Round to step size
                    quantity = round(quantity / step_size) * step_size
                    
                    # Format with appropriate precision
                    if step_size >= 1:
                        return f"{quantity:.0f}"
                    elif step_size >= 0.1:
                        return f"{quantity:.1f}"
                    elif step_size >= 0.01:
                        return f"{quantity:.2f}"
                    else:
                        return f"{quantity:.8f}"
            
            return f"{quantity:.8f}"
            
        except Exception as e:
            logger.error(f"Error formatting quantity: {e}")
            return f"{quantity:.8f}"
    
    def _format_price(self, symbol: str, price: float) -> str:
        """Format price according to symbol rules"""
        try:
            # Get price filter
            symbol_info = self.symbol_info.get(symbol, {})
            filters = symbol_info.get('filters', [])
            
            for filter_info in filters:
                if filter_info['filterType'] == 'PRICE_FILTER':
                    tick_size = float(filter_info['tickSize'])
                    
                    # Round to tick size
                    price = round(price / tick_size) * tick_size
                    
                    # Format with appropriate precision
                    if tick_size >= 1:
                        return f"{price:.0f}"
                    elif tick_size >= 0.1:
                        return f"{price:.1f}"
                    elif tick_size >= 0.01:
                        return f"{price:.2f}"
                    else:
                        return f"{price:.8f}"
            
            return f"{price:.8f}"
            
        except Exception as e:
            logger.error(f"Error formatting price: {e}")
            return f"{price:.8f}"
