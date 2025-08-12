#!/usr/bin/env python3
"""
Alpaca Broker Implementation for Stock Trading
"""

import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class AlpacaBroker:
    """Alpaca broker implementation for stocks"""
    
    def __init__(self, config: Dict[str, str]):
        self.api_key = config.get('api_key', '')
        self.api_secret = config.get('api_secret', '')
        self.paper_trading = config.get('paper_trading', True)
        
        # API endpoints
        if self.paper_trading:
            self.base_url = 'https://paper-api.alpaca.markets'
            self.data_url = 'https://data.alpaca.markets'
        else:
            self.base_url = 'https://api.alpaca.markets'
            self.data_url = 'https://data.alpaca.markets'
        
        self.session = None
        self.is_connected = False
        
        logger.info("Alpaca broker initialized")
    
    async def connect(self):
        """Connect to Alpaca API"""
        try:
            headers = {
                'APCA-API-KEY-ID': self.api_key,
                'APCA-API-SECRET-KEY': self.api_secret
            }
            
            self.session = aiohttp.ClientSession(headers=headers)
            
            # Test connection
            await self._test_connection()
            
            self.is_connected = True
            logger.info("Connected to Alpaca API")
            
        except Exception as e:
            logger.error(f"Error connecting to Alpaca: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from Alpaca API"""
        try:
            if self.session:
                await self.session.close()
            
            self.is_connected = False
            logger.info("Disconnected from Alpaca API")
            
        except Exception as e:
            logger.error(f"Error disconnecting from Alpaca: {e}")
    
    async def _test_connection(self):
        """Test API connection"""
        try:
            url = f"{self.base_url}/v2/account"
            
            async with self.session.get(url) as response:
                if response.status != 200:
                    raise Exception(f"Alpaca API test failed: {response.status}")
                    
        except Exception as e:
            logger.error(f"Alpaca connection test failed: {e}")
            raise
    
    async def supports_symbol(self, symbol: str) -> bool:
        """Check if symbol is supported (stocks only)"""
        try:
            # Alpaca supports US stocks
            return not symbol.endswith('USDT') and len(symbol) <= 5
            
        except Exception as e:
            logger.error(f"Error checking symbol support: {e}")
            return False
    
    async def execute_order(self, order_request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute order on Alpaca"""
        try:
            symbol = order_request['symbol']
            side = order_request['action'].lower()  # buy or sell
            qty = order_request['quantity']
            order_type = order_request.get('order_type', 'market').lower()
            
            # Prepare order data
            order_data = {
                'symbol': symbol,
                'side': side,
                'type': order_type,
                'time_in_force': 'day'
            }
            
            # Handle quantity (Alpaca uses notional for fractional shares)
            if qty < 1:
                order_data['notional'] = qty * 100  # Approximate dollar amount
            else:
                order_data['qty'] = int(qty)
            
            # Add price for limit orders
            if order_type == 'limit':
                order_data['limit_price'] = order_request['params'].get('limit_price')
                order_data['time_in_force'] = order_request['params'].get('time_in_force', 'day')
            
            # Execute order
            url = f"{self.base_url}/v2/orders"
            
            async with self.session.post(url, json=order_data) as response:
                if response.status in [200, 201]:
                    result = await response.json()
                    
                    return {
                        'success': True,
                        'broker_order_id': result['id'],
                        'symbol': result['symbol'],
                        'action': result['side'].upper(),
                        'quantity': float(result.get('qty', 0)),
                        'status': result['status'],
                        'timestamp': datetime.fromisoformat(result['created_at'].replace('Z', '+00:00'))
                    }
                else:
                    error_data = await response.json()
                    return {
                        'success': False,
                        'error': error_data.get('message', 'Unknown error')
                    }
                    
        except Exception as e:
            logger.error(f"Error executing Alpaca order: {e}")
            return {'success': False, 'error': str(e)}
    
    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get order status from Alpaca"""
        try:
            url = f"{self.base_url}/v2/orders/{order_id}"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    return {
                        'status': data['status'].upper(),
                        'symbol': data['symbol'],
                        'action': data['side'].upper(),
                        'quantity': float(data.get('qty', 0)),
                        'executed_quantity': float(data.get('filled_qty', 0)),
                        'executed_price': float(data.get('filled_avg_price', 0)) if data.get('filled_avg_price') else 0,
                        'commission': 0.0  # Alpaca is commission-free
                    }
                else:
                    return {'status': 'ERROR', 'error': f'HTTP {response.status}'}
                    
        except Exception as e:
            logger.error(f"Error getting Alpaca order status: {e}")
            return {'status': 'ERROR', 'error': str(e)}
    
    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel order on Alpaca"""
        try:
            url = f"{self.base_url}/v2/orders/{order_id}"
            
            async with self.session.delete(url) as response:
                if response.status == 204:
                    return {'success': True}
                else:
                    error_data = await response.json()
                    return {
                        'success': False,
                        'error': error_data.get('message', 'Unknown error')
                    }
                    
        except Exception as e:
            logger.error(f"Error cancelling Alpaca order: {e}")
            return {'success': False, 'error': str(e)}
    
    async def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """Get market data from Alpaca"""
        try:
            url = f"{self.data_url}/v2/stocks/{symbol}/quotes/latest"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    quote = data.get('quote', {})
                    
                    bid = float(quote.get('bp', 0))
                    ask = float(quote.get('ap', 0))
                    price = (bid + ask) / 2 if bid > 0 and ask > 0 else 0
                    spread = ask - bid if bid > 0 and ask > 0 else 0
                    
                    return {
                        'price': price,
                        'bid': bid,
                        'ask': ask,
                        'spread': spread / price if price > 0 else 0,
                        'volume': 0,  # Would need separate call
                        'volatility': 0.02  # Approximate
                    }
                else:
                    return {}
                    
        except Exception as e:
            logger.error(f"Error getting Alpaca market data: {e}")
            return {}
    
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get positions from Alpaca"""
        try:
            url = f"{self.base_url}/v2/positions"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    positions_data = await response.json()
                    positions = []
                    
                    for pos in positions_data:
                        positions.append({
                            'symbol': pos['symbol'],
                            'quantity': float(pos['qty']),
                            'market_value': float(pos['market_value']),
                            'current_price': float(pos['current_price']),
                            'unrealized_pnl': float(pos['unrealized_pl']),
                            'avg_entry_price': float(pos['avg_entry_price'])
                        })
                    
                    return positions
                else:
                    return []
                    
        except Exception as e:
            logger.error(f"Error getting Alpaca positions: {e}")
            return []
    
    async def get_commission_rate(self, symbol: str) -> float:
        """Get commission rate for symbol"""
        try:
            # Alpaca is commission-free
            return 0.0
            
        except Exception as e:
            logger.error(f"Error getting commission rate: {e}")
            return 0.0
    
    async def get_liquidity_score(self, symbol: str) -> float:
        """Get liquidity score for symbol"""
        try:
            # Major stocks have high liquidity
            major_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META']
            
            if symbol in major_stocks:
                return 0.9
            else:
                return 0.7  # Most US stocks have decent liquidity
                
        except Exception as e:
            logger.error(f"Error getting liquidity score: {e}")
            return 0.5
