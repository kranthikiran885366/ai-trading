import requests
import hashlib
import json
from datetime import datetime, timedelta
import pandas as pd
import time
import logging
from urllib.parse import urlencode

class ZerodhaAPI:
    def __init__(self, api_key, api_secret, access_token=None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.access_token = access_token
        self.base_url = "https://api.kite.trade"
        self.session = requests.Session()
        self.session.headers.update({
            'X-Kite-Version': '3',
            'User-Agent': 'Trading Bot v1.0'
        })
        
        if access_token:
            self.session.headers.update({
                'Authorization': f'token {api_key}:{access_token}'
            })
        
        self.logger = logging.getLogger(__name__)
    
    def generate_session(self, request_token):
        """Generate session using request token"""
        try:
            checksum = hashlib.sha256(
                (self.api_key + request_token + self.api_secret).encode()
            ).hexdigest()
            
            data = {
                'api_key': self.api_key,
                'request_token': request_token,
                'checksum': checksum
            }
            
            response = self.session.post(f"{self.base_url}/session/token", data=data)
            result = response.json()
            
            if response.status_code == 200:
                self.access_token = result['data']['access_token']
                self.session.headers.update({
                    'Authorization': f'token {self.api_key}:{self.access_token}'
                })
                return result['data']
            else:
                raise Exception(f"Session generation failed: {result}")
                
        except Exception as e:
            self.logger.error(f"Error generating session: {e}")
            raise
    
    def get_profile(self):
        """Get user profile"""
        try:
            response = self.session.get(f"{self.base_url}/user/profile")
            return self._handle_response(response)
        except Exception as e:
            self.logger.error(f"Error getting profile: {e}")
            raise
    
    def get_margins(self):
        """Get account margins"""
        try:
            response = self.session.get(f"{self.base_url}/user/margins")
            return self._handle_response(response)
        except Exception as e:
            self.logger.error(f"Error getting margins: {e}")
            raise
    
    def get_positions(self):
        """Get current positions"""
        try:
            response = self.session.get(f"{self.base_url}/portfolio/positions")
            return self._handle_response(response)
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            raise
    
    def get_holdings(self):
        """Get current holdings"""
        try:
            response = self.session.get(f"{self.base_url}/portfolio/holdings")
            return self._handle_response(response)
        except Exception as e:
            self.logger.error(f"Error getting holdings: {e}")
            raise
    
    def place_order(self, variety, exchange, tradingsymbol, transaction_type, 
                   quantity, product, order_type, price=None, trigger_price=None,
                   validity="DAY", disclosed_quantity=None, tag=None):
        """Place an order"""
        try:
            data = {
                'variety': variety,
                'exchange': exchange,
                'tradingsymbol': tradingsymbol,
                'transaction_type': transaction_type,
                'quantity': quantity,
                'product': product,
                'order_type': order_type,
                'validity': validity
            }
            
            if price:
                data['price'] = price
            if trigger_price:
                data['trigger_price'] = trigger_price
            if disclosed_quantity:
                data['disclosed_quantity'] = disclosed_quantity
            if tag:
                data['tag'] = tag
            
            response = self.session.post(f"{self.base_url}/orders/{variety}", data=data)
            return self._handle_response(response)
            
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            raise
    
    def modify_order(self, variety, order_id, quantity=None, price=None, 
                    order_type=None, trigger_price=None, validity=None,
                    disclosed_quantity=None):
        """Modify an existing order"""
        try:
            data = {}
            
            if quantity:
                data['quantity'] = quantity
            if price:
                data['price'] = price
            if order_type:
                data['order_type'] = order_type
            if trigger_price:
                data['trigger_price'] = trigger_price
            if validity:
                data['validity'] = validity
            if disclosed_quantity:
                data['disclosed_quantity'] = disclosed_quantity
            
            response = self.session.put(f"{self.base_url}/orders/{variety}/{order_id}", data=data)
            return self._handle_response(response)
            
        except Exception as e:
            self.logger.error(f"Error modifying order: {e}")
            raise
    
    def cancel_order(self, variety, order_id):
        """Cancel an order"""
        try:
            response = self.session.delete(f"{self.base_url}/orders/{variety}/{order_id}")
            return self._handle_response(response)
        except Exception as e:
            self.logger.error(f"Error cancelling order: {e}")
            raise
    
    def get_orders(self):
        """Get all orders for the day"""
        try:
            response = self.session.get(f"{self.base_url}/orders")
            return self._handle_response(response)
        except Exception as e:
            self.logger.error(f"Error getting orders: {e}")
            raise
    
    def get_order_history(self, order_id):
        """Get order history"""
        try:
            response = self.session.get(f"{self.base_url}/orders/{order_id}")
            return self._handle_response(response)
        except Exception as e:
            self.logger.error(f"Error getting order history: {e}")
            raise
    
    def get_trades(self):
        """Get all trades for the day"""
        try:
            response = self.session.get(f"{self.base_url}/trades")
            return self._handle_response(response)
        except Exception as e:
            self.logger.error(f"Error getting trades: {e}")
            raise
    
    def get_instruments(self, exchange=None):
        """Get instruments list"""
        try:
            url = f"{self.base_url}/instruments"
            if exchange:
                url += f"/{exchange}"
            
            response = self.session.get(url)
            
            if response.status_code == 200:
                # Parse CSV response
                from io import StringIO
                df = pd.read_csv(StringIO(response.text))
                return df
            else:
                raise Exception(f"Failed to get instruments: {response.text}")
                
        except Exception as e:
            self.logger.error(f"Error getting instruments: {e}")
            raise
    
    def get_quote(self, instruments):
        """Get quote for instruments"""
        try:
            if isinstance(instruments, str):
                instruments = [instruments]
            
            params = {'i': instruments}
            response = self.session.get(f"{self.base_url}/quote", params=params)
            return self._handle_response(response)
            
        except Exception as e:
            self.logger.error(f"Error getting quote: {e}")
            raise
    
    def get_ohlc(self, instruments):
        """Get OHLC data for instruments"""
        try:
            if isinstance(instruments, str):
                instruments = [instruments]
            
            params = {'i': instruments}
            response = self.session.get(f"{self.base_url}/quote/ohlc", params=params)
            return self._handle_response(response)
            
        except Exception as e:
            self.logger.error(f"Error getting OHLC: {e}")
            raise
    
    def get_ltp(self, instruments):
        """Get last traded price for instruments"""
        try:
            if isinstance(instruments, str):
                instruments = [instruments]
            
            params = {'i': instruments}
            response = self.session.get(f"{self.base_url}/quote/ltp", params=params)
            return self._handle_response(response)
            
        except Exception as e:
            self.logger.error(f"Error getting LTP: {e}")
            raise
    
    def get_historical_data(self, instrument_token, from_date, to_date, interval):
        """Get historical data"""
        try:
            params = {
                'from': from_date.strftime('%Y-%m-%d'),
                'to': to_date.strftime('%Y-%m-%d'),
                'interval': interval
            }
            
            response = self.session.get(
                f"{self.base_url}/instruments/historical/{instrument_token}",
                params=params
            )
            
            result = self._handle_response(response)
            
            # Convert to DataFrame
            if result and 'data' in result:
                df = pd.DataFrame(result['data']['candles'])
                df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                return df
            
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error getting historical data: {e}")
            raise
    
    def convert_position(self, exchange, tradingsymbol, transaction_type, 
                        position_type, quantity, old_product, new_product):
        """Convert position product type"""
        try:
            data = {
                'exchange': exchange,
                'tradingsymbol': tradingsymbol,
                'transaction_type': transaction_type,
                'position_type': position_type,
                'quantity': quantity,
                'old_product': old_product,
                'new_product': new_product
            }
            
            response = self.session.put(f"{self.base_url}/portfolio/positions", data=data)
            return self._handle_response(response)
            
        except Exception as e:
            self.logger.error(f"Error converting position: {e}")
            raise
    
    def get_mf_orders(self):
        """Get mutual fund orders"""
        try:
            response = self.session.get(f"{self.base_url}/mf/orders")
            return self._handle_response(response)
        except Exception as e:
            self.logger.error(f"Error getting MF orders: {e}")
            raise
    
    def place_mf_order(self, tradingsymbol, transaction_type, amount=None, quantity=None, tag=None):
        """Place mutual fund order"""
        try:
            data = {
                'tradingsymbol': tradingsymbol,
                'transaction_type': transaction_type
            }
            
            if amount:
                data['amount'] = amount
            if quantity:
                data['quantity'] = quantity
            if tag:
                data['tag'] = tag
            
            response = self.session.post(f"{self.base_url}/mf/orders", data=data)
            return self._handle_response(response)
            
        except Exception as e:
            self.logger.error(f"Error placing MF order: {e}")
            raise
    
    def _handle_response(self, response):
        """Handle API response"""
        try:
            result = response.json()
            
            if response.status_code == 200:
                return result.get('data', result)
            else:
                error_msg = result.get('message', 'Unknown error')
                raise Exception(f"API Error: {error_msg}")
                
        except json.JSONDecodeError:
            raise Exception(f"Invalid JSON response: {response.text}")
    
    def is_market_open(self):
        """Check if market is open"""
        try:
            now = datetime.now()
            
            # Market hours: 9:15 AM to 3:30 PM on weekdays
            if now.weekday() >= 5:  # Weekend
                return False
            
            market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
            market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
            
            return market_open <= now <= market_close
            
        except Exception as e:
            self.logger.error(f"Error checking market status: {e}")
            return False
    
    def search_instruments(self, query, exchange=None):
        """Search for instruments"""
        try:
            instruments_df = self.get_instruments(exchange)
            
            # Search in tradingsymbol and name columns
            mask = (
                instruments_df['tradingsymbol'].str.contains(query, case=False, na=False) |
                instruments_df['name'].str.contains(query, case=False, na=False)
            )
            
            return instruments_df[mask]
            
        except Exception as e:
            self.logger.error(f"Error searching instruments: {e}")
            return pd.DataFrame()

def main():
    """Test Zerodha API"""
    # Initialize with your credentials
    api_key = "your_api_key"
    api_secret = "your_api_secret"
    access_token = "your_access_token"  # Get this after login
    
    kite = ZerodhaAPI(api_key, api_secret, access_token)
    
    try:
        # Test basic functionality
        profile = kite.get_profile()
        print(f"User: {profile['user_name']}")
        
        margins = kite.get_margins()
        print(f"Available margin: {margins['equity']['available']['cash']}")
        
        positions = kite.get_positions()
        print(f"Current positions: {len(positions['net'])}")
        
        # Get quote for a stock
        quote = kite.get_quote(['NSE:INFY'])
        print(f"INFY LTP: {quote['NSE:INFY']['last_price']}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
