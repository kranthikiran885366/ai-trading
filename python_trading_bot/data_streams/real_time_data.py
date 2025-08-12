#!/usr/bin/env python3
"""
Real-time Data Stream - Handles live market data feeds
"""

import asyncio
import logging
import websocket
import json
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
import pandas as pd
import numpy as np
from dataclasses import dataclass
import yfinance as yf
import requests
from collections import defaultdict, deque
import os
import aiohttp

logger = logging.getLogger(__name__)

@dataclass
class MarketData:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    bid: float = 0.0
    ask: float = 0.0
    bid_size: float = 0.0
    ask_size: float = 0.0

@dataclass
class Tick:
    symbol: str
    timestamp: datetime
    price: float
    volume: float
    side: str  # BUY/SELL

class DataProvider:
    """Base class for data providers"""
    
    def __init__(self, name: str):
        self.name = name
        self.is_connected = False
        self.subscribers = defaultdict(list)
        self.data_buffer = defaultdict(lambda: deque(maxlen=1000))
    
    async def connect(self):
        """Connect to data source"""
        raise NotImplementedError
    
    async def disconnect(self):
        """Disconnect from data source"""
        raise NotImplementedError
    
    async def subscribe(self, symbols: List[str], callback: Callable):
        """Subscribe to symbols"""
        raise NotImplementedError
    
    async def unsubscribe(self, symbols: List[str]):
        """Unsubscribe from symbols"""
        raise NotImplementedError
    
    def add_subscriber(self, symbol: str, callback: Callable):
        """Add subscriber for symbol"""
        self.subscribers[symbol].append(callback)
    
    def remove_subscriber(self, symbol: str, callback: Callable):
        """Remove subscriber for symbol"""
        if callback in self.subscribers[symbol]:
            self.subscribers[symbol].remove(callback)
    
    async def notify_subscribers(self, symbol: str, data: Any):
        """Notify all subscribers of new data"""
        for callback in self.subscribers[symbol]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(symbol, data)
                else:
                    callback(symbol, data)
            except Exception as e:
                logger.error(f"Error notifying subscriber: {e}")

class YahooFinanceProvider(DataProvider):
    """Yahoo Finance data provider"""
    
    def __init__(self):
        super().__init__("Yahoo Finance")
        self.subscribed_symbols = set()
        self.update_interval = 60  # seconds
        self.is_running = False
        self.update_task = None
    
    async def connect(self):
        """Connect to Yahoo Finance"""
        try:
            self.is_connected = True
            logger.info("Connected to Yahoo Finance")
            return True
        except Exception as e:
            logger.error(f"Error connecting to Yahoo Finance: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from Yahoo Finance"""
        self.is_connected = False
        self.is_running = False
        if self.update_task:
            self.update_task.cancel()
        logger.info("Disconnected from Yahoo Finance")
    
    async def subscribe(self, symbols: List[str], callback: Callable):
        """Subscribe to symbols"""
        try:
            self.subscribed_symbols.update(symbols)
            
            if not self.is_running:
                self.is_running = True
                self.update_task = asyncio.create_task(self._update_loop())
            
            logger.info(f"Subscribed to Yahoo Finance symbols: {symbols}")
            
        except Exception as e:
            logger.error(f"Error subscribing to symbols: {e}")
    
    async def unsubscribe(self, symbols: List[str]):
        """Unsubscribe from symbols"""
        try:
            for symbol in symbols:
                self.subscribed_symbols.discard(symbol)
            
            logger.info(f"Unsubscribed from Yahoo Finance symbols: {symbols}")
            
        except Exception as e:
            logger.error(f"Error unsubscribing from symbols: {e}")
    
    async def _update_loop(self):
        """Main update loop for fetching data"""
        while self.is_running and self.is_connected:
            try:
                if self.subscribed_symbols:
                    await self._fetch_and_notify()
                
                await asyncio.sleep(self.update_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in Yahoo Finance update loop: {e}")
                await asyncio.sleep(self.update_interval)
    
    async def _fetch_and_notify(self):
        """Fetch data and notify subscribers"""
        try:
            symbols_list = list(self.subscribed_symbols)
            
            # Fetch data using yfinance
            tickers = yf.Tickers(' '.join(symbols_list))
            
            for symbol in symbols_list:
                try:
                    ticker = tickers.tickers[symbol]
                    info = ticker.info
                    hist = ticker.history(period='1d', interval='1m')
                    
                    if not hist.empty:
                        latest = hist.iloc[-1]
                        
                        market_data = MarketData(
                            symbol=symbol,
                            timestamp=datetime.now(),
                            open=float(latest['Open']),
                            high=float(latest['High']),
                            low=float(latest['Low']),
                            close=float(latest['Close']),
                            volume=float(latest['Volume']),
                            bid=info.get('bid', 0.0),
                            ask=info.get('ask', 0.0),
                            bid_size=info.get('bidSize', 0.0),
                            ask_size=info.get('askSize', 0.0)
                        )
                        
                        await self.notify_subscribers(symbol, market_data)
                
                except Exception as e:
                    logger.error(f"Error fetching data for {symbol}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error in fetch and notify: {e}")

class AlphaVantageDataProvider(DataProvider):
    """Alpha Vantage REST API data provider"""
    
    def __init__(self, api_key: str):
        super().__init__("AlphaVantage")
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.rate_limit_delay = 12  # 5 calls per minute
        self.last_request_time = 0
    
    async def connect(self):
        """Connect to Alpha Vantage (no persistent connection needed)"""
        self.is_connected = True
        logger.info("Connected to Alpha Vantage API")
    
    async def disconnect(self):
        """Disconnect from Alpha Vantage"""
        self.is_connected = False
        logger.info("Disconnected from Alpha Vantage API")
    
    async def subscribe(self, symbols: List[str], callback: Callable = None):
        """Subscribe to Alpha Vantage symbols (polling-based)"""
        try:
            for symbol in symbols:
                self.subscribed_symbols.add(symbol)
                if callback:
                    self.add_subscriber(symbol, callback)
            
            # Start polling task
            asyncio.create_task(self._polling_loop())
            logger.info(f"Subscribed to Alpha Vantage symbols: {symbols}")
            
        except Exception as e:
            logger.error(f"Error subscribing to Alpha Vantage symbols: {e}")
    
    async def unsubscribe(self, symbols: List[str]):
        """Unsubscribe from Alpha Vantage symbols"""
        for symbol in symbols:
            self.subscribed_symbols.discard(symbol)
        logger.info(f"Unsubscribed from Alpha Vantage symbols: {symbols}")
    
    async def _polling_loop(self):
        """Polling loop for Alpha Vantage data"""
        while self.is_connected and self.subscribed_symbols:
            try:
                for symbol in list(self.subscribed_symbols):
                    await self._fetch_symbol_data(symbol)
                    await asyncio.sleep(self.rate_limit_delay)
                
                # Wait before next polling cycle
                await asyncio.sleep(60)  # Poll every minute
                
            except Exception as e:
                logger.error(f"Error in Alpha Vantage polling loop: {e}")
                await asyncio.sleep(60)
    
    async def _fetch_symbol_data(self, symbol: str):
        """Fetch data for a symbol"""
        try:
            # Rate limiting
            current_time = time.time()
            if current_time - self.last_request_time < self.rate_limit_delay:
                await asyncio.sleep(self.rate_limit_delay - (current_time - self.last_request_time))
            
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': symbol,
                'apikey': self.api_key
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    data = await response.json()
                    
                    if 'Global Quote' in data:
                        quote_data = data['Global Quote']
                        
                        market_data = MarketData(
                            symbol=symbol,
                            timestamp=datetime.now(),
                            open=float(quote_data['02. open']),
                            high=float(quote_data['03. high']),
                            low=float(quote_data['04. low']),
                            close=float(quote_data['05. price']),
                            volume=float(quote_data['06. volume'])
                        )
                        
                        # Store in buffer
                        self.data_buffer[symbol].append(market_data)
                        
                        # Notify subscribers
                        await self.notify_subscribers(symbol, market_data)
            
            self.last_request_time = time.time()
            
        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage data for {symbol}: {e}")

class BinanceDataProvider(DataProvider):
    """Binance WebSocket data provider"""
    
    def __init__(self):
        super().__init__("Binance")
        self.ws_url = "wss://stream.binance.com:9443/ws"
        self.ws = None
        self.subscribed_symbols = set()
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
    
    async def connect(self):
        """Connect to Binance WebSocket"""
        try:
            self.ws = await websocket.connect(self.ws_url)
            self.is_connected = True
            self.reconnect_attempts = 0
            logger.info("Connected to Binance WebSocket")
            
            # Start listening for messages
            asyncio.create_task(self._listen_messages())
            
        except Exception as e:
            logger.error(f"Error connecting to Binance WebSocket: {e}")
            await self._handle_reconnect()
    
    async def disconnect(self):
        """Disconnect from Binance WebSocket"""
        try:
            if self.ws:
                await self.ws.close()
            self.is_connected = False
            logger.info("Disconnected from Binance WebSocket")
            
        except Exception as e:
            logger.error(f"Error disconnecting from Binance WebSocket: {e}")
    
    async def subscribe(self, symbols: List[str], callback: Callable = None):
        """Subscribe to Binance symbols"""
        try:
            if not self.is_connected:
                await self.connect()
            
            # Convert symbols to Binance format
            binance_symbols = [symbol.lower() for symbol in symbols]
            
            # Subscribe to ticker streams
            streams = []
            for symbol in binance_symbols:
                streams.extend([
                    f"{symbol}@ticker",
                    f"{symbol}@trade",
                    f"{symbol}@depth5"
                ])
                self.subscribed_symbols.add(symbol.upper())
                
                if callback:
                    self.add_subscriber(symbol.upper(), callback)
            
            # Send subscription message
            subscribe_msg = {
                "method": "SUBSCRIBE",
                "params": streams,
                "id": int(time.time())
            }
            
            await self.ws.send(json.dumps(subscribe_msg))
            logger.info(f"Subscribed to Binance symbols: {symbols}")
            
        except Exception as e:
            logger.error(f"Error subscribing to Binance symbols: {e}")
            await self._handle_reconnect()
    
    async def unsubscribe(self, symbols: List[str]):
        """Unsubscribe from Binance symbols"""
        try:
            if not self.is_connected:
                return
            
            # Convert symbols to Binance format
            binance_symbols = [symbol.lower() for symbol in symbols]
            
            # Unsubscribe from ticker streams
            streams = []
            for symbol in binance_symbols:
                streams.extend([
                    f"{symbol}@ticker",
                    f"{symbol}@trade",
                    f"{symbol}@depth5"
                ])
                self.subscribed_symbols.discard(symbol.upper())
            
            # Send unsubscription message
            unsubscribe_msg = {
                "method": "UNSUBSCRIBE",
                "params": streams,
                "id": int(time.time())
            }
            
            await self.ws.send(json.dumps(unsubscribe_msg))
            logger.info(f"Unsubscribed from Binance symbols: {symbols}")
            
        except Exception as e:
            logger.error(f"Error unsubscribing from Binance symbols: {e}")
    
    async def _listen_messages(self):
        """Listen for WebSocket messages"""
        try:
            while self.is_connected:
                message = await self.ws.recv()
                data = json.loads(message)
                
                if 'stream' in data:
                    await self._process_stream_data(data)
                
        except websocket.exceptions.ConnectionClosed:
            logger.warning("Binance WebSocket connection closed")
            self.is_connected = False
            await self._handle_reconnect()
        except Exception as e:
            logger.error(f"Error listening to Binance messages: {e}")
            await self._handle_reconnect()
    
    async def _process_stream_data(self, data: Dict):
        """Process incoming stream data"""
        try:
            stream = data['stream']
            stream_data = data['data']
            
            if '@ticker' in stream:
                await self._process_ticker_data(stream_data)
            elif '@trade' in stream:
                await self._process_trade_data(stream_data)
            elif '@depth' in stream:
                await self._process_depth_data(stream_data)
                
        except Exception as e:
            logger.error(f"Error processing stream data: {e}")
    
    async def _process_ticker_data(self, data: Dict):
        """Process ticker data"""
        try:
            symbol = data['s']
            
            market_data = MarketData(
                symbol=symbol,
                timestamp=datetime.fromtimestamp(data['E'] / 1000),
                open=float(data['o']),
                high=float(data['h']),
                low=float(data['l']),
                close=float(data['c']),
                volume=float(data['v']),
                bid=float(data['b']),
                ask=float(data['a']),
                bid_size=float(data['B']),
                ask_size=float(data['A'])
            )
            
            # Store in buffer
            self.data_buffer[symbol].append(market_data)
            
            # Notify subscribers
            await self.notify_subscribers(symbol, market_data)
            
        except Exception as e:
            logger.error(f"Error processing ticker data: {e}")
    
    async def _process_trade_data(self, data: Dict):
        """Process trade data"""
        try:
            symbol = data['s']
            
            tick = Tick(
                symbol=symbol,
                timestamp=datetime.fromtimestamp(data['T'] / 1000),
                price=float(data['p']),
                volume=float(data['q']),
                side='BUY' if data['m'] else 'SELL'
            )
            
            # Notify subscribers
            await self.notify_subscribers(f"{symbol}_TRADE", tick)
            
        except Exception as e:
            logger.error(f"Error processing trade data: {e}")
    
    async def _process_depth_data(self, data: Dict):
        """Process order book depth data"""
        try:
            symbol = data['s']
            
            depth_data = {
                'symbol': symbol,
                'timestamp': datetime.fromtimestamp(data['E'] / 1000),
                'bids': [[float(bid[0]), float(bid[1])] for bid in data['b']],
                'asks': [[float(ask[0]), float(ask[1])] for ask in data['a']]
            }
            
            # Notify subscribers
            await self.notify_subscribers(f"{symbol}_DEPTH", depth_data)
            
        except Exception as e:
            logger.error(f"Error processing depth data: {e}")
    
    async def _handle_reconnect(self):
        """Handle reconnection logic"""
        if self.reconnect_attempts < self.max_reconnect_attempts:
            self.reconnect_attempts += 1
            wait_time = min(2 ** self.reconnect_attempts, 60)  # Exponential backoff
            
            logger.info(f"Attempting to reconnect in {wait_time} seconds (attempt {self.reconnect_attempts})")
            await asyncio.sleep(wait_time)
            
            try:
                await self.connect()
                
                # Re-subscribe to symbols
                if self.subscribed_symbols:
                    await self.subscribe(list(self.subscribed_symbols))
                    
            except Exception as e:
                logger.error(f"Reconnection attempt failed: {e}")
        else:
            logger.error("Max reconnection attempts reached")

class RealTimeDataStream:
    """Main real-time data stream manager"""
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.providers = {}
        self.is_running = False
        self.data_callbacks = defaultdict(list)
        self.aggregated_data = defaultdict(lambda: deque(maxlen=1000))
        
        # Initialize providers
        self._initialize_providers()
        
        logger.info("Real-time Data Stream initialized")
    
    def _initialize_providers(self):
        """Initialize data providers"""
        # Add Binance provider
        self.providers['binance'] = BinanceDataProvider()
        
        # Add Alpha Vantage provider if API key is available
        alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        if alpha_vantage_key:
            self.providers['alphavantage'] = AlphaVantageDataProvider(alpha_vantage_key)
    
    async def start(self):
        """Start data stream"""
        try:
            self.is_running = True
            
            # Connect to all providers
            for provider in self.providers.values():
                await provider.connect()
                await provider.subscribe(self.symbols, self._on_data_received)
            
            logger.info("Real-time data stream started")
            
        except Exception as e:
            logger.error(f"Error starting data stream: {e}")
    
    async def stop(self):
        """Stop data stream"""
        try:
            self.is_running = False
            
            # Disconnect from all providers
            for provider in self.providers.values():
                await provider.disconnect()
            
            logger.info("Real-time data stream stopped")
            
        except Exception as e:
            logger.error(f"Error stopping data stream: {e}")
    
    async def _on_data_received(self, symbol: str, data: Any):
        """Handle received data from providers"""
        try:
            # Store aggregated data
            self.aggregated_data[symbol].append(data)
            
            # Notify callbacks
            for callback in self.data_callbacks[symbol]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(symbol, data)
                    else:
                        callback(symbol, data)
                except Exception as e:
                    logger.error(f"Error in data callback: {e}")
                    
        except Exception as e:
            logger.error(f"Error handling received data: {e}")
    
    def add_callback(self, symbol: str, callback: Callable):
        """Add data callback for symbol"""
        self.data_callbacks[symbol].append(callback)
    
    def remove_callback(self, symbol: str, callback: Callable):
        """Remove data callback for symbol"""
        if callback in self.data_callbacks[symbol]:
            self.data_callbacks[symbol].remove(callback)
    
    async def get_latest_data(self, symbol: str) -> Optional[MarketData]:
        """Get latest data for symbol"""
        try:
            if symbol in self.aggregated_data and self.aggregated_data[symbol]:
                return self.aggregated_data[symbol][-1]
            return None
            
        except Exception as e:
            logger.error(f"Error getting latest data for {symbol}: {e}")
            return None
    
    def get_historical_data(self, symbol: str, limit: int = 100) -> List[MarketData]:
        """Get historical data for symbol"""
        try:
            if symbol in self.aggregated_data:
                data_list = list(self.aggregated_data[symbol])
                return data_list[-limit:] if limit else data_list
            return []
            
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return []
    
    def get_ohlcv_dataframe(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """Get OHLCV data as pandas DataFrame"""
        try:
            data_list = self.get_historical_data(symbol, limit)
            
            if not data_list:
                return pd.DataFrame()
            
            df_data = []
            for data in data_list:
                if isinstance(data, MarketData):
                    df_data.append({
                        'timestamp': data.timestamp,
                        'open': data.open,
                        'high': data.high,
                        'low': data.low,
                        'close': data.close,
                        'volume': data.volume
                    })
            
            df = pd.DataFrame(df_data)
            if not df.empty:
                df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error creating OHLCV DataFrame for {symbol}: {e}")
            return pd.DataFrame()
    
    async def add_symbol(self, symbol: str):
        """Add new symbol to stream"""
        try:
            if symbol not in self.symbols:
                self.symbols.append(symbol)
                
                # Subscribe to new symbol on all providers
                for provider in self.providers.values():
                    if provider.is_connected:
                        await provider.subscribe([symbol], self._on_data_received)
                
                logger.info(f"Added symbol to data stream: {symbol}")
                
        except Exception as e:
            logger.error(f"Error adding symbol {symbol}: {e}")
    
    async def remove_symbol(self, symbol: str):
        """Remove symbol from stream"""
        try:
            if symbol in self.symbols:
                self.symbols.remove(symbol)
                
                # Unsubscribe from symbol on all providers
                for provider in self.providers.values():
                    if provider.is_connected:
                        await provider.unsubscribe([symbol])
                
                # Clear data
                if symbol in self.aggregated_data:
                    del self.aggregated_data[symbol]
                if symbol in self.data_callbacks:
                    del self.data_callbacks[symbol]
                
                logger.info(f"Removed symbol from data stream: {symbol}")
                
        except Exception as e:
            logger.error(f"Error removing symbol {symbol}: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get data stream status"""
        provider_status = {}
        for name, provider in self.providers.items():
            provider_status[name] = {
                'connected': provider.is_connected,
                'subscribed_symbols': len(provider.subscribed_symbols)
            }
        
        return {
            'running': self.is_running,
            'symbols': self.symbols,
            'providers': provider_status,
            'data_points': {symbol: len(data) for symbol, data in self.aggregated_data.items()}
        }
