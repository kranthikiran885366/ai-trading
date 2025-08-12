#!/usr/bin/env python3
"""
Real-time Market Data Stream with Multiple Data Sources
"""

import asyncio
import websockets
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dataclasses import dataclass
import aiohttp
import yfinance as yf

logger = logging.getLogger(__name__)

@dataclass
class MarketData:
    """Market data structure"""
    symbol: str
    price: float
    volume: float
    high: float
    low: float
    open: float
    close: float
    timestamp: datetime
    source: str

class MarketDataStream:
    """Real-time market data streaming"""
    
    def __init__(self, symbols: List[str], kafka_producer=None):
        self.symbols = symbols
        self.kafka_producer = kafka_producer
        
        # Data storage
        self.current_data = {}
        self.historical_data = {}
        
        # WebSocket connections
        self.websocket_connections = {}
        self.is_running = False
        
        # Data sources
        self.data_sources = {
            'binance': 'wss://stream.binance.com:9443/ws/',
            'finnhub': 'wss://ws.finnhub.io',
            'alpaca': 'wss://stream.data.alpaca.markets/v2/iex'
        }
        
        logger.info(f"Market Data Stream initialized for {len(symbols)} symbols")
    
    async def start(self):
        """Start market data streaming"""
        try:
            logger.info("Starting market data streams...")
            self.is_running = True
            
            # Start WebSocket connections for different sources
            tasks = []
            
            # Binance for crypto
            crypto_symbols = [s for s in self.symbols if s.endswith('USDT')]
            if crypto_symbols:
                tasks.append(self._start_binance_stream(crypto_symbols))
            
            # Finnhub for stocks
            stock_symbols = [s for s in self.symbols if not s.endswith('USDT')]
            if stock_symbols:
                tasks.append(self._start_finnhub_stream(stock_symbols))
            
            # Start historical data fetching
            tasks.append(self._fetch_historical_data())
            
            # Start data processing
            tasks.append(self._process_data_loop())
            
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            logger.error(f"Error starting market data streams: {e}")
    
    async def stop(self):
        """Stop market data streaming"""
        logger.info("Stopping market data streams...")
        self.is_running = False
        
        # Close WebSocket connections
        for connection in self.websocket_connections.values():
            if connection and not connection.closed:
                await connection.close()
    
    async def _start_binance_stream(self, symbols: List[str]):
        """Start Binance WebSocket stream for crypto"""
        try:
            # Convert symbols to Binance format
            binance_symbols = [s.lower() for s in symbols]
            
            # Create stream URL
            streams = [f"{symbol}@ticker" for symbol in binance_symbols]
            url = f"{self.data_sources['binance']}{'stream?streams=' + '/'.join(streams)}"
            
            while self.is_running:
                try:
                    async with websockets.connect(url) as websocket:
                        self.websocket_connections['binance'] = websocket
                        logger.info("Connected to Binance WebSocket")
                        
                        async for message in websocket:
                            if not self.is_running:
                                break
                            
                            await self._process_binance_message(message)
                            
                except websockets.exceptions.ConnectionClosed:
                    logger.warning("Binance WebSocket connection closed, reconnecting...")
                    await asyncio.sleep(5)
                except Exception as e:
                    logger.error(f"Binance WebSocket error: {e}")
                    await asyncio.sleep(5)
                    
        except Exception as e:
            logger.error(f"Error in Binance stream: {e}")
    
    async def _start_finnhub_stream(self, symbols: List[str]):
        """Start Finnhub WebSocket stream for stocks"""
        try:
            api_key = "your_finnhub_api_key"  # Replace with actual API key
            url = f"{self.data_sources['finnhub']}?token={api_key}"
            
            while self.is_running:
                try:
                    async with websockets.connect(url) as websocket:
                        self.websocket_connections['finnhub'] = websocket
                        logger.info("Connected to Finnhub WebSocket")
                        
                        # Subscribe to symbols
                        for symbol in symbols:
                            subscribe_msg = {"type": "subscribe", "symbol": symbol}
                            await websocket.send(json.dumps(subscribe_msg))
                        
                        async for message in websocket:
                            if not self.is_running:
                                break
                            
                            await self._process_finnhub_message(message)
                            
                except websockets.exceptions.ConnectionClosed:
                    logger.warning("Finnhub WebSocket connection closed, reconnecting...")
                    await asyncio.sleep(5)
                except Exception as e:
                    logger.error(f"Finnhub WebSocket error: {e}")
                    await asyncio.sleep(5)
                    
        except Exception as e:
            logger.error(f"Error in Finnhub stream: {e}")
    
    async def _process_binance_message(self, message: str):
        """Process Binance WebSocket message"""
        try:
            data = json.loads(message)
            
            if 'data' in data:
                ticker_data = data['data']
                
                if ticker_data.get('e') == '24hrTicker':
                    symbol = ticker_data['s']
                    
                    market_data = MarketData(
                        symbol=symbol,
                        price=float(ticker_data['c']),
                        volume=float(ticker_data['v']),
                        high=float(ticker_data['h']),
                        low=float(ticker_data['l']),
                        open=float(ticker_data['o']),
                        close=float(ticker_data['c']),
                        timestamp=datetime.fromtimestamp(ticker_data['E'] / 1000),
                        source='binance'
                    )
                    
                    await self._update_market_data(market_data)
                    
        except Exception as e:
            logger.error(f"Error processing Binance message: {e}")
    
    async def _process_finnhub_message(self, message: str):
        """Process Finnhub WebSocket message"""
        try:
            data = json.loads(message)
            
            if data.get('type') == 'trade' and 'data' in data:
                for trade in data['data']:
                    symbol = trade['s']
                    
                    market_data = MarketData(
                        symbol=symbol,
                        price=float(trade['p']),
                        volume=float(trade['v']),
                        high=float(trade['p']),  # Use trade price as high/low for now
                        low=float(trade['p']),
                        open=float(trade['p']),
                        close=float(trade['p']),
                        timestamp=datetime.fromtimestamp(trade['t'] / 1000),
                        source='finnhub'
                    )
                    
                    await self._update_market_data(market_data)
                    
        except Exception as e:
            logger.error(f"Error processing Finnhub message: {e}")
    
    async def _update_market_data(self, market_data: MarketData):
        """Update market data and broadcast"""
        try:
            # Store current data
            self.current_data[market_data.symbol] = market_data
            
            # Add to historical data
            if market_data.symbol not in self.historical_data:
                self.historical_data[market_data.symbol] = []
            
            self.historical_data[market_data.symbol].append(market_data)
            
            # Keep only last 1000 data points per symbol
            if len(self.historical_data[market_data.symbol]) > 1000:
                self.historical_data[market_data.symbol] = self.historical_data[market_data.symbol][-1000:]
            
            # Send to Kafka if available
            if self.kafka_producer:
                message = {
                    'type': 'market_data',
                    'symbol': market_data.symbol,
                    'price': market_data.price,
                    'volume': market_data.volume,
                    'timestamp': market_data.timestamp.isoformat(),
                    'source': market_data.source
                }
                
                self.kafka_producer.send('market_data', message)
            
            logger.debug(f"Updated market data for {market_data.symbol}: ${market_data.price:.4f}")
            
        except Exception as e:
            logger.error(f"Error updating market data: {e}")
    
    async def _fetch_historical_data(self):
        """Fetch historical data for all symbols"""
        try:
            while self.is_running:
                for symbol in self.symbols:
                    try:
                        # Fetch historical data using yfinance
                        ticker = yf.Ticker(symbol)
                        hist_data = ticker.history(period='1d', interval='1m')
                        
                        if not hist_data.empty:
                            # Convert to our format
                            for index, row in hist_data.iterrows():
                                market_data = MarketData(
                                    symbol=symbol,
                                    price=row['Close'],
                                    volume=row['Volume'],
                                    high=row['High'],
                                    low=row['Low'],
                                    open=row['Open'],
                                    close=row['Close'],
                                    timestamp=index.to_pydatetime(),
                                    source='yfinance'
                                )
                                
                                # Only add if we don't have recent data
                                if (symbol not in self.current_data or 
                                    (datetime.now() - self.current_data[symbol].timestamp).total_seconds() > 60):
                                    await self._update_market_data(market_data)
                        
                    except Exception as e:
                        logger.error(f"Error fetching historical data for {symbol}: {e}")
                
                # Wait 5 minutes before next fetch
                await asyncio.sleep(300)
                
        except Exception as e:
            logger.error(f"Error in historical data fetching: {e}")
    
    async def _process_data_loop(self):
        """Process and analyze market data"""
        try:
            while self.is_running:
                # Calculate technical indicators
                await self._calculate_technical_indicators()
                
                # Detect patterns
                await self._detect_patterns()
                
                # Update market overview
                await self._update_market_overview()
                
                # Wait 30 seconds
                await asyncio.sleep(30)
                
        except Exception as e:
            logger.error(f"Error in data processing loop: {e}")
    
    async def _calculate_technical_indicators(self):
        """Calculate technical indicators for all symbols"""
        try:
            for symbol, data_list in self.historical_data.items():
                if len(data_list) < 50:
                    continue
                
                # Convert to DataFrame
                df = pd.DataFrame([
                    {
                        'timestamp': d.timestamp,
                        'open': d.open,
                        'high': d.high,
                        'low': d.low,
                        'close': d.close,
                        'volume': d.volume
                    }
                    for d in data_list[-100:]  # Last 100 data points
                ])
                
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)
                
                # Calculate indicators
                indicators = await self._compute_indicators(df)
                
                # Store indicators with current data
                if symbol in self.current_data:
                    self.current_data[symbol].indicators = indicators
                    
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
    
    async def _compute_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Compute technical indicators"""
        try:
            indicators = {}
            
            # Simple Moving Averages
            indicators['sma_20'] = df['close'].rolling(20).mean().iloc[-1]
            indicators['sma_50'] = df['close'].rolling(50).mean().iloc[-1]
            
            # Exponential Moving Averages
            indicators['ema_12'] = df['close'].ewm(span=12).mean().iloc[-1]
            indicators['ema_26'] = df['close'].ewm(span=26).mean().iloc[-1]
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            indicators['rsi'] = (100 - (100 / (1 + rs))).iloc[-1]
            
            # MACD
            macd = indicators['ema_12'] - indicators['ema_26']
            signal = pd.Series([macd]).ewm(span=9).mean().iloc[0]
            indicators['macd'] = macd
            indicators['macd_signal'] = signal
            indicators['macd_histogram'] = macd - signal
            
            # Bollinger Bands
            sma_20 = df['close'].rolling(20).mean()
            std_20 = df['close'].rolling(20).std()
            indicators['bb_upper'] = (sma_20 + (std_20 * 2)).iloc[-1]
            indicators['bb_lower'] = (sma_20 - (std_20 * 2)).iloc[-1]
            indicators['bb_middle'] = sma_20.iloc[-1]
            
            # Volume indicators
            indicators['volume_sma'] = df['volume'].rolling(20).mean().iloc[-1]
            indicators['volume_ratio'] = df['volume'].iloc[-1] / indicators['volume_sma']
            
            # Volatility
            returns = df['close'].pct_change()
            indicators['volatility'] = returns.std() * np.sqrt(252)  # Annualized
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error computing indicators: {e}")
            return {}
    
    async def _detect_patterns(self):
        """Detect chart patterns"""
        try:
            for symbol, data_list in self.historical_data.items():
                if len(data_list) < 20:
                    continue
                
                # Get recent prices
                prices = [d.close for d in data_list[-20:]]
                
                # Simple pattern detection
                patterns = []
                
                # Uptrend detection
                if len(prices) >= 10:
                    recent_trend = np.polyfit(range(10), prices[-10:], 1)[0]
                    if recent_trend > 0:
                        patterns.append('uptrend')
                    elif recent_trend < 0:
                        patterns.append('downtrend')
                
                # Support/Resistance levels
                highs = [d.high for d in data_list[-20:]]
                lows = [d.low for d in data_list[-20:]]
                
                resistance = max(highs)
                support = min(lows)
                
                current_price = prices[-1]
                
                if abs(current_price - resistance) / resistance < 0.01:
                    patterns.append('near_resistance')
                elif abs(current_price - support) / support < 0.01:
                    patterns.append('near_support')
                
                # Store patterns
                if symbol in self.current_data:
                    self.current_data[symbol].patterns = patterns
                    
        except Exception as e:
            logger.error(f"Error detecting patterns: {e}")
    
    async def _update_market_overview(self):
        """Update market overview statistics"""
        try:
            if not self.current_data:
                return
            
            # Calculate market statistics
            prices = [data.price for data in self.current_data.values()]
            volumes = [data.volume for data in self.current_data.values()]
            
            self.market_overview = {
                'timestamp': datetime.now().isoformat(),
                'symbols_count': len(self.current_data),
                'avg_price': np.mean(prices),
                'total_volume': sum(volumes),
                'price_range': {
                    'min': min(prices),
                    'max': max(prices)
                },
                'active_symbols': list(self.current_data.keys())
            }
            
        except Exception as e:
            logger.error(f"Error updating market overview: {e}")
    
    async def get_latest_data(self, symbol: str) -> Optional[MarketData]:
        """Get latest market data for symbol"""
        return self.current_data.get(symbol)
    
    async def get_all_data(self) -> Dict[str, pd.DataFrame]:
        """Get all current market data as DataFrames"""
        try:
            result = {}
            
            for symbol, data_list in self.historical_data.items():
                if len(data_list) > 0:
                    df = pd.DataFrame([
                        {
                            'timestamp': d.timestamp,
                            'open': d.open,
                            'high': d.high,
                            'low': d.low,
                            'close': d.close,
                            'volume': d.volume
                        }
                        for d in data_list
                    ])
                    
                    df.set_index('timestamp', inplace=True)
                    df.sort_index(inplace=True)
                    result[symbol] = df
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting all data: {e}")
            return {}
    
    async def get_market_overview(self) -> Dict[str, Any]:
        """Get market overview"""
        return getattr(self, 'market_overview', {})
