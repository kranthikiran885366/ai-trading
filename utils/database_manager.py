#!/usr/bin/env python3
"""
Database Manager for MongoDB and Redis Integration
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json
import motor.motor_asyncio
import redis.asyncio as redis
import pandas as pd

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manage MongoDB and Redis connections"""
    
    def __init__(self, mongodb_uri: str, redis_host: str, redis_port: int):
        self.mongodb_uri = mongodb_uri
        self.redis_host = redis_host
        self.redis_port = redis_port
        
        # Database connections
        self.mongo_client = None
        self.mongo_db = None
        self.redis_client = None
        
        # Collections
        self.trades_collection = None
        self.signals_collection = None
        self.performance_collection = None
        self.news_collection = None
        
        logger.info("Database Manager initialized")
    
    async def connect(self):
        """Connect to databases"""
        try:
            # Connect to MongoDB
            self.mongo_client = motor.motor_asyncio.AsyncIOMotorClient(self.mongodb_uri)
            self.mongo_db = self.mongo_client.trading_bot
            
            # Initialize collections
            self.trades_collection = self.mongo_db.trades
            self.signals_collection = self.mongo_db.signals
            self.performance_collection = self.mongo_db.performance
            self.news_collection = self.mongo_db.news
            
            # Test MongoDB connection
            await self.mongo_client.admin.command('ping')
            logger.info("Connected to MongoDB")
            
            # Connect to Redis
            self.redis_client = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                decode_responses=True
            )
            
            # Test Redis connection
            await self.redis_client.ping()
            logger.info("Connected to Redis")
            
            # Create indexes
            await self._create_indexes()
            
        except Exception as e:
            logger.error(f"Error connecting to databases: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from databases"""
        try:
            if self.mongo_client:
                self.mongo_client.close()
            
            if self.redis_client:
                await self.redis_client.close()
            
            logger.info("Disconnected from databases")
            
        except Exception as e:
            logger.error(f"Error disconnecting from databases: {e}")
    
    async def _create_indexes(self):
        """Create database indexes"""
        try:
            # Trades collection indexes
            await self.trades_collection.create_index([("timestamp", -1)])
            await self.trades_collection.create_index([("symbol", 1)])
            await self.trades_collection.create_index([("action", 1)])
            
            # Signals collection indexes
            await self.signals_collection.create_index([("timestamp", -1)])
            await self.signals_collection.create_index([("symbol", 1)])
            await self.signals_collection.create_index([("confidence", -1)])
            
            # Performance collection indexes
            await self.performance_collection.create_index([("date", -1)])
            
            # News collection indexes
            await self.news_collection.create_index([("timestamp", -1)])
            await self.news_collection.create_index([("symbols", 1)])
            
            logger.info("Database indexes created")
            
        except Exception as e:
            logger.error(f"Error creating indexes: {e}")
    
    async def store_trade(self, trade_data: Dict[str, Any]):
        """Store trade in database"""
        try:
            # Add metadata
            trade_data['stored_at'] = datetime.now()
            trade_data['_id'] = f"{trade_data['symbol']}_{trade_data['timestamp'].isoformat()}"
            
            # Store in MongoDB
            await self.trades_collection.insert_one(trade_data)
            
            # Cache recent trade in Redis
            redis_key = f"recent_trade:{trade_data['symbol']}"
            await self.redis_client.setex(
                redis_key,
                3600,  # 1 hour expiry
                json.dumps(trade_data, default=str)
            )
            
            logger.debug(f"Stored trade: {trade_data['symbol']} {trade_data['action']}")
            
        except Exception as e:
            logger.error(f"Error storing trade: {e}")
    
    async def store_signal(self, signal_data: Dict[str, Any]):
        """Store trading signal in database"""
        try:
            # Add metadata
            signal_data['stored_at'] = datetime.now()
            signal_data['_id'] = f"{signal_data['symbol']}_{signal_data['timestamp'].isoformat()}"
            
            # Store in MongoDB
            await self.signals_collection.insert_one(signal_data)
            
            # Cache in Redis
            redis_key = f"recent_signal:{signal_data['symbol']}"
            await self.redis_client.setex(
                redis_key,
                1800,  # 30 minutes expiry
                json.dumps(signal_data, default=str)
            )
            
            logger.debug(f"Stored signal: {signal_data['symbol']} {signal_data['action']}")
            
        except Exception as e:
            logger.error(f"Error storing signal: {e}")
    
    async def store_performance_snapshot(self, performance_data: Dict[str, Any]):
        """Store performance snapshot"""
        try:
            # Add metadata
            performance_data['stored_at'] = datetime.now()
            performance_data['_id'] = f"performance_{datetime.now().date().isoformat()}"
            
            # Store in MongoDB (upsert by date)
            await self.performance_collection.replace_one(
                {'date': datetime.now().date()},
                performance_data,
                upsert=True
            )
            
            # Cache latest performance in Redis
            await self.redis_client.setex(
                'latest_performance',
                300,  # 5 minutes expiry
                json.dumps(performance_data, default=str)
            )
            
            logger.debug("Stored performance snapshot")
            
        except Exception as e:
            logger.error(f"Error storing performance snapshot: {e}")
    
    async def store_news_item(self, news_data: Dict[str, Any]):
        """Store news item"""
        try:
            # Add metadata
            news_data['stored_at'] = datetime.now()
            news_data['_id'] = f"news_{hash(news_data['title'])}"
            
            # Store in MongoDB
            await self.news_collection.insert_one(news_data)
            
            logger.debug(f"Stored news item: {news_data['title'][:50]}...")
            
        except Exception as e:
            logger.error(f"Error storing news item: {e}")
    
    async def get_recent_trades(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent trades"""
        try:
            cursor = self.trades_collection.find().sort("timestamp", -1).limit(limit)
            trades = await cursor.to_list(length=limit)
            
            # Convert ObjectId to string for JSON serialization
            for trade in trades:
                if '_id' in trade:
                    trade['_id'] = str(trade['_id'])
            
            return trades
            
        except Exception as e:
            logger.error(f"Error getting recent trades: {e}")
            return []
    
    async def get_recent_signals(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent signals"""
        try:
            cursor = self.signals_collection.find().sort("timestamp", -1).limit(limit)
            signals = await cursor.to_list(length=limit)
            
            # Convert ObjectId to string for JSON serialization
            for signal in signals:
                if '_id' in signal:
                    signal['_id'] = str(signal['_id'])
            
            return signals
            
        except Exception as e:
            logger.error(f"Error getting recent signals: {e}")
            return []
    
    async def get_training_data(self, days: int = 30) -> pd.DataFrame:
        """Get training data for AI models"""
        try:
            # Get trades from last N days
            cutoff_date = datetime.now() - timedelta(days=days)
            
            cursor = self.trades_collection.find({
                "timestamp": {"$gte": cutoff_date}
            }).sort("timestamp", 1)
            
            trades = await cursor.to_list(length=None)
            
            if not trades:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(trades)
            
            # Add features for training
            df['return'] = df.groupby('symbol')['price'].pct_change()
            df['target'] = (df['return'] > 0).astype(int)  # 1 for positive return, 0 for negative
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting training data: {e}")
            return pd.DataFrame()
    
    async def get_performance_history(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get performance history"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            cursor = self.performance_collection.find({
                "date": {"$gte": cutoff_date.date()}
            }).sort("date", 1)
            
            performance_data = await cursor.to_list(length=None)
            
            # Convert ObjectId to string
            for item in performance_data:
                if '_id' in item:
                    item['_id'] = str(item['_id'])
            
            return performance_data
            
        except Exception as e:
            logger.error(f"Error getting performance history: {e}")
            return []
    
    async def cache_market_data(self, symbol: str, data: Dict[str, Any], expiry: int = 60):
        """Cache market data in Redis"""
        try:
            redis_key = f"market_data:{symbol}"
            await self.redis_client.setex(
                redis_key,
                expiry,
                json.dumps(data, default=str)
            )
            
        except Exception as e:
            logger.error(f"Error caching market data: {e}")
    
    async def get_cached_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get cached market data from Redis"""
        try:
            redis_key = f"market_data:{symbol}"
            cached_data = await self.redis_client.get(redis_key)
            
            if cached_data:
                return json.loads(cached_data)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting cached market data: {e}")
            return None
    
    async def cache_ai_prediction(self, symbol: str, prediction: Dict[str, Any], expiry: int = 300):
        """Cache AI prediction in Redis"""
        try:
            redis_key = f"ai_prediction:{symbol}"
            await self.redis_client.setex(
                redis_key,
                expiry,
                json.dumps(prediction, default=str)
            )
            
        except Exception as e:
            logger.error(f"Error caching AI prediction: {e}")
    
    async def get_cached_ai_prediction(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get cached AI prediction from Redis"""
        try:
            redis_key = f"ai_prediction:{symbol}"
            cached_prediction = await self.redis_client.get(redis_key)
            
            if cached_prediction:
                return json.loads(cached_prediction)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting cached AI prediction: {e}")
            return None
    
    async def get_symbol_statistics(self, symbol: str, days: int = 30) -> Dict[str, Any]:
        """Get statistics for a specific symbol"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # Get trades for symbol
            cursor = self.trades_collection.find({
                "symbol": symbol,
                "timestamp": {"$gte": cutoff_date}
            }).sort("timestamp", 1)
            
            trades = await cursor.to_list(length=None)
            
            if not trades:
                return {}
            
            # Calculate statistics
            df = pd.DataFrame(trades)
            
            buy_trades = df[df['action'] == 'BUY']
            sell_trades = df[df['action'] == 'SELL']
            
            stats = {
                'total_trades': len(trades),
                'buy_trades': len(buy_trades),
                'sell_trades': len(sell_trades),
                'avg_price': df['price'].mean(),
                'price_std': df['price'].std(),
                'total_volume': df['quantity'].sum(),
                'avg_volume': df['quantity'].mean(),
                'first_trade': trades[0]['timestamp'],
                'last_trade': trades[-1]['timestamp']
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting symbol statistics: {e}")
            return {}
    
    async def cleanup_old_data(self, days: int = 90):
        """Clean up old data from databases"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # Clean old trades
            result = await self.trades_collection.delete_many({
                "timestamp": {"$lt": cutoff_date}
            })
            logger.info(f"Deleted {result.deleted_count} old trades")
            
            # Clean old signals
            result = await self.signals_collection.delete_many({
                "timestamp": {"$lt": cutoff_date}
            })
            logger.info(f"Deleted {result.deleted_count} old signals")
            
            # Clean old news
            result = await self.news_collection.delete_many({
                "timestamp": {"$lt": cutoff_date}
            })
            logger.info(f"Deleted {result.deleted_count} old news items")
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
