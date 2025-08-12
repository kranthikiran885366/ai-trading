#!/usr/bin/env python3
"""
Advanced AI Trading Bot System - Main Entry Point
Fully automated trading bot with 95% success rate target
"""

import asyncio
import logging
import signal
import sys
import os
import json
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
import argparse
import yaml
import uvicorn
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import redis
from kafka import KafkaProducer, KafkaConsumer
import motor.motor_asyncio
from pymongo import MongoClient

# Import our custom modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_engine.advanced_ai_engine import AdvancedAIEngine
from ai_engine.signal_processor import SignalProcessor
from ai_engine.risk_manager import AdvancedRiskManager
from trading_engine.execution_engine import ExecutionEngine
from trading_engine.portfolio_manager import PortfolioManager
from data_streams.market_data_stream import MarketDataStream
from data_streams.news_stream import NewsStream
from brokers.multi_broker_manager import MultiBrokerManager
from utils.database_manager import DatabaseManager
from utils.notification_manager import NotificationManager
from utils.performance_tracker import PerformanceTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/trading_bot.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TradingConfig:
    """Complete trading bot configuration"""
    # Database Configuration
    mongodb_uri: str = "mongodb://localhost:27017"
    redis_host: str = "localhost"
    redis_port: int = 6379
    kafka_bootstrap_servers: str = "localhost:9092"
    
    # API Keys
    binance_api_key: str = ""
    binance_api_secret: str = ""
    alpaca_api_key: str = ""
    alpaca_api_secret: str = ""
    alpha_vantage_key: str = ""
    newsapi_key: str = ""
    
    # Trading Parameters
    initial_capital: float = 100000.0
    max_daily_loss: float = 2000.0
    max_drawdown: float = 0.10
    max_position_size: float = 0.05
    target_success_rate: float = 0.95
    
    # AI Configuration
    ai_confidence_threshold: float = 0.85
    ensemble_models: int = 5
    retrain_interval_hours: int = 6
    
    # Symbols to trade
    symbols: List[str] = None
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = [
                'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA',
                'NVDA', 'META', 'NFLX', 'AMD', 'CRM',
                'BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT'
            ]

class AdvancedTradingBot:
    """Main trading bot class with 95% success rate target"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.is_running = False
        self.shutdown_event = threading.Event()
        
        # Initialize FastAPI app
        self.app = FastAPI(title="Advanced Trading Bot API", version="2.0.0")
        self._setup_fastapi()
        
        # Initialize core components
        self._initialize_components()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("Advanced Trading Bot initialized with 95% success rate target")
    
    def _setup_fastapi(self):
        """Setup FastAPI application"""
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Mount static files for frontend
        self.app.mount("/static", StaticFiles(directory="frontend/dist"), name="static")
        
        # Setup API routes
        self._setup_api_routes()
        self._setup_websocket_routes()
    
    def _initialize_components(self):
        """Initialize all trading bot components"""
        logger.info("Initializing advanced trading bot components...")
        
        # Database connections
        self.db_manager = DatabaseManager(
            mongodb_uri=self.config.mongodb_uri,
            redis_host=self.config.redis_host,
            redis_port=self.config.redis_port
        )
        
        # Kafka for real-time messaging
        self.kafka_producer = KafkaProducer(
            bootstrap_servers=[self.config.kafka_bootstrap_servers],
            value_serializer=lambda x: json.dumps(x).encode('utf-8')
        )
        
        # Advanced AI Engine with ensemble models
        self.ai_engine = AdvancedAIEngine(
            confidence_threshold=self.config.ai_confidence_threshold,
            ensemble_size=self.config.ensemble_models,
            target_success_rate=self.config.target_success_rate
        )
        
        # Signal processing with advanced filtering
        self.signal_processor = SignalProcessor(
            min_confidence=0.8,
            max_signals_per_minute=10,
            correlation_threshold=0.7
        )
        
        # Advanced risk management
        self.risk_manager = AdvancedRiskManager(
            max_daily_loss=self.config.max_daily_loss,
            max_drawdown=self.config.max_drawdown,
            max_position_size=self.config.max_position_size,
            portfolio_value=self.config.initial_capital
        )
        
        # Multi-broker management
        self.broker_manager = MultiBrokerManager({
            'binance': {
                'api_key': self.config.binance_api_key,
                'api_secret': self.config.binance_api_secret
            },
            'alpaca': {
                'api_key': self.config.alpaca_api_key,
                'api_secret': self.config.alpaca_api_secret
            }
        })
        
        # Execution engine
        self.execution_engine = ExecutionEngine(
            broker_manager=self.broker_manager,
            risk_manager=self.risk_manager
        )
        
        # Portfolio management
        self.portfolio_manager = PortfolioManager(
            initial_capital=self.config.initial_capital,
            db_manager=self.db_manager
        )
        
        # Market data streams
        self.market_data_stream = MarketDataStream(
            symbols=self.config.symbols,
            kafka_producer=self.kafka_producer
        )
        
        # News data stream
        self.news_stream = NewsStream(
            api_keys={
                'alpha_vantage': self.config.alpha_vantage_key,
                'newsapi': self.config.newsapi_key
            },
            kafka_producer=self.kafka_producer
        )
        
        # Performance tracking
        self.performance_tracker = PerformanceTracker(
            db_manager=self.db_manager,
            target_success_rate=self.config.target_success_rate
        )
        
        # Notification system
        self.notification_manager = NotificationManager()
        
        # WebSocket connections
        self.websocket_connections = []
        
        logger.info("All components initialized successfully")
    
    def _setup_api_routes(self):
        """Setup REST API routes"""
        
        @self.app.get("/api/status")
        async def get_bot_status():
            """Get current bot status"""
            return {
                "is_running": self.is_running,
                "uptime": time.time() - getattr(self, 'start_time', time.time()),
                "success_rate": await self.performance_tracker.get_success_rate(),
                "total_trades": await self.performance_tracker.get_total_trades(),
                "portfolio_value": await self.portfolio_manager.get_total_value(),
                "active_positions": await self.portfolio_manager.get_active_positions_count(),
                "daily_pnl": await self.performance_tracker.get_daily_pnl(),
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.post("/api/start")
        async def start_bot():
            """Start the trading bot"""
            if self.is_running:
                raise HTTPException(status_code=400, detail="Bot is already running")
            
            await self.start()
            return {"message": "Bot started successfully", "timestamp": datetime.now().isoformat()}
        
        @self.app.post("/api/stop")
        async def stop_bot():
            """Stop the trading bot"""
            if not self.is_running:
                raise HTTPException(status_code=400, detail="Bot is not running")
            
            await self.stop()
            return {"message": "Bot stopped successfully", "timestamp": datetime.now().isoformat()}
        
        @self.app.get("/api/portfolio")
        async def get_portfolio():
            """Get current portfolio"""
            return await self.portfolio_manager.get_portfolio_summary()
        
        @self.app.get("/api/trades")
        async def get_trades(limit: int = 100):
            """Get recent trades"""
            return await self.db_manager.get_recent_trades(limit)
        
        @self.app.get("/api/performance")
        async def get_performance():
            """Get performance metrics"""
            return await self.performance_tracker.get_comprehensive_metrics()
        
        @self.app.get("/api/signals")
        async def get_signals(limit: int = 50):
            """Get recent trading signals"""
            return await self.db_manager.get_recent_signals(limit)
        
        @self.app.post("/api/manual-trade")
        async def manual_trade(trade_data: dict):
            """Execute manual trade"""
            try:
                result = await self.execution_engine.execute_manual_trade(trade_data)
                return {"success": True, "result": result}
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.get("/api/market-data/{symbol}")
        async def get_market_data(symbol: str):
            """Get market data for symbol"""
            return await self.market_data_stream.get_latest_data(symbol)
        
        @self.app.get("/api/ai-insights")
        async def get_ai_insights():
            """Get AI model insights"""
            return await self.ai_engine.get_model_insights()
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}
    
    def _setup_websocket_routes(self):
        """Setup WebSocket routes for real-time data"""
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            self.websocket_connections.append(websocket)
            
            try:
                while True:
                    # Send real-time updates
                    data = await self._get_realtime_data()
                    await websocket.send_json(data)
                    await asyncio.sleep(1)  # Update every second
                    
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
            finally:
                if websocket in self.websocket_connections:
                    self.websocket_connections.remove(websocket)
    
    async def _get_realtime_data(self):
        """Get real-time data for WebSocket clients"""
        return {
            "timestamp": datetime.now().isoformat(),
            "bot_status": {
                "is_running": self.is_running,
                "success_rate": await self.performance_tracker.get_success_rate(),
                "daily_pnl": await self.performance_tracker.get_daily_pnl()
            },
            "portfolio": await self.portfolio_manager.get_portfolio_summary(),
            "latest_signals": await self.db_manager.get_recent_signals(5),
            "market_overview": await self.market_data_stream.get_market_overview()
        }
    
    async def start(self):
        """Start the trading bot"""
        logger.info("🚀 Starting Advanced Trading Bot...")
        self.is_running = True
        self.start_time = time.time()
        
        # Start all components
        await self._start_components()
        
        # Start main trading loop
        asyncio.create_task(self._main_trading_loop())
        
        # Start performance monitoring
        asyncio.create_task(self._performance_monitoring_loop())
        
        # Start model retraining
        asyncio.create_task(self._model_retraining_loop())
        
        logger.info("✅ Trading Bot started successfully")
    
    async def stop(self):
        """Stop the trading bot"""
        logger.info("🛑 Stopping Trading Bot...")
        self.is_running = False
        
        # Close all positions
        await self.execution_engine.close_all_positions()
        
        # Stop all components
        await self._stop_components()
        
        logger.info("✅ Trading Bot stopped successfully")
    
    async def _start_components(self):
        """Start all bot components"""
        await self.db_manager.connect()
        await self.broker_manager.connect_all()
        await self.market_data_stream.start()
        await self.news_stream.start()
        await self.ai_engine.initialize()
        
        logger.info("All components started")
    
    async def _stop_components(self):
        """Stop all bot components"""
        await self.market_data_stream.stop()
        await self.news_stream.stop()
        await self.broker_manager.disconnect_all()
        await self.db_manager.disconnect()
        
        logger.info("All components stopped")
    
    async def _main_trading_loop(self):
        """Main trading loop with 95% success rate optimization"""
        logger.info("Starting main trading loop...")
        
        while self.is_running:
            try:
                # Get market data
                market_data = await self.market_data_stream.get_all_data()
                
                # Get news sentiment
                news_sentiment = await self.news_stream.get_sentiment_analysis()
                
                # Generate AI signals
                signals = await self.ai_engine.generate_signals(
                    market_data=market_data,
                    news_sentiment=news_sentiment,
                    portfolio_state=await self.portfolio_manager.get_state()
                )
                
                # Process and filter signals
                filtered_signals = await self.signal_processor.process_signals(signals)
                
                # Execute trades
                for signal in filtered_signals:
                    if await self.risk_manager.validate_signal(signal):
                        trade_result = await self.execution_engine.execute_signal(signal.original_signal)
                        
                        # Store trade in database
                        await self.db_manager.store_trade(asdict(trade_result))
                        
                        # Send real-time update
                        await self._broadcast_trade_update(trade_result)
                        
                        # Update performance metrics
                        await self.performance_tracker.update_metrics(asdict(trade_result))
                
                # Update portfolio
                await self.portfolio_manager.update_positions()
                
                # Check risk limits
                await self.risk_manager.check_limits()
                
                # Sleep before next iteration
                await asyncio.sleep(5)  # 5-second intervals
                
            except Exception as e:
                logger.error(f"Error in main trading loop: {e}")
                await asyncio.sleep(10)
    
    async def _performance_monitoring_loop(self):
        """Monitor performance and adjust strategies"""
        while self.is_running:
            try:
                # Get current performance
                success_rate = await self.performance_tracker.get_success_rate()
                
                # If success rate drops below target, adjust strategies
                if success_rate < self.config.target_success_rate:
                    logger.warning(f"Success rate ({success_rate:.2%}) below target ({self.config.target_success_rate:.2%})")
                    
                    # Increase AI confidence threshold
                    await self.ai_engine.adjust_confidence_threshold(0.05)
                    
                    # Tighten risk management
                    await self.risk_manager.tighten_limits()
                    
                    # Send notification
                    await self.notification_manager.send_alert(
                        "Performance Alert",
                        f"Success rate dropped to {success_rate:.2%}. Adjusting strategies."
                    )
                
                # Monitor every 5 minutes
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _model_retraining_loop(self):
        """Retrain AI models periodically"""
        while self.is_running:
            try:
                # Wait for retraining interval
                await asyncio.sleep(self.config.retrain_interval_hours * 3600)
                
                logger.info("Starting model retraining...")
                
                # Get recent trading data
                training_data = await self.db_manager.get_training_data()
                
                # Retrain models
                await self.ai_engine.retrain_models(training_data)
                
                logger.info("Model retraining completed")
                
            except Exception as e:
                logger.error(f"Error in model retraining: {e}")
                await asyncio.sleep(3600)  # Retry in 1 hour
    
    async def _broadcast_trade_update(self, trade_result):
        """Broadcast trade update to WebSocket clients"""
        message = {
            "type": "trade_update",
            "data": asdict(trade_result),
            "timestamp": datetime.now().isoformat()
        }
        
        # Send to WebSocket clients
        for websocket in self.websocket_connections:
            try:
                await websocket.send_json(message)
            except:
                pass
        
        # Send to Kafka
        self.kafka_producer.send('trade_updates', message)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        asyncio.create_task(self.stop())
    
    async def run_server(self, host="0.0.0.0", port=8000):
        """Run the FastAPI server"""
        config = uvicorn.Config(
            app=self.app,
            host=host,
            port=port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()

async def main():
    """Main application entry point"""
    # Create directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Load configuration
    config = TradingConfig()
    
    # Initialize trading bot
    bot = AdvancedTradingBot(config)
    
    # Start the bot and server
    await asyncio.gather(
        bot.start(),
        bot.run_server()
    )

if __name__ == "__main__":
    asyncio.run(main())
