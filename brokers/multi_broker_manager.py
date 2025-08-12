#!/usr/bin/env python3
"""
Multi-Broker Manager for Smart Order Routing
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import aiohttp
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class BrokerInfo:
    """Broker information"""
    name: str
    status: str
    latency: float
    commission_rate: float
    available_symbols: List[str]
    liquidity_score: float
    reliability_score: float
    last_updated: datetime

class MultiBrokerManager:
    """Manage multiple brokers for optimal execution"""
    
    def __init__(self, broker_configs: Dict[str, Dict[str, str]]):
        self.broker_configs = broker_configs
        self.brokers = {}
        self.broker_stats = {}
        
        # Initialize broker connections
        self._initialize_brokers()
        
        logger.info(f"Multi-Broker Manager initialized with {len(broker_configs)} brokers")
    
    def _initialize_brokers(self):
        """Initialize broker connections"""
        try:
            for broker_name, config in self.broker_configs.items():
                if broker_name == 'binance':
                    from .binance_broker import BinanceBroker
                    self.brokers[broker_name] = BinanceBroker(config)
                elif broker_name == 'alpaca':
                    from .alpaca_broker import AlpacaBroker
                    self.brokers[broker_name] = AlpacaBroker(config)
                
                # Initialize stats
                self.broker_stats[broker_name] = {
                    'total_orders': 0,
                    'successful_orders': 0,
                    'failed_orders': 0,
                    'avg_latency': 0.0,
                    'uptime': 1.0
                }
            
        except Exception as e:
            logger.error(f"Error initializing brokers: {e}")
    
    async def connect_all(self):
        """Connect to all brokers"""
        try:
            tasks = []
            for broker_name, broker in self.brokers.items():
                tasks.append(self._connect_broker(broker_name, broker))
            
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            logger.error(f"Error connecting to brokers: {e}")
    
    async def _connect_broker(self, broker_name: str, broker):
        """Connect to a specific broker"""
        try:
            await broker.connect()
            logger.info(f"Connected to {broker_name}")
            
        except Exception as e:
            logger.error(f"Error connecting to {broker_name}: {e}")
    
    async def disconnect_all(self):
        """Disconnect from all brokers"""
        try:
            tasks = []
            for broker_name, broker in self.brokers.items():
                tasks.append(broker.disconnect())
            
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            logger.error(f"Error disconnecting from brokers: {e}")
    
    async def get_available_brokers(self, symbol: str) -> List[str]:
        """Get brokers that support the symbol"""
        try:
            available_brokers = []
            
            for broker_name, broker in self.brokers.items():
                if await broker.supports_symbol(symbol):
                    available_brokers.append(broker_name)
            
            return available_brokers
            
        except Exception as e:
            logger.error(f"Error getting available brokers: {e}")
            return []
    
    async def execute_order(self, broker_name: str, order_request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute order through specific broker"""
        try:
            if broker_name not in self.brokers:
                return {'success': False, 'error': f'Broker {broker_name} not available'}
            
            broker = self.brokers[broker_name]
            start_time = datetime.now()
            
            # Execute order
            result = await broker.execute_order(order_request)
            
            # Update statistics
            execution_time = (datetime.now() - start_time).total_seconds()
            await self._update_broker_stats(broker_name, result['success'], execution_time)
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing order through {broker_name}: {e}")
            await self._update_broker_stats(broker_name, False, 0)
            return {'success': False, 'error': str(e)}
    
    async def get_order_status(self, broker_name: str, order_id: str) -> Dict[str, Any]:
        """Get order status from broker"""
        try:
            if broker_name not in self.brokers:
                return {'status': 'UNKNOWN', 'error': f'Broker {broker_name} not available'}
            
            broker = self.brokers[broker_name]
            return await broker.get_order_status(order_id)
            
        except Exception as e:
            logger.error(f"Error getting order status from {broker_name}: {e}")
            return {'status': 'ERROR', 'error': str(e)}
    
    async def cancel_order(self, broker_name: str, order_id: str) -> Dict[str, Any]:
        """Cancel order through broker"""
        try:
            if broker_name not in self.brokers:
                return {'success': False, 'error': f'Broker {broker_name} not available'}
            
            broker = self.brokers[broker_name]
            return await broker.cancel_order(order_id)
            
        except Exception as e:
            logger.error(f"Error cancelling order through {broker_name}: {e}")
            return {'success': False, 'error': str(e)}
    
    async def get_market_data(self, broker_name: str, symbol: str) -> Dict[str, Any]:
        """Get market data from broker"""
        try:
            if broker_name not in self.brokers:
                return {}
            
            broker = self.brokers[broker_name]
            return await broker.get_market_data(symbol)
            
        except Exception as e:
            logger.error(f"Error getting market data from {broker_name}: {e}")
            return {}
    
    async def get_all_positions(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get positions from all brokers"""
        try:
            all_positions = {}
            
            for broker_name, broker in self.brokers.items():
                try:
                    positions = await broker.get_positions()
                    all_positions[broker_name] = positions
                except Exception as e:
                    logger.error(f"Error getting positions from {broker_name}: {e}")
                    all_positions[broker_name] = []
            
            return all_positions
            
        except Exception as e:
            logger.error(f"Error getting all positions: {e}")
            return {}
    
    async def get_broker_latency(self, broker_name: str) -> float:
        """Get broker latency"""
        try:
            stats = self.broker_stats.get(broker_name, {})
            return stats.get('avg_latency', 100.0)  # Default 100ms
            
        except Exception as e:
            logger.error(f"Error getting broker latency: {e}")
            return 100.0
    
    async def get_commission_rate(self, broker_name: str, symbol: str) -> float:
        """Get commission rate for broker and symbol"""
        try:
            if broker_name not in self.brokers:
                return 0.001  # Default 0.1%
            
            broker = self.brokers[broker_name]
            return await broker.get_commission_rate(symbol)
            
        except Exception as e:
            logger.error(f"Error getting commission rate: {e}")
            return 0.001
    
    async def get_liquidity_score(self, broker_name: str, symbol: str) -> float:
        """Get liquidity score for broker and symbol"""
        try:
            if broker_name not in self.brokers:
                return 0.5
            
            broker = self.brokers[broker_name]
            return await broker.get_liquidity_score(symbol)
            
        except Exception as e:
            logger.error(f"Error getting liquidity score: {e}")
            return 0.5
    
    async def get_reliability_score(self, broker_name: str) -> float:
        """Get reliability score for broker"""
        try:
            stats = self.broker_stats.get(broker_name, {})
            total_orders = stats.get('total_orders', 0)
            successful_orders = stats.get('successful_orders', 0)
            
            if total_orders == 0:
                return 1.0
            
            return successful_orders / total_orders
            
        except Exception as e:
            logger.error(f"Error getting reliability score: {e}")
            return 0.5
    
    async def _update_broker_stats(self, broker_name: str, success: bool, execution_time: float):
        """Update broker statistics"""
        try:
            if broker_name not in self.broker_stats:
                self.broker_stats[broker_name] = {
                    'total_orders': 0,
                    'successful_orders': 0,
                    'failed_orders': 0,
                    'avg_latency': 0.0,
                    'uptime': 1.0
                }
            
            stats = self.broker_stats[broker_name]
            
            # Update order counts
            stats['total_orders'] += 1
            if success:
                stats['successful_orders'] += 1
            else:
                stats['failed_orders'] += 1
            
            # Update average latency
            if execution_time > 0:
                old_avg = stats['avg_latency']
                total_orders = stats['total_orders']
                new_avg = (old_avg * (total_orders - 1) + execution_time) / total_orders
                stats['avg_latency'] = new_avg
            
        except Exception as e:
            logger.error(f"Error updating broker stats: {e}")
    
    def get_broker_stats(self) -> Dict[str, Any]:
        """Get broker statistics"""
        return self.broker_stats.copy()
