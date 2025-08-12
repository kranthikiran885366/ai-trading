#!/usr/bin/env python3
"""
Advanced Portfolio Manager with Real-time Position Tracking and Performance Analytics
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
import json

logger = logging.getLogger(__name__)

@dataclass
class Position:
    """Portfolio position data structure"""
    symbol: str
    quantity: float
    avg_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float
    cost_basis: float
    weight: float
    last_updated: datetime
    broker: str

@dataclass
class PortfolioMetrics:
    """Portfolio performance metrics"""
    total_value: float
    total_pnl: float
    daily_pnl: float
    unrealized_pnl: float
    realized_pnl: float
    total_return: float
    daily_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    positions_count: int
    cash_balance: float
    leverage: float

class PortfolioManager:
    """Advanced portfolio management with real-time tracking"""
    
    def __init__(self, initial_capital: float, db_manager):
        self.initial_capital = initial_capital
        self.db_manager = db_manager
        
        # Portfolio state
        self.positions = {}
        self.cash_balance = initial_capital
        self.trade_history = []
        self.daily_snapshots = []
        
        # Performance tracking
        self.performance_metrics = {}
        self.benchmark_data = {}
        
        # Risk metrics
        self.risk_metrics = {
            'var_95': 0.0,
            'expected_shortfall': 0.0,
            'beta': 1.0,
            'correlation': 0.0
        }
        
        logger.info(f"Portfolio Manager initialized with ${initial_capital:,.2f}")
    
    async def update_positions(self):
        """Update all positions with current market data"""
        try:
            if not self.positions:
                return
            
            # Get current prices for all symbols
            symbols = list(self.positions.keys())
            current_prices = await self._get_current_prices(symbols)
            
            total_value = self.cash_balance
            
            for symbol, position in self.positions.items():
                if symbol in current_prices:
                    # Update current price and market value
                    position.current_price = current_prices[symbol]
                    position.market_value = position.quantity * position.current_price
                    
                    # Update unrealized P&L
                    position.unrealized_pnl = (position.current_price - position.avg_price) * position.quantity
                    
                    # Update last updated timestamp
                    position.last_updated = datetime.now()
                    
                    total_value += position.market_value
            
            # Update position weights
            for position in self.positions.values():
                position.weight = position.market_value / total_value if total_value > 0 else 0
            
            # Update performance metrics
            await self._update_performance_metrics()
            
            logger.debug(f"Updated {len(self.positions)} positions, total value: ${total_value:,.2f}")
            
        except Exception as e:
            logger.error(f"Error updating positions: {e}")
    
    async def add_trade(self, trade_data: Dict[str, Any]):
        """Add a new trade to the portfolio"""
        try:
            symbol = trade_data['symbol']
            action = trade_data['action']
            quantity = trade_data['quantity']
            price = trade_data['price']
            commission = trade_data.get('commission', 0)
            timestamp = trade_data.get('timestamp', datetime.now())
            
            # Update cash balance
            if action == 'BUY':
                self.cash_balance -= (quantity * price + commission)
            else:  # SELL
                self.cash_balance += (quantity * price - commission)
            
            # Update or create position
            if symbol in self.positions:
                position = self.positions[symbol]
                
                if action == 'BUY':
                    # Add to position
                    total_cost = position.quantity * position.avg_price + quantity * price
                    total_quantity = position.quantity + quantity
                    
                    if total_quantity > 0:
                        position.avg_price = total_cost / total_quantity
                        position.quantity = total_quantity
                        position.cost_basis = total_cost
                    
                else:  # SELL
                    # Reduce position
                    if position.quantity > 0:
                        # Calculate realized P&L
                        realized_pnl = (price - position.avg_price) * quantity
                        position.realized_pnl += realized_pnl
                        
                        # Update quantity
                        position.quantity -= quantity
                        
                        # If position is closed, remove it
                        if abs(position.quantity) < 1e-6:
                            del self.positions[symbol]
                        else:
                            position.cost_basis = position.quantity * position.avg_price
            
            else:
                # Create new position
                if action == 'BUY':
                    self.positions[symbol] = Position(
                        symbol=symbol,
                        quantity=quantity,
                        avg_price=price,
                        current_price=price,
                        market_value=quantity * price,
                        unrealized_pnl=0.0,
                        realized_pnl=0.0,
                        cost_basis=quantity * price,
                        weight=0.0,
                        last_updated=timestamp,
                        broker=trade_data.get('broker', 'unknown')
                    )
            
            # Store trade in history
            trade_record = {
                'timestamp': timestamp,
                'symbol': symbol,
                'action': action,
                'quantity': quantity,
                'price': price,
                'commission': commission,
                'cash_balance': self.cash_balance
            }
            
            self.trade_history.append(trade_record)
            
            # Store in database
            await self.db_manager.store_trade(trade_record)
            
            logger.info(f"Added trade: {action} {quantity} {symbol} @ ${price:.4f}")
            
        except Exception as e:
            logger.error(f"Error adding trade: {e}")
    
    async def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary"""
        try:
            # Update positions first
            await self.update_positions()
            
            # Calculate totals
            total_market_value = sum(pos.market_value for pos in self.positions.values())
            total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
            total_realized_pnl = sum(pos.realized_pnl for pos in self.positions.values())
            total_value = self.cash_balance + total_market_value
            
            # Calculate returns
            total_return = (total_value - self.initial_capital) / self.initial_capital
            daily_return = await self._calculate_daily_return()
            
            # Get performance metrics
            metrics = await self._calculate_performance_metrics()
            
            summary = {
                'timestamp': datetime.now().isoformat(),
                'total_value': total_value,
                'cash_balance': self.cash_balance,
                'market_value': total_market_value,
                'initial_capital': self.initial_capital,
                'total_pnl': total_value - self.initial_capital,
                'unrealized_pnl': total_unrealized_pnl,
                'realized_pnl': total_realized_pnl,
                'total_return': total_return,
                'daily_return': daily_return,
                'positions_count': len(self.positions),
                'positions': [asdict(pos) for pos in self.positions.values()],
                'performance_metrics': metrics,
                'top_performers': await self._get_top_performers(),
                'sector_allocation': await self._get_sector_allocation(),
                'risk_metrics': self.risk_metrics
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting portfolio summary: {e}")
            return {}
    
    async def get_state(self) -> Dict[str, Any]:
        """Get current portfolio state for AI engine"""
        try:
            await self.update_positions()
            
            total_value = self.cash_balance + sum(pos.market_value for pos in self.positions.values())
            
            state = {
                'total_value': total_value,
                'cash_balance': self.cash_balance,
                'positions': {pos.symbol: pos.quantity for pos in self.positions.values()},
                'position_values': {pos.symbol: pos.market_value for pos in self.positions.values()},
                'unrealized_pnl': sum(pos.unrealized_pnl for pos in self.positions.values()),
                'realized_pnl': sum(pos.realized_pnl for pos in self.positions.values()),
                'total_return': (total_value - self.initial_capital) / self.initial_capital,
                'heat_level': await self._calculate_heat_level(),
                'correlation_score': await self._calculate_correlation_score(),
                'recent_performance': await self._get_recent_performance(),
                'drawdown_level': await self._calculate_current_drawdown(),
                'win_rate': await self._calculate_win_rate(),
                'avg_win': await self._calculate_avg_win(),
                'avg_loss': await self._calculate_avg_loss(),
                'sharpe_ratio': await self._calculate_sharpe_ratio(),
                'correlations': await self._get_position_correlations()
            }
            
            return state
            
        except Exception as e:
            logger.error(f"Error getting portfolio state: {e}")
            return {}
    
    async def get_total_value(self) -> float:
        """Get total portfolio value"""
        try:
            await self.update_positions()
            market_value = sum(pos.market_value for pos in self.positions.values())
            return self.cash_balance + market_value
            
        except Exception as e:
            logger.error(f"Error getting total value: {e}")
            return self.initial_capital
    
    async def get_active_positions_count(self) -> int:
        """Get number of active positions"""
        return len(self.positions)
    
    async def _get_current_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get current prices for symbols"""
        try:
            # This would integrate with market data provider
            # For now, return mock prices
            prices = {}
            
            for symbol in symbols:
                # Mock price with some randomness
                base_price = 100.0
                if symbol == 'AAPL':
                    base_price = 150.0
                elif symbol == 'GOOGL':
                    base_price = 2500.0
                elif symbol == 'BTCUSDT':
                    base_price = 45000.0
                elif symbol == 'ETHUSDT':
                    base_price = 3000.0
                
                # Add some random movement
                import random
                price_change = random.uniform(-0.02, 0.02)  # ±2%
                prices[symbol] = base_price * (1 + price_change)
            
            return prices
            
        except Exception as e:
            logger.error(f"Error getting current prices: {e}")
            return {}
    
    async def _update_performance_metrics(self):
        """Update performance metrics"""
        try:
            total_value = await self.get_total_value()
            
            # Store daily snapshot
            snapshot = {
                'date': datetime.now().date(),
                'total_value': total_value,
                'cash_balance': self.cash_balance,
                'positions_count': len(self.positions),
                'total_return': (total_value - self.initial_capital) / self.initial_capital
            }
            
            # Add to daily snapshots
            self.daily_snapshots.append(snapshot)
            
            # Keep only last year of snapshots
            if len(self.daily_snapshots) > 365:
                self.daily_snapshots = self.daily_snapshots[-365:]
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    async def _calculate_daily_return(self) -> float:
        """Calculate daily return"""
        try:
            if len(self.daily_snapshots) < 2:
                return 0.0
            
            today_value = self.daily_snapshots[-1]['total_value']
            yesterday_value = self.daily_snapshots[-2]['total_value']
            
            return (today_value - yesterday_value) / yesterday_value
            
        except Exception as e:
            logger.error(f"Error calculating daily return: {e}")
            return 0.0
    
    async def _calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        try:
            if len(self.daily_snapshots) < 30:
                return {}
            
            # Get returns series
            values = [snapshot['total_value'] for snapshot in self.daily_snapshots]
            returns = [(values[i] - values[i-1]) / values[i-1] for i in range(1, len(values))]
            
            metrics = {
                'sharpe_ratio': await self._calculate_sharpe_ratio(),
                'sortino_ratio': await self._calculate_sortino_ratio(),
                'max_drawdown': await self._calculate_max_drawdown(),
                'volatility': np.std(returns) * np.sqrt(252) if returns else 0,
                'win_rate': await self._calculate_win_rate(),
                'profit_factor': await self._calculate_profit_factor(),
                'calmar_ratio': await self._calculate_calmar_ratio()
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {}
    
    async def _calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        try:
            if len(self.daily_snapshots) < 30:
                return 0.0
            
            values = [snapshot['total_value'] for snapshot in self.daily_snapshots]
            returns = [(values[i] - values[i-1]) / values[i-1] for i in range(1, len(values))]
            
            if not returns:
                return 0.0
            
            excess_returns = [r - risk_free_rate/252 for r in returns]
            
            if np.std(returns) == 0:
                return 0.0
            
            return np.mean(excess_returns) / np.std(returns) * np.sqrt(252)
            
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0
    
    async def _calculate_sortino_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio"""
        try:
            if len(self.daily_snapshots) < 30:
                return 0.0
            
            values = [snapshot['total_value'] for snapshot in self.daily_snapshots]
            returns = [(values[i] - values[i-1]) / values[i-1] for i in range(1, len(values))]
            
            if not returns:
                return 0.0
            
            excess_returns = [r - risk_free_rate/252 for r in returns]
            negative_returns = [r for r in excess_returns if r < 0]
            
            if not negative_returns:
                return float('inf')
            
            downside_deviation = np.sqrt(np.mean([r**2 for r in negative_returns]))
            
            if downside_deviation == 0:
                return 0.0
            
            return np.mean(excess_returns) / downside_deviation * np.sqrt(252)
            
        except Exception as e:
            logger.error(f"Error calculating Sortino ratio: {e}")
            return 0.0
    
    async def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        try:
            if len(self.daily_snapshots) < 2:
                return 0.0
            
            values = [snapshot['total_value'] for snapshot in self.daily_snapshots]
            peak = values[0]
            max_dd = 0.0
            
            for value in values:
                if value > peak:
                    peak = value
                
                drawdown = (peak - value) / peak
                if drawdown > max_dd:
                    max_dd = drawdown
            
            return max_dd
            
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {e}")
            return 0.0
    
    async def _calculate_win_rate(self) -> float:
        """Calculate win rate from trade history"""
        try:
            if not self.trade_history:
                return 0.5
            
            # Calculate P&L for each trade
            winning_trades = 0
            total_trades = 0
            
            for i, trade in enumerate(self.trade_history):
                if trade['action'] == 'SELL' and i > 0:
                    # Find corresponding buy trade
                    for j in range(i-1, -1, -1):
                        prev_trade = self.trade_history[j]
                        if (prev_trade['symbol'] == trade['symbol'] and 
                            prev_trade['action'] == 'BUY'):
                            
                            pnl = (trade['price'] - prev_trade['price']) * trade['quantity']
                            if pnl > 0:
                                winning_trades += 1
                            total_trades += 1
                            break
            
            return winning_trades / total_trades if total_trades > 0 else 0.5
            
        except Exception as e:
            logger.error(f"Error calculating win rate: {e}")
            return 0.5
    
    async def _calculate_profit_factor(self) -> float:
        """Calculate profit factor"""
        try:
            if not self.trade_history:
                return 1.0
            
            gross_profit = 0.0
            gross_loss = 0.0
            
            for i, trade in enumerate(self.trade_history):
                if trade['action'] == 'SELL' and i > 0:
                    # Find corresponding buy trade
                    for j in range(i-1, -1, -1):
                        prev_trade = self.trade_history[j]
                        if (prev_trade['symbol'] == trade['symbol'] and 
                            prev_trade['action'] == 'BUY'):
                            
                            pnl = (trade['price'] - prev_trade['price']) * trade['quantity']
                            if pnl > 0:
                                gross_profit += pnl
                            else:
                                gross_loss += abs(pnl)
                            break
            
            return gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
        except Exception as e:
            logger.error(f"Error calculating profit factor: {e}")
            return 1.0
    
    async def _calculate_calmar_ratio(self) -> float:
        """Calculate Calmar ratio"""
        try:
            if len(self.daily_snapshots) < 365:
                return 0.0
            
            # Annual return
            start_value = self.daily_snapshots[0]['total_value']
            end_value = self.daily_snapshots[-1]['total_value']
            annual_return = (end_value - start_value) / start_value
            
            # Max drawdown
            max_dd = await self._calculate_max_drawdown()
            
            return annual_return / max_dd if max_dd > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating Calmar ratio: {e}")
            return 0.0
    
    async def _get_top_performers(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get top performing positions"""
        try:
            performers = []
            
            for position in self.positions.values():
                return_pct = (position.unrealized_pnl + position.realized_pnl) / position.cost_basis
                
                performers.append({
                    'symbol': position.symbol,
                    'return_pct': return_pct,
                    'pnl': position.unrealized_pnl + position.realized_pnl,
                    'weight': position.weight
                })
            
            # Sort by return percentage
            performers.sort(key=lambda x: x['return_pct'], reverse=True)
            
            return performers[:limit]
            
        except Exception as e:
            logger.error(f"Error getting top performers: {e}")
            return []
    
    async def _get_sector_allocation(self) -> Dict[str, float]:
        """Get sector allocation"""
        try:
            sector_values = {}
            total_value = sum(pos.market_value for pos in self.positions.values())
            
            if total_value == 0:
                return {}
            
            for position in self.positions.values():
                sector = await self._get_symbol_sector(position.symbol)
                
                if sector not in sector_values:
                    sector_values[sector] = 0
                
                sector_values[sector] += position.market_value
            
            # Convert to percentages
            sector_allocation = {
                sector: value / total_value 
                for sector, value in sector_values.items()
            }
            
            return sector_allocation
            
        except Exception as e:
            logger.error(f"Error getting sector allocation: {e}")
            return {}
    
    async def _get_symbol_sector(self, symbol: str) -> str:
        """Get sector for symbol"""
        try:
            sectors = {
                'AAPL': 'Technology',
                'MSFT': 'Technology',
                'GOOGL': 'Technology',
                'AMZN': 'Consumer Discretionary',
                'TSLA': 'Consumer Discretionary',
                'NVDA': 'Technology',
                'META': 'Technology',
                'NFLX': 'Communication Services',
                'AMD': 'Technology',
                'CRM': 'Technology',
                'BTCUSDT': 'Cryptocurrency',
                'ETHUSDT': 'Cryptocurrency',
                'ADAUSDT': 'Cryptocurrency',
                'DOTUSDT': 'Cryptocurrency'
            }
            
            return sectors.get(symbol, 'Other')
            
        except Exception as e:
            logger.error(f"Error getting symbol sector: {e}")
            return 'Other'
    
    async def _calculate_heat_level(self) -> float:
        """Calculate portfolio heat level"""
        try:
            total_value = await self.get_total_value()
            if total_value == 0:
                return 0.0
            
            total_exposure = sum(abs(pos.market_value) for pos in self.positions.values())
            return total_exposure / total_value
            
        except Exception as e:
            logger.error(f"Error calculating heat level: {e}")
            return 0.0
    
    async def _calculate_correlation_score(self) -> float:
        """Calculate average correlation score"""
        try:
            if len(self.positions) < 2:
                return 0.0
            
            # Simplified correlation calculation
            # In practice, this would use historical price correlations
            symbols = list(self.positions.keys())
            correlations = []
            
            for i, symbol1 in enumerate(symbols):
                for symbol2 in symbols[i+1:]:
                    # Mock correlation based on sectors
                    sector1 = await self._get_symbol_sector(symbol1)
                    sector2 = await self._get_symbol_sector(symbol2)
                    
                    if sector1 == sector2:
                        correlation = 0.7  # High correlation within sector
                    else:
                        correlation = 0.3  # Lower correlation across sectors
                    
                    correlations.append(correlation)
            
            return np.mean(correlations) if correlations else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating correlation score: {e}")
            return 0.0
    
    async def _get_recent_performance(self) -> float:
        """Get recent performance (last 7 days)"""
        try:
            if len(self.daily_snapshots) < 7:
                return 0.0
            
            current_value = self.daily_snapshots[-1]['total_value']
            week_ago_value = self.daily_snapshots[-7]['total_value']
            
            return (current_value - week_ago_value) / week_ago_value
            
        except Exception as e:
            logger.error(f"Error getting recent performance: {e}")
            return 0.0
    
    async def _calculate_current_drawdown(self) -> float:
        """Calculate current drawdown"""
        try:
            if len(self.daily_snapshots) < 2:
                return 0.0
            
            values = [snapshot['total_value'] for snapshot in self.daily_snapshots]
            current_value = values[-1]
            peak_value = max(values)
            
            if peak_value == 0:
                return 0.0
            
            return (peak_value - current_value) / peak_value
            
        except Exception as e:
            logger.error(f"Error calculating current drawdown: {e}")
            return 0.0
    
    async def _calculate_avg_win(self) -> float:
        """Calculate average winning trade"""
        try:
            if not self.trade_history:
                return 0.02
            
            winning_trades = []
            
            for i, trade in enumerate(self.trade_history):
                if trade['action'] == 'SELL' and i > 0:
                    for j in range(i-1, -1, -1):
                        prev_trade = self.trade_history[j]
                        if (prev_trade['symbol'] == trade['symbol'] and 
                            prev_trade['action'] == 'BUY'):
                            
                            pnl = (trade['price'] - prev_trade['price']) * trade['quantity']
                            if pnl > 0:
                                return_pct = pnl / (prev_trade['price'] * prev_trade['quantity'])
                                winning_trades.append(return_pct)
                            break
            
            return np.mean(winning_trades) if winning_trades else 0.02
            
        except Exception as e:
            logger.error(f"Error calculating avg win: {e}")
            return 0.02
    
    async def _calculate_avg_loss(self) -> float:
        """Calculate average losing trade"""
        try:
            if not self.trade_history:
                return 0.01
            
            losing_trades = []
            
            for i, trade in enumerate(self.trade_history):
                if trade['action'] == 'SELL' and i > 0:
                    for j in range(i-1, -1, -1):
                        prev_trade = self.trade_history[j]
                        if (prev_trade['symbol'] == trade['symbol'] and 
                            prev_trade['action'] == 'BUY'):
                            
                            pnl = (trade['price'] - prev_trade['price']) * trade['quantity']
                            if pnl < 0:
                                return_pct = abs(pnl) / (prev_trade['price'] * prev_trade['quantity'])
                                losing_trades.append(return_pct)
                            break
            
            return np.mean(losing_trades) if losing_trades else 0.01
            
        except Exception as e:
            logger.error(f"Error calculating avg loss: {e}")
            return 0.01
    
    async def _get_position_correlations(self) -> Dict[str, float]:
        """Get correlations for each position"""
        try:
            correlations = {}
            
            for symbol in self.positions.keys():
                # Mock correlation calculation
                sector = await self._get_symbol_sector(symbol)
                
                if sector == 'Technology':
                    correlations[symbol] = 0.6
                elif sector == 'Cryptocurrency':
                    correlations[symbol] = 0.8
                else:
                    correlations[symbol] = 0.4
            
            return correlations
            
        except Exception as e:
            logger.error(f"Error getting position correlations: {e}")
            return {}
    
    def get_portfolio_metrics(self) -> PortfolioMetrics:
        """Get portfolio metrics object"""
        try:
            total_value = asyncio.run(self.get_total_value())
            total_pnl = total_value - self.initial_capital
            
            return PortfolioMetrics(
                total_value=total_value,
                total_pnl=total_pnl,
                daily_pnl=asyncio.run(self._calculate_daily_return()) * total_value,
                unrealized_pnl=sum(pos.unrealized_pnl for pos in self.positions.values()),
                realized_pnl=sum(pos.realized_pnl for pos in self.positions.values()),
                total_return=total_pnl / self.initial_capital,
                daily_return=asyncio.run(self._calculate_daily_return()),
                sharpe_ratio=asyncio.run(self._calculate_sharpe_ratio()),
                max_drawdown=asyncio.run(self._calculate_max_drawdown()),
                win_rate=asyncio.run(self._calculate_win_rate()),
                profit_factor=asyncio.run(self._calculate_profit_factor()),
                positions_count=len(self.positions),
                cash_balance=self.cash_balance,
                leverage=asyncio.run(self._calculate_heat_level())
            )
            
        except Exception as e:
            logger.error(f"Error getting portfolio metrics: {e}")
            return PortfolioMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
