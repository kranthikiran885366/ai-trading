#!/usr/bin/env python3
"""
Performance Tracker with Real-time Metrics and Analytics
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class TradeResult:
    """Trade result for performance tracking"""
    symbol: str
    action: str
    entry_price: float
    exit_price: float
    quantity: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    return_pct: float
    holding_period: float
    commission: float

class PerformanceTracker:
    """Track and analyze trading performance"""
    
    def __init__(self, db_manager, target_success_rate: float = 0.95):
        self.db_manager = db_manager
        self.target_success_rate = target_success_rate
        
        # Performance data
        self.trade_results = []
        self.daily_pnl = []
        self.portfolio_values = []
        
        # Metrics cache
        self.metrics_cache = {}
        self.cache_expiry = {}
        
        logger.info(f"Performance Tracker initialized with {target_success_rate:.1%} target success rate")
    
    async def update_metrics(self, trade_result: Dict[str, Any]):
        """Update performance metrics with new trade result"""
        try:
            # Convert to TradeResult if needed
            if isinstance(trade_result, dict):
                trade_result = self._dict_to_trade_result(trade_result)
            
            # Add to trade results
            self.trade_results.append(trade_result)
            
            # Keep only recent trades (last 1000)
            if len(self.trade_results) > 1000:
                self.trade_results = self.trade_results[-500:]
            
            # Update daily P&L
            await self._update_daily_pnl()
            
            # Clear metrics cache
            self.metrics_cache.clear()
            self.cache_expiry.clear()
            
            logger.debug(f"Updated metrics with trade: {trade_result.symbol} P&L: ${trade_result.pnl:.2f}")
            
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
    
    def _dict_to_trade_result(self, trade_dict: Dict[str, Any]) -> TradeResult:
        """Convert dictionary to TradeResult"""
        try:
            return TradeResult(
                symbol=trade_dict.get('symbol', ''),
                action=trade_dict.get('action', ''),
                entry_price=trade_dict.get('entry_price', 0.0),
                exit_price=trade_dict.get('exit_price', 0.0),
                quantity=trade_dict.get('quantity', 0.0),
                entry_time=trade_dict.get('entry_time', datetime.now()),
                exit_time=trade_dict.get('exit_time', datetime.now()),
                pnl=trade_dict.get('pnl', 0.0),
                return_pct=trade_dict.get('return_pct', 0.0),
                holding_period=trade_dict.get('holding_period', 0.0),
                commission=trade_dict.get('commission', 0.0)
            )
            
        except Exception as e:
            logger.error(f"Error converting dict to TradeResult: {e}")
            return TradeResult('', '', 0, 0, 0, datetime.now(), datetime.now(), 0, 0, 0, 0)
    
    async def _update_daily_pnl(self):
        """Update daily P&L tracking"""
        try:
            # Group trades by date
            daily_pnl = {}
            
            for trade in self.trade_results:
                date = trade.exit_time.date()
                if date not in daily_pnl:
                    daily_pnl[date] = 0.0
                daily_pnl[date] += trade.pnl
            
            # Convert to sorted list
            self.daily_pnl = [
                {'date': date, 'pnl': pnl}
                for date, pnl in sorted(daily_pnl.items())
            ]
            
        except Exception as e:
            logger.error(f"Error updating daily P&L: {e}")
    
    async def get_success_rate(self) -> float:
        """Get current success rate"""
        try:
            cache_key = 'success_rate'
            if self._is_cached(cache_key):
                return self.metrics_cache[cache_key]
            
            if not self.trade_results:
                return 0.5
            
            winning_trades = sum(1 for trade in self.trade_results if trade.pnl > 0)
            success_rate = winning_trades / len(self.trade_results)
            
            self._cache_metric(cache_key, success_rate)
            return success_rate
            
        except Exception as e:
            logger.error(f"Error calculating success rate: {e}")
            return 0.5
    
    async def get_total_trades(self) -> int:
        """Get total number of trades"""
        return len(self.trade_results)
    
    async def get_daily_pnl(self) -> float:
        """Get today's P&L"""
        try:
            today = datetime.now().date()
            
            for daily_data in reversed(self.daily_pnl):
                if daily_data['date'] == today:
                    return daily_data['pnl']
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error getting daily P&L: {e}")
            return 0.0
    
    async def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        try:
            cache_key = 'comprehensive_metrics'
            if self._is_cached(cache_key):
                return self.metrics_cache[cache_key]
            
            if not self.trade_results:
                return {}
            
            # Basic metrics
            total_trades = len(self.trade_results)
            winning_trades = sum(1 for trade in self.trade_results if trade.pnl > 0)
            losing_trades = total_trades - winning_trades
            
            # P&L metrics
            total_pnl = sum(trade.pnl for trade in self.trade_results)
            gross_profit = sum(trade.pnl for trade in self.trade_results if trade.pnl > 0)
            gross_loss = sum(abs(trade.pnl) for trade in self.trade_results if trade.pnl < 0)
            
            # Return metrics
            returns = [trade.return_pct for trade in self.trade_results]
            avg_return = np.mean(returns) if returns else 0
            return_std = np.std(returns) if returns else 0
            
            # Win/Loss metrics
            winning_returns = [trade.return_pct for trade in self.trade_results if trade.pnl > 0]
            losing_returns = [abs(trade.return_pct) for trade in self.trade_results if trade.pnl < 0]
            
            avg_win = np.mean(winning_returns) if winning_returns else 0
            avg_loss = np.mean(losing_returns) if losing_returns else 0
            
            # Risk metrics
            sharpe_ratio = await self._calculate_sharpe_ratio()
            max_drawdown = await self._calculate_max_drawdown()
            
            # Time metrics
            holding_periods = [trade.holding_period for trade in self.trade_results]
            avg_holding_period = np.mean(holding_periods) if holding_periods else 0
            
            # Symbol performance
            symbol_performance = await self._calculate_symbol_performance()
            
            # Recent performance (last 30 days)
            recent_performance = await self._calculate_recent_performance(30)
            
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
                'total_pnl': total_pnl,
                'gross_profit': gross_profit,
                'gross_loss': gross_loss,
                'profit_factor': gross_profit / gross_loss if gross_loss > 0 else float('inf'),
                'avg_return': avg_return,
                'return_std': return_std,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'avg_win_loss_ratio': avg_win / avg_loss if avg_loss > 0 else float('inf'),
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'avg_holding_period': avg_holding_period,
                'symbol_performance': symbol_performance,
                'recent_performance': recent_performance,
                'target_success_rate': self.target_success_rate,
                'success_rate_gap': (winning_trades / total_trades) - self.target_success_rate if total_trades > 0 else -self.target_success_rate
            }
            
            self._cache_metric(cache_key, metrics)
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating comprehensive metrics: {e}")
            return {}
    
    async def _calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        try:
            if not self.trade_results:
                return 0.0
            
            returns = [trade.return_pct for trade in self.trade_results]
            
            if not returns or np.std(returns) == 0:
                return 0.0
            
            excess_returns = [r - risk_free_rate/252 for r in returns]  # Daily risk-free rate
            
            return np.mean(excess_returns) / np.std(returns) * np.sqrt(252)  # Annualized
            
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0
    
    async def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        try:
            if not self.daily_pnl:
                return 0.0
            
            # Calculate cumulative P&L
            cumulative_pnl = []
            running_total = 0
            
            for daily_data in self.daily_pnl:
                running_total += daily_data['pnl']
                cumulative_pnl.append(running_total)
            
            if not cumulative_pnl:
                return 0.0
            
            # Calculate drawdown
            peak = cumulative_pnl[0]
            max_dd = 0.0
            
            for value in cumulative_pnl:
                if value > peak:
                    peak = value
                
                drawdown = (peak - value) / abs(peak) if peak != 0 else 0
                if drawdown > max_dd:
                    max_dd = drawdown
            
            return max_dd
            
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {e}")
            return 0.0
    
    async def _calculate_symbol_performance(self) -> Dict[str, Dict[str, Any]]:
        """Calculate performance by symbol"""
        try:
            symbol_stats = {}
            
            for trade in self.trade_results:
                symbol = trade.symbol
                
                if symbol not in symbol_stats:
                    symbol_stats[symbol] = {
                        'total_trades': 0,
                        'winning_trades': 0,
                        'total_pnl': 0.0,
                        'returns': []
                    }
                
                stats = symbol_stats[symbol]
                stats['total_trades'] += 1
                stats['total_pnl'] += trade.pnl
                stats['returns'].append(trade.return_pct)
                
                if trade.pnl > 0:
                    stats['winning_trades'] += 1
            
            # Calculate derived metrics
            for symbol, stats in symbol_stats.items():
                stats['win_rate'] = stats['winning_trades'] / stats['total_trades']
                stats['avg_return'] = np.mean(stats['returns'])
                stats['return_std'] = np.std(stats['returns'])
                stats['sharpe_ratio'] = stats['avg_return'] / stats['return_std'] if stats['return_std'] > 0 else 0
                
                # Remove raw returns to reduce size
                del stats['returns']
            
            return symbol_stats
            
        except Exception as e:
            logger.error(f"Error calculating symbol performance: {e}")
            return {}
    
    async def _calculate_recent_performance(self, days: int) -> Dict[str, Any]:
        """Calculate recent performance metrics"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            recent_trades = [
                trade for trade in self.trade_results
                if trade.exit_time > cutoff_date
            ]
            
            if not recent_trades:
                return {}
            
            total_trades = len(recent_trades)
            winning_trades = sum(1 for trade in recent_trades if trade.pnl > 0)
            total_pnl = sum(trade.pnl for trade in recent_trades)
            
            returns = [trade.return_pct for trade in recent_trades]
            avg_return = np.mean(returns) if returns else 0
            
            return {
                'period_days': days,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'win_rate': winning_trades / total_trades,
                'total_pnl': total_pnl,
                'avg_return': avg_return,
                'trades_per_day': total_trades / days
            }
            
        except Exception as e:
            logger.error(f"Error calculating recent performance: {e}")
            return {}
    
    def _is_cached(self, key: str) -> bool:
        """Check if metric is cached and not expired"""
        try:
            if key not in self.metrics_cache:
                return False
            
            if key not in self.cache_expiry:
                return False
            
            return datetime.now() < self.cache_expiry[key]
            
        except Exception as e:
            logger.error(f"Error checking cache: {e}")
            return False
    
    def _cache_metric(self, key: str, value: Any, expiry_minutes: int = 5):
        """Cache metric with expiry"""
        try:
            self.metrics_cache[key] = value
            self.cache_expiry[key] = datetime.now() + timedelta(minutes=expiry_minutes)
            
        except Exception as e:
            logger.error(f"Error caching metric: {e}")
    
    async def get_performance_report(self) -> str:
        """Generate a formatted performance report"""
        try:
            metrics = await self.get_comprehensive_metrics()
            
            if not metrics:
                return "No performance data available"
            
            report = f"""
TRADING PERFORMANCE REPORT
==========================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

OVERALL PERFORMANCE
-------------------
Total Trades: {metrics['total_trades']}
Win Rate: {metrics['win_rate']:.2%}
Target Success Rate: {metrics['target_success_rate']:.2%}
Success Rate Gap: {metrics['success_rate_gap']:+.2%}

PROFIT & LOSS
-------------
Total P&L: ${metrics['total_pnl']:,.2f}
Gross Profit: ${metrics['gross_profit']:,.2f}
Gross Loss: ${metrics['gross_loss']:,.2f}
Profit Factor: {metrics['profit_factor']:.2f}

RETURNS
-------
Average Return: {metrics['avg_return']:.2%}
Return Std Dev: {metrics['return_std']:.2%}
Average Win: {metrics['avg_win']:.2%}
Average Loss: {metrics['avg_loss']:.2%}
Win/Loss Ratio: {metrics['avg_win_loss_ratio']:.2f}

RISK METRICS
------------
Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
Maximum Drawdown: {metrics['max_drawdown']:.2%}
Average Holding Period: {metrics['avg_holding_period']:.1f} hours

RECENT PERFORMANCE (30 days)
----------------------------
"""
            
            if metrics.get('recent_performance'):
                recent = metrics['recent_performance']
                report += f"""Trades: {recent['total_trades']}
Win Rate: {recent['win_rate']:.2%}
Total P&L: ${recent['total_pnl']:,.2f}
Avg Return: {recent['avg_return']:.2%}
Trades/Day: {recent['trades_per_day']:.1f}
"""
            else:
                report += "No recent performance data available"
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return "Error generating performance report"
    
    async def analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends"""
        try:
            if len(self.daily_pnl) < 7:
                return {'error': 'Insufficient data for trend analysis'}
            
            # Get recent daily P&L
            recent_pnl = [item['pnl'] for item in self.daily_pnl[-30:]]  # Last 30 days
            
            # Calculate trends
            trend_slope = np.polyfit(range(len(recent_pnl)), recent_pnl, 1)[0]
            
            # Performance consistency
            positive_days = sum(1 for pnl in recent_pnl if pnl > 0)
            consistency = positive_days / len(recent_pnl)
            
            # Volatility trend
            volatility = np.std(recent_pnl)
            
            # Best and worst days
            best_day = max(recent_pnl)
            worst_day = min(recent_pnl)
            
            return {
                'trend_direction': 'improving' if trend_slope > 0 else 'declining',
                'trend_slope': trend_slope,
                'consistency': consistency,
                'volatility': volatility,
                'best_day_pnl': best_day,
                'worst_day_pnl': worst_day,
                'analysis_period_days': len(recent_pnl)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing performance trends: {e}")
            return {'error': str(e)}
    
    def get_trade_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent trade history"""
        try:
            recent_trades = self.trade_results[-limit:] if limit else self.trade_results
            
            return [
                {
                    'symbol': trade.symbol,
                    'action': trade.action,
                    'entry_price': trade.entry_price,
                    'exit_price': trade.exit_price,
                    'quantity': trade.quantity,
                    'pnl': trade.pnl,
                    'return_pct': trade.return_pct,
                    'holding_period': trade.holding_period,
                    'entry_time': trade.entry_time.isoformat(),
                    'exit_time': trade.exit_time.isoformat()
                }
                for trade in recent_trades
            ]
            
        except Exception as e:
            logger.error(f"Error getting trade history: {e}")
            return []
