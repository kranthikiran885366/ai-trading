#!/usr/bin/env python3
"""
Performance Monitor - Tracks and analyzes trading performance
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
import json
import os
import sqlite3
from collections import deque
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    timestamp: datetime
    total_return: float
    daily_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    average_win: float
    average_loss: float
    largest_win: float
    largest_loss: float
    consecutive_wins: int
    consecutive_losses: int
    portfolio_value: float
    cash_balance: float
    total_pnl: float
    unrealized_pnl: float
    realized_pnl: float
    total_fees: float
    net_pnl: float
    max_drawdown_duration: int
    avg_trade_duration: float

@dataclass
class TradeMetrics:
    """Individual trade metrics"""
    trade_id: str
    symbol: str
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    side: str  # BUY/SELL
    pnl: float
    pnl_percent: float
    duration: Optional[timedelta]
    strategy: str
    confidence: float
    fees: float

class PerformanceMonitor:
    """Advanced performance monitoring and analytics"""
    
    def __init__(self, initial_capital: float = 100000.0, db_path: str = "performance.db"):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.is_running = False
        self.monitor_thread = None
        self.db_path = db_path
        
        # Performance tracking
        self.trades = []
        self.daily_returns = []
        self.portfolio_values = []
        self.drawdown_history = []
        self.metrics_history = []
        
        # Current metrics
        self.current_metrics = None
        
        # Settings
        self.update_interval = 60  # seconds
        self.risk_free_rate = 0.02  # 2% annual
        
        # File paths
        self.data_dir = "data/performance"
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Real-time tracking
        self.current_positions = {}
        self.trade_history = []
        self.equity_curve = deque(maxlen=10000)
        self.drawdown_series = deque(maxlen=10000)
        
        # Performance caches
        self.performance_cache = {}
        self.last_update = datetime.now()
        
        # Initialize database
        self._init_database()
        
        logger.info("Performance Monitor initialized")
    
    def _init_database(self):
        """Initialize SQLite database for performance tracking"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    symbol TEXT,
                    entry_time TIMESTAMP,
                    exit_time TIMESTAMP,
                    entry_price REAL,
                    exit_price REAL,
                    quantity REAL,
                    side TEXT,
                    pnl REAL,
                    pnl_percent REAL,
                    duration_hours REAL,
                    commission REAL,
                    strategy TEXT,
                    max_favorable_excursion REAL,
                    max_adverse_excursion REAL,
                    fees REAL
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS daily_performance (
                    date DATE PRIMARY KEY,
                    portfolio_value REAL,
                    daily_return REAL,
                    cumulative_return REAL,
                    drawdown REAL,
                    trades_count INTEGER,
                    winning_trades INTEGER,
                    losing_trades INTEGER,
                    total_volume REAL,
                    total_fees REAL,
                    equity_value REAL,
                    volatility REAL
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS positions (
                    symbol TEXT PRIMARY KEY,
                    quantity REAL,
                    avg_price REAL,
                    current_price REAL,
                    market_value REAL,
                    unrealized_pnl REAL,
                    last_updated TIMESTAMP,
                    broker TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
    
    def start(self):
        """Start performance monitoring"""
        if not self.is_running:
            self.is_running = True
            self.monitor_thread = threading.Thread(target=self._monitoring_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            logger.info("Performance monitoring started")
    
    def stop(self):
        """Stop performance monitoring"""
        self.is_running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_running:
            try:
                self.update()
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.update_interval)
    
    def update(self):
        """Update performance metrics"""
        try:
            # Calculate current metrics
            metrics = self._calculate_metrics()
            
            if metrics:
                self.current_metrics = metrics
                self.metrics_history.append(metrics)
                
                # Save metrics to file
                self._save_metrics(metrics)
                
                # Log key metrics
                if len(self.metrics_history) % 10 == 0:  # Log every 10 updates
                    logger.info(f"Performance Update - Total Return: {metrics.total_return:.2%}, "
                              f"Sharpe: {metrics.sharpe_ratio:.2f}, "
                              f"Max Drawdown: {metrics.max_drawdown:.2%}")
        
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    def _calculate_metrics(self) -> Optional[PerformanceMetrics]:
        """Calculate comprehensive performance metrics"""
        try:
            if not self.trades and not self.portfolio_values:
                return None
            
            # Basic calculations
            current_value = self.portfolio_values[-1] if self.portfolio_values else self.initial_capital
            total_return = (current_value - self.initial_capital) / self.initial_capital
            
            # Daily return
            daily_return = 0.0
            if len(self.portfolio_values) >= 2:
                daily_return = (self.portfolio_values[-1] - self.portfolio_values[-2]) / self.portfolio_values[-2]
            
            # Sharpe ratio
            sharpe_ratio = self._calculate_sharpe_ratio()
            
            # Maximum drawdown
            max_drawdown, max_dd_duration = self._calculate_max_drawdown()
            
            # Trade statistics
            trade_stats = self._calculate_trade_statistics()
            
            # PnL calculations
            total_pnl = current_value - self.initial_capital
            realized_pnl = sum(trade.pnl for trade in self.trades if trade.exit_time is not None)
            unrealized_pnl = total_pnl - realized_pnl
            total_fees = sum(trade.fees for trade in self.trades)
            net_pnl = total_pnl - total_fees
            
            # Average trade duration
            avg_trade_duration = sum(trade.duration.total_seconds() / 3600 for trade in self.trades if trade.duration) / len(self.trades) if self.trades else 0
            
            return PerformanceMetrics(
                timestamp=datetime.now(),
                total_return=total_return,
                daily_return=daily_return,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=trade_stats['win_rate'],
                profit_factor=trade_stats['profit_factor'],
                total_trades=trade_stats['total_trades'],
                winning_trades=trade_stats['winning_trades'],
                losing_trades=trade_stats['losing_trades'],
                average_win=trade_stats['average_win'],
                average_loss=trade_stats['average_loss'],
                largest_win=trade_stats['largest_win'],
                largest_loss=trade_stats['largest_loss'],
                consecutive_wins=trade_stats['consecutive_wins'],
                consecutive_losses=trade_stats['consecutive_losses'],
                portfolio_value=current_value,
                cash_balance=self.current_capital,
                total_pnl=total_pnl,
                unrealized_pnl=unrealized_pnl,
                realized_pnl=realized_pnl,
                total_fees=total_fees,
                net_pnl=net_pnl,
                max_drawdown_duration=max_dd_duration,
                avg_trade_duration=avg_trade_duration
            )
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return None
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio"""
        try:
            if len(self.daily_returns) < 2:
                return 0.0
            
            returns_array = np.array(self.daily_returns)
            excess_returns = returns_array - (self.risk_free_rate / 252)  # Daily risk-free rate
            
            if np.std(excess_returns) == 0:
                return 0.0
            
            sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
            return sharpe
            
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0
    
    def _calculate_max_drawdown(self) -> tuple:
        """Calculate maximum drawdown and duration"""
        try:
            if len(self.portfolio_values) < 2:
                return 0.0, 0
            
            values = np.array(self.portfolio_values)
            peak = np.maximum.accumulate(values)
            drawdown = (values - peak) / peak
            
            max_dd = np.min(drawdown)
            max_dd_duration = 0
            current_dd_duration = 0
            
            for value in values:
                if value < peak:
                    current_dd_duration += 1
                    max_dd_duration = max(max_dd_duration, current_dd_duration)
                else:
                    current_dd_duration = 0
            
            return abs(max_dd), max_dd_duration
            
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {e}")
            return 0.0, 0
    
    def _calculate_trade_statistics(self) -> Dict:
        """Calculate trade-related statistics"""
        try:
            completed_trades = [t for t in self.trades if t.exit_time is not None]
            
            if not completed_trades:
                return {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'win_rate': 0.0,
                    'profit_factor': 0.0,
                    'average_win': 0.0,
                    'average_loss': 0.0,
                    'largest_win': 0.0,
                    'largest_loss': 0.0,
                    'consecutive_wins': 0,
                    'consecutive_losses': 0
                }
            
            # Basic counts
            total_trades = len(completed_trades)
            winning_trades = len([t for t in completed_trades if t.pnl > 0])
            losing_trades = len([t for t in completed_trades if t.pnl < 0])
            
            # Win rate
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # Profit factor
            gross_profit = sum(t.pnl for t in completed_trades if t.pnl > 0)
            gross_loss = abs(sum(t.pnl for t in completed_trades if t.pnl < 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Average win/loss
            wins = [t.pnl for t in completed_trades if t.pnl > 0]
            losses = [t.pnl for t in completed_trades if t.pnl < 0]
            
            average_win = np.mean(wins) if wins else 0.0
            average_loss = np.mean(losses) if losses else 0.0
            largest_win = max(wins) if wins else 0.0
            largest_loss = min(losses) if losses else 0.0
            
            # Consecutive wins/losses
            consecutive_wins, consecutive_losses = self._calculate_consecutive_trades(completed_trades)
            
            return {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'average_win': average_win,
                'average_loss': average_loss,
                'largest_win': largest_win,
                'largest_loss': largest_loss,
                'consecutive_wins': consecutive_wins,
                'consecutive_losses': consecutive_losses
            }
            
        except Exception as e:
            logger.error(f"Error calculating trade statistics: {e}")
            return {}
    
    def _calculate_consecutive_trades(self, trades: List[TradeMetrics]) -> tuple:
        """Calculate maximum consecutive wins and losses"""
        try:
            if not trades:
                return 0, 0
            
            max_consecutive_wins = 0
            max_consecutive_losses = 0
            current_consecutive_wins = 0
            current_consecutive_losses = 0
            
            for trade in sorted(trades, key=lambda x: x.exit_time):
                if trade.pnl > 0:
                    current_consecutive_wins += 1
                    current_consecutive_losses = 0
                    max_consecutive_wins = max(max_consecutive_wins, current_consecutive_wins)
                elif trade.pnl < 0:
                    current_consecutive_losses += 1
                    current_consecutive_wins = 0
                    max_consecutive_losses = max(max_consecutive_losses, current_consecutive_losses)
                else:
                    current_consecutive_wins = 0
                    current_consecutive_losses = 0
            
            return max_consecutive_wins, max_consecutive_losses
            
        except Exception as e:
            logger.error(f"Error calculating consecutive trades: {e}")
            return 0, 0
    
    def add_trade(self, trade_data: Dict):
        """Add a new trade to tracking"""
        try:
            trade = TradeMetrics(
                trade_id=trade_data.get('trade_id', f"trade_{len(self.trades)}"),
                symbol=trade_data['symbol'],
                entry_time=trade_data.get('entry_time', datetime.now()),
                exit_time=trade_data.get('exit_time'),
                entry_price=trade_data['entry_price'],
                exit_price=trade_data.get('exit_price'),
                quantity=trade_data['quantity'],
                side=trade_data['side'],
                pnl=trade_data.get('pnl', 0.0),
                pnl_percent=trade_data.get('pnl_percent', 0.0),
                duration=trade_data.get('duration'),
                strategy=trade_data.get('strategy', 'unknown'),
                confidence=trade_data.get('confidence', 0.0),
                fees=trade_data.get('fees', 0.0)
            )
            
            self.trades.append(trade)
            logger.info(f"Trade added: {trade.symbol} {trade.side} PnL: {trade.pnl:.2f}")
            
        except Exception as e:
            logger.error(f"Error adding trade: {e}")
    
    def update_portfolio_value(self, value: float):
        """Update portfolio value"""
        try:
            self.portfolio_values.append(value)
            
            # Calculate daily return
            if len(self.portfolio_values) >= 2:
                daily_return = (value - self.portfolio_values[-2]) / self.portfolio_values[-2]
                self.daily_returns.append(daily_return)
            
            # Keep only recent values to manage memory
            if len(self.portfolio_values) > 10000:
                self.portfolio_values = self.portfolio_values[-5000:]
                self.daily_returns = self.daily_returns[-5000:]
            
        except Exception as e:
            logger.error(f"Error updating portfolio value: {e}")
    
    def get_metrics(self) -> Dict:
        """Get current performance metrics"""
        if self.current_metrics:
            return asdict(self.current_metrics)
        return {}
    
    def get_trade_history(self, limit: int = 100) -> List[Dict]:
        """Get trade history"""
        try:
            recent_trades = self.trades[-limit:] if limit else self.trades
            return [asdict(trade) for trade in recent_trades]
        except Exception as e:
            logger.error(f"Error getting trade history: {e}")
            return []
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary"""
        try:
            if not self.current_metrics:
                return {}
            
            return {
                'total_return': f"{self.current_metrics.total_return:.2%}",
                'sharpe_ratio': f"{self.current_metrics.sharpe_ratio:.2f}",
                'max_drawdown': f"{self.current_metrics.max_drawdown:.2%}",
                'win_rate': f"{self.current_metrics.win_rate:.2%}",
                'profit_factor': f"{self.current_metrics.profit_factor:.2f}",
                'total_trades': self.current_metrics.total_trades,
                'portfolio_value': f"${self.current_metrics.portfolio_value:,.2f}",
                'total_pnl': f"${self.current_metrics.total_pnl:,.2f}",
                'net_pnl': f"${self.current_metrics.net_pnl:,.2f}",
                'max_drawdown_duration': self.current_metrics.max_drawdown_duration,
                'avg_trade_duration': self.current_metrics.avg_trade_duration
            }
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {}
    
    def generate_report(self) -> str:
        """Generate detailed performance report"""
        try:
            if not self.current_metrics:
                return "No performance data available"
            
            report = f"""
TRADING PERFORMANCE REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

OVERALL PERFORMANCE
Total Return: {self.current_metrics.total_return:.2%}
Portfolio Value: ${self.current_metrics.portfolio_value:,.2f}
Total P&L: ${self.current_metrics.total_pnl:,.2f}
Realized P&L: ${self.current_metrics.realized_pnl:,.2f}
Unrealized P&L: ${self.current_metrics.unrealized_pnl:,.2f}
Net P&L: ${self.current_metrics.net_pnl:,.2f}

RISK METRICS
Sharpe Ratio: {self.current_metrics.sharpe_ratio:.2f}
Maximum Drawdown: {self.current_metrics.max_drawdown:.2%}
Max Drawdown Duration: {self.current_metrics.max_drawdown_duration} days

TRADE STATISTICS
Total Trades: {self.current_metrics.total_trades}
Winning Trades: {self.current_metrics.winning_trades}
Losing Trades: {self.current_metrics.losing_trades}
Win Rate: {self.current_metrics.win_rate:.2%}
Profit Factor: {self.current_metrics.profit_factor:.2f}
Average Trade Duration: {self.current_metrics.avg_trade_duration:.2f} hours

TRADE ANALYSIS
Average Win: ${self.current_metrics.average_win:.2f}
Average Loss: ${self.current_metrics.average_loss:.2f}
Largest Win: ${self.current_metrics.largest_win:.2f}
Largest Loss: ${self.current_metrics.largest_loss:.2f}
Max Consecutive Wins: {self.current_metrics.consecutive_wins}
Max Consecutive Losses: {self.current_metrics.consecutive_losses}
"""
            return report
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return f"Error generating report: {e}"

    def _save_metrics(self, metrics: PerformanceMetrics):
        """Save metrics to file"""
        try:
            filename = os.path.join(self.data_dir, f"metrics_{datetime.now().strftime('%Y%m%d')}.json")
            
            # Load existing data
            data = []
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    data = json.load(f)
            
            # Add new metrics
            metrics_dict = asdict(metrics)
            metrics_dict['timestamp'] = metrics.timestamp.isoformat()
            data.append(metrics_dict)
            
            # Save updated data
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
    
    def export_data(self, format: str = 'csv') -> str:
        """Export performance data"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            if format.lower() == 'csv':
                # Export trades
                if self.trades:
                    trades_df = pd.DataFrame([asdict(trade) for trade in self.trades])
                    trades_file = os.path.join(self.data_dir, f"trades_{timestamp}.csv")
                    trades_df.to_csv(trades_file, index=False)
                
                # Export metrics
                if self.metrics_history:
                    metrics_df = pd.DataFrame([asdict(m) for m in self.metrics_history])
                    metrics_file = os.path.join(self.data_dir, f"metrics_{timestamp}.csv")
                    metrics_df.to_csv(metrics_file, index=False)
                
                return f"Data exported to {self.data_dir}"
            
            return "Unsupported format"
            
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            return f"Export failed: {e}"

    def record_trade(self, trade_data: Dict):
        """Record a completed trade"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO trades (
                    timestamp, symbol, side, quantity, entry_price, exit_price,
                    pnl, fees, duration_hours, strategy, broker, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade_data.get('timestamp', datetime.now()),
                trade_data.get('symbol'),
                trade_data.get('side'),
                trade_data.get('quantity', 0),
                trade_data.get('entry_price', 0),
                trade_data.get('exit_price', 0),
                trade_data.get('pnl', 0),
                trade_data.get('fees', 0),
                trade_data.get('duration_hours', 0),
                trade_data.get('strategy', 'Unknown'),
                trade_data.get('broker', 'Unknown'),
                json.dumps(trade_data.get('metadata', {}))
            ))
            
            conn.commit()
            conn.close()
            
            # Update trade history
            self.trade_history.append(trade_data)
            
            logger.info(f"Trade recorded: {trade_data.get('symbol')} {trade_data.get('side')} PnL: {trade_data.get('pnl', 0):.2f}")
            
        except Exception as e:
            logger.error(f"Error recording trade: {e}")
    
    def record_daily_performance(self, performance_data: Dict):
        """Record daily performance metrics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            today = datetime.now().date()
            
            cursor.execute('''
                INSERT OR REPLACE INTO daily_performance (
                    date, total_pnl, realized_pnl, unrealized_pnl, total_trades,
                    winning_trades, losing_trades, total_volume, total_fees,
                    equity_value, max_drawdown, volatility
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                today,
                performance_data.get('total_pnl', 0),
                performance_data.get('realized_pnl', 0),
                performance_data.get('unrealized_pnl', 0),
                performance_data.get('total_trades', 0),
                performance_data.get('winning_trades', 0),
                performance_data.get('losing_trades', 0),
                performance_data.get('total_volume', 0),
                performance_data.get('total_fees', 0),
                performance_data.get('equity_value', 0),
                performance_data.get('max_drawdown', 0),
                performance_data.get('volatility', 0)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error recording daily performance: {e}")
    
    def update_positions(self, positions: List[Dict]):
        """Update current positions"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Clear existing positions
            cursor.execute('DELETE FROM positions')
            
            # Insert current positions
            for position in positions:
                cursor.execute('''
                    INSERT INTO positions (
                        timestamp, symbol, quantity, avg_price, current_price,
                        unrealized_pnl, broker
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    datetime.now(),
                    position.get('symbol'),
                    position.get('quantity', 0),
                    position.get('avg_price', 0),
                    position.get('current_price', 0),
                    position.get('unrealized_pnl', 0),
                    position.get('broker', 'Unknown')
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error updating positions: {e}")
    
    def calculate_performance_metrics(self, days: int = 30) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get trades from last N days
            start_date = datetime.now() - timedelta(days=days)
            trades_df = pd.read_sql_query('''
                SELECT * FROM trades 
                WHERE timestamp >= ? 
                ORDER BY timestamp
            ''', conn, params=[start_date])
            
            if trades_df.empty:
                return PerformanceMetrics(
                    timestamp=datetime.now(),
                    total_return=0, daily_return=0, sharpe_ratio=0,
                    max_drawdown=0, win_rate=0, profit_factor=0,
                    total_trades=0, winning_trades=0, losing_trades=0,
                    average_win=0, average_loss=0, largest_win=0,
                    largest_loss=0, consecutive_wins=0, consecutive_losses=0,
                    portfolio_value=self.initial_capital, cash_balance=self.initial_capital,
                    total_pnl=0, unrealized_pnl=0, realized_pnl=0,
                    total_fees=0, net_pnl=0,
                    max_drawdown_duration=0, avg_trade_duration=0
                )
            
            # Basic metrics
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['pnl'] > 0])
            losing_trades = len(trades_df[trades_df['pnl'] < 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # PnL metrics
            total_pnl = trades_df['pnl'].sum()
            total_fees = trades_df['fees'].sum()
            net_pnl = total_pnl - total_fees
            
            # Win/Loss metrics
            winning_trades_df = trades_df[trades_df['pnl'] > 0]
            losing_trades_df = trades_df[trades_df['pnl'] < 0]
            
            avg_win = winning_trades_df['pnl'].mean() if not winning_trades_df.empty else 0
            avg_loss = abs(losing_trades_df['pnl'].mean()) if not losing_trades_df.empty else 0
            
            # Profit factor
            total_wins = winning_trades_df['pnl'].sum() if not winning_trades_df.empty else 0
            total_losses = abs(losing_trades_df['pnl'].sum()) if not losing_trades_df.empty else 0
            profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
            
            # Sharpe ratio
            daily_returns = self._calculate_daily_returns(trades_df)
            sharpe_ratio = self._calculate_sharpe_ratio(daily_returns)
            
            # Drawdown metrics
            max_drawdown, max_dd_duration = self._calculate_max_drawdown(trades_df)
            
            # Average trade duration
            avg_trade_duration = trades_df['duration_hours'].mean() if 'duration_hours' in trades_df.columns else 0
            
            conn.close()
            
            return PerformanceMetrics(
                timestamp=datetime.now(),
                total_return=0, daily_return=0, sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown, win_rate=win_rate, profit_factor=profit_factor,
                total_trades=total_trades, winning_trades=winning_trades, losing_trades=losing_trades,
                average_win=avg_win, average_loss=avg_loss, largest_win=0,
                largest_loss=0, consecutive_wins=0, consecutive_losses=0,
                portfolio_value=self.initial_capital, cash_balance=self.initial_capital,
                total_pnl=total_pnl, unrealized_pnl=0, realized_pnl=0,
                total_fees=total_fees, net_pnl=net_pnl,
                max_drawdown_duration=max_dd_duration, avg_trade_duration=avg_trade_duration
            )
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return PerformanceMetrics(
                timestamp=datetime.now(),
                total_return=0, daily_return=0, sharpe_ratio=0,
                max_drawdown=0, win_rate=0, profit_factor=0,
                total_trades=0, winning_trades=0, losing_trades=0,
                average_win=0, average_loss=0, largest_win=0,
                largest_loss=0, consecutive_wins=0, consecutive_losses=0,
                portfolio_value=self.initial_capital, cash_balance=self.initial_capital,
                total_pnl=0, unrealized_pnl=0, realized_pnl=0,
                total_fees=0, net_pnl=0,
                max_drawdown_duration=0, avg_trade_duration=0
            )
    
    def _calculate_daily_returns(self, trades_df: pd.DataFrame) -> pd.Series:
        """Calculate daily returns from trades"""
        try:
            trades_df['date'] = pd.to_datetime(trades_df['timestamp']).dt.date
            daily_pnl = trades_df.groupby('date')['pnl'].sum()
            
            # Convert to returns (assuming starting capital)
            starting_capital = self.initial_capital  # Default starting capital
            daily_returns = daily_pnl / starting_capital
            
            return daily_returns
            
        except Exception as e:
            logger.error(f"Error calculating daily returns: {e}")
            return pd.Series()
    
    def _calculate_sharpe_ratio(self, daily_returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        try:
            if daily_returns.empty or daily_returns.std() == 0:
                return 0.0
            
            # Annualized metrics
            annual_return = daily_returns.mean() * 252
            annual_volatility = daily_returns.std() * np.sqrt(252)
            
            # Sharpe ratio
            sharpe = (annual_return - risk_free_rate) / annual_volatility
            
            return sharpe
            
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0
    
    def _calculate_max_drawdown(self, trades_df: pd.DataFrame) -> tuple:
        """Calculate maximum drawdown and duration"""
        try:
            # Calculate cumulative PnL
            trades_df = trades_df.sort_values('timestamp')
            trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
            
            # Calculate running maximum
            trades_df['running_max'] = trades_df['cumulative_pnl'].expanding().max()
            
            # Calculate drawdown
            trades_df['drawdown'] = trades_df['cumulative_pnl'] - trades_df['running_max']
            
            # Maximum drawdown
            max_drawdown = trades_df['drawdown'].min()
            
            # Calculate drawdown duration
            max_dd_duration = 0
            current_dd_duration = 0
            
            for _, row in trades_df.iterrows():
                if row['drawdown'] < 0:
                    current_dd_duration += 1
                    max_dd_duration = max(max_dd_duration, current_dd_duration)
                else:
                    current_dd_duration = 0
            
            return abs(max_drawdown), max_dd_duration
            
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {e}")
            return 0.0, 0
    
    def get_equity_curve(self, days: int = 30) -> List[Dict]:
        """Get equity curve data"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            start_date = datetime.now() - timedelta(days=days)
            trades_df = pd.read_sql_query('''
                SELECT timestamp, pnl FROM trades 
                WHERE timestamp >= ? 
                ORDER BY timestamp
            ''', conn, params=[start_date])
            
            if trades_df.empty:
                return []
            
            # Calculate cumulative PnL
            trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
            
            # Convert to list of dictionaries
            equity_curve = []
            for _, row in trades_df.iterrows():
                equity_curve.append({
                    'timestamp': row['timestamp'],
                    'cumulative_pnl': row['cumulative_pnl'],
                    'trade_pnl': row['pnl']
                })
            
            conn.close()
            return equity_curve
            
        except Exception as e:
            logger.error(f"Error getting equity curve: {e}")
            return []
    
    def get_trade_distribution(self, days: int = 30) -> Dict[str, Any]:
        """Get trade distribution analysis"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            start_date = datetime.now() - timedelta(days=days)
            trades_df = pd.read_sql_query('''
                SELECT pnl, symbol, strategy FROM trades 
                WHERE timestamp >= ?
            ''', conn, params=[start_date])
            
            if trades_df.empty:
                return {}
            
            # PnL distribution
            pnl_bins = [-float('inf'), -1000, -500, -100, -50, 0, 50, 100, 500, 1000, float('inf')]
            pnl_labels = ['< -1000', '-1000 to -500', '-500 to -100', '-100 to -50', 
                         '-50 to 0', '0 to 50', '50 to 100', '100 to 500', '500 to 1000', '> 1000']
            
            trades_df['pnl_bucket'] = pd.cut(trades_df['pnl'], bins=pnl_bins, labels=pnl_labels)
            pnl_distribution = trades_df['pnl_bucket'].value_counts().to_dict()
            
            # Symbol performance
            symbol_performance = trades_df.groupby('symbol')['pnl'].agg(['count', 'sum', 'mean']).to_dict('index')
            
            # Strategy performance
            strategy_performance = trades_df.groupby('strategy')['pnl'].agg(['count', 'sum', 'mean']).to_dict('index')
            
            conn.close()
            
            return {
                'pnl_distribution': pnl_distribution,
                'symbol_performance': symbol_performance,
                'strategy_performance': strategy_performance
            }
            
        except Exception as e:
            logger.error(f"Error getting trade distribution: {e}")
            return {}
    
    def get_monthly_performance(self) -> List[Dict]:
        """Get monthly performance summary"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            monthly_df = pd.read_sql_query('''
                SELECT 
                    strftime('%Y-%m', timestamp) as month,
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                    SUM(pnl) as total_pnl,
                    SUM(fees) as total_fees,
                    AVG(pnl) as avg_pnl
                FROM trades 
                GROUP BY strftime('%Y-%m', timestamp)
                ORDER BY month DESC
            ''', conn)
            
            conn.close()
            
            return monthly_df.to_dict('records')
            
        except Exception as e:
            logger.error(f"Error getting monthly performance: {e}")
            return []
