#!/usr/bin/env python3
"""
Advanced Risk Manager with Real-time Monitoring and Dynamic Adjustments
"""

import numpy as np
import pandas as pd
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from scipy import stats
import json

logger = logging.getLogger(__name__)

@dataclass
class RiskMetrics:
    """Comprehensive risk metrics"""
    portfolio_var: float
    expected_shortfall: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    beta: float
    correlation_risk: float
    concentration_risk: float
    liquidity_risk: float
    leverage_ratio: float

@dataclass
class RiskAlert:
    """Risk alert information"""
    alert_type: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    message: str
    recommended_action: str
    affected_positions: List[str]
    timestamp: datetime

class AdvancedRiskManager:
    """Advanced risk management with real-time monitoring"""
    
    def __init__(self, max_daily_loss=2000, max_drawdown=0.10, max_position_size=0.05, portfolio_value=100000):
        self.max_daily_loss = max_daily_loss
        self.max_drawdown = max_drawdown
        self.max_position_size = max_position_size
        self.portfolio_value = portfolio_value
        
        # Risk tracking
        self.daily_pnl_history = []
        self.position_history = []
        self.risk_alerts = []
        self.current_positions = {}
        
        # Risk limits
        self.risk_limits = {
            'max_portfolio_var': max_daily_loss,
            'max_leverage': 2.0,
            'max_correlation': 0.8,
            'min_liquidity_score': 0.3,
            'max_sector_concentration': 0.3,
            'max_single_position': max_position_size,
            'max_daily_trades': 50,
            'max_position_count': 20
        }
        
        # Dynamic adjustment factors
        self.adjustment_factors = {
            'volatility_multiplier': 1.0,
            'correlation_multiplier': 1.0,
            'liquidity_multiplier': 1.0,
            'performance_multiplier': 1.0
        }
        
        # Market data cache
        self.market_data_cache = {}
        self.correlation_matrix = None
        
        logger.info("Advanced Risk Manager initialized")
    
    async def validate_signal(self, signal: Any) -> bool:
        """Validate if signal meets risk criteria"""
        try:
            # Basic position size check
            if signal.position_size > self.max_position_size:
                logger.warning(f"Signal rejected: position size {signal.position_size:.2%} exceeds limit {self.max_position_size:.2%}")
                return False
            
            # Portfolio heat check
            current_heat = await self._calculate_portfolio_heat()
            if current_heat + signal.position_size > 0.5:  # 50% max portfolio heat
                logger.warning(f"Signal rejected: would exceed portfolio heat limit")
                return False
            
            # Concentration check
            if await self._would_exceed_concentration(signal):
                logger.warning(f"Signal rejected: would exceed concentration limits")
                return False
            
            # Correlation check
            if await self._check_correlation_risk(signal):
                logger.warning(f"Signal rejected: high correlation risk")
                return False
            
            # Liquidity check
            liquidity_score = await self._get_liquidity_score(signal.symbol)
            if liquidity_score < self.risk_limits['min_liquidity_score']:
                logger.warning(f"Signal rejected: low liquidity score {liquidity_score:.2f}")
                return False
            
            # Market conditions check
            if await self._check_market_stress():
                logger.warning(f"Signal rejected: market stress detected")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating signal: {e}")
            return False
    
    async def _calculate_portfolio_heat(self) -> float:
        """Calculate current portfolio heat (total position exposure)"""
        try:
            total_exposure = 0.0
            
            for symbol, position in self.current_positions.items():
                position_value = abs(position.get('market_value', 0))
                total_exposure += position_value
            
            return total_exposure / self.portfolio_value if self.portfolio_value > 0 else 0
            
        except Exception as e:
            logger.error(f"Error calculating portfolio heat: {e}")
            return 0.0
    
    async def _would_exceed_concentration(self, signal: Any) -> bool:
        """Check if signal would exceed concentration limits"""
        try:
            symbol = signal.symbol
            new_position_value = signal.position_size * self.portfolio_value
            
            # Current position value for this symbol
            current_position = self.current_positions.get(symbol, {})
            current_value = abs(current_position.get('market_value', 0))
            
            # Total position value after signal
            total_symbol_value = current_value + new_position_value
            
            # Check single position limit
            single_position_ratio = total_symbol_value / self.portfolio_value
            if single_position_ratio > self.risk_limits['max_single_position']:
                return True
            
            # Check sector concentration
            sector = await self._get_symbol_sector(symbol)
            sector_exposure = await self._calculate_sector_exposure(sector)
            sector_ratio = (sector_exposure + new_position_value) / self.portfolio_value
            
            if sector_ratio > self.risk_limits['max_sector_concentration']:
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking concentration: {e}")
            return True  # Conservative approach
    
    async def _check_correlation_risk(self, signal: Any) -> bool:
        """Check correlation risk with existing positions"""
        try:
            if not self.current_positions:
                return False
            
            symbol = signal.symbol
            
            # Get correlation with existing positions
            high_correlation_count = 0
            
            for existing_symbol in self.current_positions.keys():
                correlation = await self._get_correlation(symbol, existing_symbol)
                
                if abs(correlation) > self.risk_limits['max_correlation']:
                    high_correlation_count += 1
            
            # If more than 30% of positions are highly correlated, reject
            correlation_ratio = high_correlation_count / len(self.current_positions)
            return correlation_ratio > 0.3
            
        except Exception as e:
            logger.error(f"Error checking correlation risk: {e}")
            return False
    
    async def _get_correlation(self, symbol1: str, symbol2: str) -> float:
        """Get correlation between two symbols"""
        try:
            if symbol1 == symbol2:
                return 1.0
            
            # Use cached correlation matrix if available
            if self.correlation_matrix is not None:
                if symbol1 in self.correlation_matrix.columns and symbol2 in self.correlation_matrix.columns:
                    return self.correlation_matrix.loc[symbol1, symbol2]
            
            # Default correlations based on asset classes
            crypto_symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT']
            tech_symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META']
            
            if symbol1 in crypto_symbols and symbol2 in crypto_symbols:
                return 0.7  # High correlation between cryptos
            elif symbol1 in tech_symbols and symbol2 in tech_symbols:
                return 0.6  # Moderate correlation between tech stocks
            else:
                return 0.3  # Low default correlation
                
        except Exception as e:
            logger.error(f"Error getting correlation: {e}")
            return 0.5
    
    async def _get_liquidity_score(self, symbol: str) -> float:
        """Get liquidity score for symbol"""
        try:
            # High liquidity symbols
            high_liquidity = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'BTCUSDT', 'ETHUSDT']
            if symbol in high_liquidity:
                return 0.9
            
            # Medium liquidity
            medium_liquidity = ['NVDA', 'META', 'NFLX', 'AMD', 'CRM']
            if symbol in medium_liquidity:
                return 0.7
            
            # Lower liquidity for other symbols
            return 0.5
            
        except Exception as e:
            logger.error(f"Error getting liquidity score: {e}")
            return 0.5
    
    async def _check_market_stress(self) -> bool:
        """Check if market is in stressed conditions"""
        try:
            # Simple market stress indicators
            # In a real implementation, this would check VIX, market volatility, etc.
            
            # Check recent portfolio performance
            if len(self.daily_pnl_history) >= 5:
                recent_pnl = self.daily_pnl_history[-5:]
                negative_days = sum(1 for pnl in recent_pnl if pnl < 0)
                
                if negative_days >= 4:  # 4 out of 5 negative days
                    return True
            
            # Check current drawdown
            current_drawdown = await self._calculate_current_drawdown()
            if current_drawdown > self.max_drawdown * 0.8:  # 80% of max drawdown
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking market stress: {e}")
            return False
    
    async def _get_symbol_sector(self, symbol: str) -> str:
        """Get sector for symbol"""
        try:
            # Sector mapping
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
    
    async def _calculate_sector_exposure(self, sector: str) -> float:
        """Calculate current exposure to a sector"""
        try:
            sector_exposure = 0.0
            
            for symbol, position in self.current_positions.items():
                symbol_sector = await self._get_symbol_sector(symbol)
                if symbol_sector == sector:
                    sector_exposure += abs(position.get('market_value', 0))
            
            return sector_exposure
            
        except Exception as e:
            logger.error(f"Error calculating sector exposure: {e}")
            return 0.0
    
    async def check_limits(self) -> List[RiskAlert]:
        """Check all risk limits and generate alerts"""
        alerts = []
        
        try:
            # Check daily loss limit
            daily_pnl = await self._get_daily_pnl()
            if daily_pnl < -self.max_daily_loss:
                alerts.append(RiskAlert(
                    alert_type="DAILY_LOSS_LIMIT",
                    severity="CRITICAL",
                    message=f"Daily loss limit exceeded: ${daily_pnl:.2f}",
                    recommended_action="STOP_TRADING",
                    affected_positions=list(self.current_positions.keys()),
                    timestamp=datetime.now()
                ))
            
            # Check drawdown limit
            current_drawdown = await self._calculate_current_drawdown()
            if current_drawdown > self.max_drawdown:
                alerts.append(RiskAlert(
                    alert_type="DRAWDOWN_LIMIT",
                    severity="CRITICAL",
                    message=f"Maximum drawdown exceeded: {current_drawdown:.2%}",
                    recommended_action="REDUCE_POSITIONS",
                    affected_positions=list(self.current_positions.keys()),
                    timestamp=datetime.now()
                ))
            
            # Check portfolio heat
            portfolio_heat = await self._calculate_portfolio_heat()
            if portfolio_heat > 0.8:  # 80% portfolio heat
                alerts.append(RiskAlert(
                    alert_type="HIGH_PORTFOLIO_HEAT",
                    severity="HIGH",
                    message=f"High portfolio heat: {portfolio_heat:.2%}",
                    recommended_action="REDUCE_POSITION_SIZES",
                    affected_positions=list(self.current_positions.keys()),
                    timestamp=datetime.now()
                ))
            
            # Check concentration risk
            concentration_alerts = await self._check_concentration_risk()
            alerts.extend(concentration_alerts)
            
            # Check correlation risk
            correlation_alerts = await self._check_portfolio_correlation_risk()
            alerts.extend(correlation_alerts)
            
            # Store alerts
            self.risk_alerts.extend(alerts)
            
            # Keep only recent alerts
            cutoff_time = datetime.now() - timedelta(hours=24)
            self.risk_alerts = [alert for alert in self.risk_alerts if alert.timestamp > cutoff_time]
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error checking risk limits: {e}")
            return []
    
    async def _get_daily_pnl(self) -> float:
        """Get current daily P&L"""
        try:
            if not self.daily_pnl_history:
                return 0.0
            
            return self.daily_pnl_history[-1] if self.daily_pnl_history else 0.0
            
        except Exception as e:
            logger.error(f"Error getting daily PnL: {e}")
            return 0.0
    
    async def _calculate_current_drawdown(self) -> float:
        """Calculate current drawdown"""
        try:
            if len(self.daily_pnl_history) < 2:
                return 0.0
            
            # Calculate cumulative returns
            cumulative_pnl = np.cumsum(self.daily_pnl_history)
            
            # Calculate running maximum
            running_max = np.maximum.accumulate(cumulative_pnl)
            
            # Calculate drawdown
            drawdown = (cumulative_pnl - running_max) / (self.portfolio_value + running_max)
            
            return abs(drawdown[-1]) if len(drawdown) > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating drawdown: {e}")
            return 0.0
    
    async def _check_concentration_risk(self) -> List[RiskAlert]:
        """Check for concentration risk"""
        alerts = []
        
        try:
            # Check single position concentration
            for symbol, position in self.current_positions.items():
                position_value = abs(position.get('market_value', 0))
                concentration = position_value / self.portfolio_value
                
                if concentration > self.risk_limits['max_single_position']:
                    alerts.append(RiskAlert(
                        alert_type="POSITION_CONCENTRATION",
                        severity="HIGH",
                        message=f"High concentration in {symbol}: {concentration:.2%}",
                        recommended_action="REDUCE_POSITION",
                        affected_positions=[symbol],
                        timestamp=datetime.now()
                    ))
            
            # Check sector concentration
            sector_exposures = {}
            for symbol, position in self.current_positions.items():
                sector = await self._get_symbol_sector(symbol)
                position_value = abs(position.get('market_value', 0))
                
                if sector not in sector_exposures:
                    sector_exposures[sector] = 0
                sector_exposures[sector] += position_value
            
            for sector, exposure in sector_exposures.items():
                concentration = exposure / self.portfolio_value
                if concentration > self.risk_limits['max_sector_concentration']:
                    affected_symbols = [
                        symbol for symbol in self.current_positions.keys()
                        if await self._get_symbol_sector(symbol) == sector
                    ]
                    
                    alerts.append(RiskAlert(
                        alert_type="SECTOR_CONCENTRATION",
                        severity="MEDIUM",
                        message=f"High {sector} sector concentration: {concentration:.2%}",
                        recommended_action="DIVERSIFY_SECTORS",
                        affected_positions=affected_symbols,
                        timestamp=datetime.now()
                    ))
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error checking concentration risk: {e}")
            return []
    
    async def _check_portfolio_correlation_risk(self) -> List[RiskAlert]:
        """Check portfolio correlation risk"""
        alerts = []
        
        try:
            if len(self.current_positions) < 2:
                return alerts
            
            symbols = list(self.current_positions.keys())
            high_correlation_pairs = []
            
            for i, symbol1 in enumerate(symbols):
                for symbol2 in symbols[i+1:]:
                    correlation = await self._get_correlation(symbol1, symbol2)
                    
                    if abs(correlation) > self.risk_limits['max_correlation']:
                        high_correlation_pairs.append((symbol1, symbol2, correlation))
            
            if len(high_correlation_pairs) > len(symbols) * 0.3:  # More than 30% of pairs highly correlated
                affected_symbols = list(set([symbol for pair in high_correlation_pairs for symbol in pair[:2]]))
                
                alerts.append(RiskAlert(
                    alert_type="HIGH_CORRELATION",
                    severity="MEDIUM",
                    message=f"High correlation detected in {len(high_correlation_pairs)} position pairs",
                    recommended_action="REDUCE_CORRELATED_POSITIONS",
                    affected_positions=affected_symbols,
                    timestamp=datetime.now()
                ))
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error checking correlation risk: {e}")
            return []
    
    async def calculate_risk_metrics(self) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        try:
            # Portfolio VaR (95% confidence)
            portfolio_var = await self._calculate_portfolio_var()
            
            # Expected Shortfall
            expected_shortfall = await self._calculate_expected_shortfall()
            
            # Maximum Drawdown
            max_drawdown = await self._calculate_max_drawdown()
            
            # Sharpe Ratio
            sharpe_ratio = await self._calculate_sharpe_ratio()
            
            # Sortino Ratio
            sortino_ratio = await self._calculate_sortino_ratio()
            
            # Beta
            beta = await self._calculate_portfolio_beta()
            
            # Risk scores
            correlation_risk = await self._calculate_correlation_risk_score()
            concentration_risk = await self._calculate_concentration_risk_score()
            liquidity_risk = await self._calculate_liquidity_risk_score()
            leverage_ratio = await self._calculate_leverage_ratio()
            
            return RiskMetrics(
                portfolio_var=portfolio_var,
                expected_shortfall=expected_shortfall,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                beta=beta,
                correlation_risk=correlation_risk,
                concentration_risk=concentration_risk,
                liquidity_risk=liquidity_risk,
                leverage_ratio=leverage_ratio
            )
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return RiskMetrics(0, 0, 0, 0, 0, 1, 0, 0, 0, 1)
    
    async def _calculate_portfolio_var(self, confidence_level: float = 0.95) -> float:
        """Calculate portfolio Value at Risk"""
        try:
            if len(self.daily_pnl_history) < 30:
                return 0.0
            
            # Use historical method
            returns = np.array(self.daily_pnl_history[-252:])  # Last year
            var = np.percentile(returns, (1 - confidence_level) * 100)
            
            return abs(var)
            
        except Exception as e:
            logger.error(f"Error calculating portfolio VaR: {e}")
            return 0.0
    
    async def _calculate_expected_shortfall(self, confidence_level: float = 0.95) -> float:
        """Calculate Expected Shortfall (Conditional VaR)"""
        try:
            if len(self.daily_pnl_history) < 30:
                return 0.0
            
            returns = np.array(self.daily_pnl_history[-252:])
            var = np.percentile(returns, (1 - confidence_level) * 100)
            
            # Expected shortfall is the mean of returns below VaR
            tail_returns = returns[returns <= var]
            
            if len(tail_returns) == 0:
                return abs(var)
            
            return abs(np.mean(tail_returns))
            
        except Exception as e:
            logger.error(f"Error calculating expected shortfall: {e}")
            return 0.0
    
    async def _calculate_max_drawdown(self) -> float:
        """Calculate maximum historical drawdown"""
        try:
            if len(self.daily_pnl_history) < 2:
                return 0.0
            
            cumulative_pnl = np.cumsum(self.daily_pnl_history)
            running_max = np.maximum.accumulate(cumulative_pnl)
            drawdown = (cumulative_pnl - running_max) / (self.portfolio_value + running_max)
            
            return abs(np.min(drawdown))
            
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {e}")
            return 0.0
    
    async def _calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        try:
            if len(self.daily_pnl_history) < 30:
                return 0.0
            
            returns = np.array(self.daily_pnl_history) / self.portfolio_value
            excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
            
            if np.std(returns) == 0:
                return 0.0
            
            return np.mean(excess_returns) / np.std(returns) * np.sqrt(252)
            
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0
    
    async def _calculate_sortino_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio"""
        try:
            if len(self.daily_pnl_history) < 30:
                return 0.0
            
            returns = np.array(self.daily_pnl_history) / self.portfolio_value
            excess_returns = returns - (risk_free_rate / 252)
            
            # Downside deviation
            negative_returns = excess_returns[excess_returns < 0]
            if len(negative_returns) == 0:
                return float('inf')
            
            downside_deviation = np.sqrt(np.mean(negative_returns ** 2))
            
            if downside_deviation == 0:
                return 0.0
            
            return np.mean(excess_returns) / downside_deviation * np.sqrt(252)
            
        except Exception as e:
            logger.error(f"Error calculating Sortino ratio: {e}")
            return 0.0
    
    async def _calculate_portfolio_beta(self) -> float:
        """Calculate portfolio beta relative to market"""
        try:
            # Simplified beta calculation
            # In practice, this would use market index returns
            return 1.0  # Placeholder
            
        except Exception as e:
            logger.error(f"Error calculating portfolio beta: {e}")
            return 1.0
    
    async def _calculate_correlation_risk_score(self) -> float:
        """Calculate correlation risk score"""
        try:
            if len(self.current_positions) < 2:
                return 0.0
            
            symbols = list(self.current_positions.keys())
            correlations = []
            
            for i, symbol1 in enumerate(symbols):
                for symbol2 in symbols[i+1:]:
                    correlation = await self._get_correlation(symbol1, symbol2)
                    correlations.append(abs(correlation))
            
            if not correlations:
                return 0.0
            
            # Risk score based on average correlation
            avg_correlation = np.mean(correlations)
            return min(1.0, avg_correlation)
            
        except Exception as e:
            logger.error(f"Error calculating correlation risk score: {e}")
            return 0.0
    
    async def _calculate_concentration_risk_score(self) -> float:
        """Calculate concentration risk score"""
        try:
            if not self.current_positions:
                return 0.0
            
            # Calculate Herfindahl-Hirschman Index
            total_value = sum(abs(pos.get('market_value', 0)) for pos in self.current_positions.values())
            
            if total_value == 0:
                return 0.0
            
            hhi = sum((abs(pos.get('market_value', 0)) / total_value) ** 2 
                     for pos in self.current_positions.values())
            
            # Normalize HHI to 0-1 scale
            n = len(self.current_positions)
            min_hhi = 1 / n  # Perfectly diversified
            max_hhi = 1.0    # Perfectly concentrated
            
            if max_hhi == min_hhi:
                return 0.0
            
            normalized_hhi = (hhi - min_hhi) / (max_hhi - min_hhi)
            return max(0.0, min(1.0, normalized_hhi))
            
        except Exception as e:
            logger.error(f"Error calculating concentration risk score: {e}")
            return 0.0
    
    async def _calculate_liquidity_risk_score(self) -> float:
        """Calculate liquidity risk score"""
        try:
            if not self.current_positions:
                return 0.0
            
            liquidity_scores = []
            total_value = 0
            
            for symbol, position in self.current_positions.items():
                liquidity_score = await self._get_liquidity_score(symbol)
                position_value = abs(position.get('market_value', 0))
                
                liquidity_scores.append(liquidity_score * position_value)
                total_value += position_value
            
            if total_value == 0:
                return 0.0
            
            # Weighted average liquidity score
            weighted_liquidity = sum(liquidity_scores) / total_value
            
            # Risk score is inverse of liquidity
            return 1.0 - weighted_liquidity
            
        except Exception as e:
            logger.error(f"Error calculating liquidity risk score: {e}")
            return 0.0
    
    async def _calculate_leverage_ratio(self) -> float:
        """Calculate current leverage ratio"""
        try:
            total_position_value = sum(abs(pos.get('market_value', 0)) for pos in self.current_positions.values())
            
            if self.portfolio_value == 0:
                return 1.0
            
            return total_position_value / self.portfolio_value
            
        except Exception as e:
            logger.error(f"Error calculating leverage ratio: {e}")
            return 1.0
    
    async def tighten_limits(self):
        """Tighten risk limits due to poor performance"""
        try:
            # Reduce position size limits
            self.max_position_size *= 0.8
            self.risk_limits['max_single_position'] *= 0.8
            
            # Reduce daily loss limit
            self.max_daily_loss *= 0.9
            self.risk_limits['max_portfolio_var'] *= 0.9
            
            # Increase correlation threshold (more restrictive)
            self.risk_limits['max_correlation'] *= 0.9
            
            # Reduce sector concentration
            self.risk_limits['max_sector_concentration'] *= 0.9
            
            logger.info("Risk limits tightened due to performance concerns")
            
        except Exception as e:
            logger.error(f"Error tightening limits: {e}")
    
    def update_positions(self, positions: Dict[str, Any]):
        """Update current positions"""
        try:
            self.current_positions = positions
            
            # Update daily P&L
            current_pnl = sum(pos.get('unrealized_pnl', 0) + pos.get('realized_pnl', 0) 
                            for pos in positions.values())
            
            # Store daily P&L (assuming this is called once per day)
            today = datetime.now().date()
            if not self.daily_pnl_history or len(self.daily_pnl_history) == 0:
                self.daily_pnl_history.append(current_pnl)
            else:
                # Update today's P&L
                self.daily_pnl_history[-1] = current_pnl
            
            # Keep only last year of data
            if len(self.daily_pnl_history) > 252:
                self.daily_pnl_history = self.daily_pnl_history[-252:]
                
        except Exception as e:
            logger.error(f"Error updating positions: {e}")
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary"""
        try:
            return {
                'risk_limits': self.risk_limits,
                'current_positions_count': len(self.current_positions),
                'portfolio_heat': asyncio.run(self._calculate_portfolio_heat()),
                'daily_pnl': asyncio.run(self._get_daily_pnl()),
                'current_drawdown': asyncio.run(self._calculate_current_drawdown()),
                'recent_alerts_count': len([alert for alert in self.risk_alerts 
                                          if (datetime.now() - alert.timestamp).total_seconds() < 3600]),
                'adjustment_factors': self.adjustment_factors,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting risk summary: {e}")
            return {}
