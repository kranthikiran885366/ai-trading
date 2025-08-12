#!/usr/bin/env python3
"""
Advanced Risk Manager - Comprehensive risk management system
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from scipy import stats
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class RiskMetrics:
    """Risk metrics data structure"""
    var_95: float
    var_99: float
    expected_shortfall: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    beta: float
    correlation_risk: float
    concentration_risk: float
    liquidity_risk: float

@dataclass
class PositionRisk:
    """Position risk data structure"""
    symbol: str
    position_size: float
    market_value: float
    var_contribution: float
    beta: float
    correlation: float
    liquidity_score: float
    risk_score: float

class AdvancedRiskManager:
    """Advanced risk management system with multiple risk models"""
    
    def __init__(self, max_daily_loss: float = 5000, max_drawdown: float = 0.15, 
                 max_position_size: float = 0.05, portfolio_value: float = 100000):
        
        self.max_daily_loss = max_daily_loss
        self.max_drawdown = max_drawdown
        self.max_position_size = max_position_size
        self.portfolio_value = portfolio_value
        
        # Risk tracking
        self.daily_pnl = []
        self.positions = {}
        self.trade_history = []
        self.risk_metrics_history = []
        
        # Market data for risk calculations
        self.market_data = {}
        self.correlation_matrix = None
        self.volatility_estimates = {}
        
        # Risk limits
        self.risk_limits = {
            'max_var': max_daily_loss,
            'max_leverage': 2.0,
            'max_correlation': 0.8,
            'min_liquidity_score': 0.3,
            'max_sector_concentration': 0.3
        }
        
        logger.info("Advanced Risk Manager initialized")
    
    def update_market_data(self, symbols: List[str], period: str = '1y'):
        """Update market data for risk calculations"""
        logger.info("Updating market data for risk calculations...")
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period)
                
                if not data.empty:
                    # Calculate returns
                    data['returns'] = data['Close'].pct_change().dropna()
                    self.market_data[symbol] = data
                    
                    # Calculate volatility
                    self.volatility_estimates[symbol] = self._calculate_volatility_estimate(data['returns'])
                
            except Exception as e:
                logger.error(f"Error updating market data for {symbol}: {e}")
        
        # Update correlation matrix
        self._update_correlation_matrix()
    
    def _calculate_volatility_estimate(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate multiple volatility estimates"""
        volatility_estimates = {}
        
        # Historical volatility
        volatility_estimates['historical'] = returns.std() * np.sqrt(252)
        
        # EWMA volatility
        volatility_estimates['ewma'] = self._calculate_ewma_volatility(returns)
        
        # Realized volatility
        volatility_estimates['realized'] = self._calculate_realized_volatility(returns)
        
        return volatility_estimates
    
    def _calculate_ewma_volatility(self, returns: pd.Series, lambda_param: float = 0.94) -> float:
        """Calculate EWMA volatility"""
        if len(returns) < 2:
            return 0.0
        
        weights = np.array([(1 - lambda_param) * (lambda_param ** i) for i in range(len(returns))])
        weights = weights / weights.sum()
        
        weighted_variance = np.sum(weights * (returns.values ** 2))
        return np.sqrt(weighted_variance * 252)
    
    def _calculate_realized_volatility(self, returns: pd.Series) -> float:
        """Calculate realized volatility"""
        return returns.std() * np.sqrt(252)
    
    def _update_correlation_matrix(self):
        """Update correlation matrix for all symbols"""
        if len(self.market_data) < 2:
            return
        
        returns_data = {}
        for symbol, data in self.market_data.items():
            if 'returns' in data.columns and len(data['returns']) > 0:
                returns_data[symbol] = data['returns'].dropna()
        
        if len(returns_data) >= 2:
            # Align all return series
            returns_df = pd.DataFrame(returns_data).dropna()
            
            if not returns_df.empty:
                self.correlation_matrix = returns_df.corr()
    
    def _calculate_historical_var(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """Calculate Historical Value at Risk"""
        if len(returns) == 0:
            return 0.0
        
        return np.percentile(returns, (1 - confidence_level) * 100)
    
    def _calculate_parametric_var(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """Calculate Parametric Value at Risk"""
        if len(returns) == 0:
            return 0.0
        
        mean_return = returns.mean()
        std_return = returns.std()
        
        # Z-score for confidence level
        z_score = stats.norm.ppf(1 - confidence_level)
        
        return mean_return + z_score * std_return
    
    def _calculate_expected_shortfall(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """Calculate Expected Shortfall (Conditional VaR)"""
        if len(returns) == 0:
            return 0.0
        
        var = self._calculate_historical_var(returns, confidence_level)
        
        # Expected shortfall is the mean of returns below VaR
        tail_returns = returns[returns <= var]
        
        if len(tail_returns) == 0:
            return var
        
        return tail_returns.mean()
    
    def _calculate_maximum_drawdown(self, returns: pd.Series) -> float:
        """Calculate Maximum Drawdown"""
        if len(returns) == 0:
            return 0.0
        
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        
        return drawdown.min()
    
    def calculate_portfolio_var(self, positions: Dict[str, float], confidence_level: float = 0.95) -> float:
        """Calculate portfolio Value at Risk"""
        if not positions or self.correlation_matrix is None:
            return 0.0
        
        # Get position weights
        total_value = sum(abs(pos) for pos in positions.values())
        if total_value == 0:
            return 0.0
        
        weights = np.array([positions.get(symbol, 0) / total_value 
                          for symbol in self.correlation_matrix.columns])
        
        # Get volatilities
        volatilities = np.array([
            self.volatility_estimates.get(symbol, {}).get('ewma', 0.2)
            for symbol in self.correlation_matrix.columns
        ])
        
        # Calculate portfolio variance
        portfolio_variance = np.dot(weights, np.dot(self.correlation_matrix.values * 
                                                  np.outer(volatilities, volatilities), weights))
        
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Convert to VaR
        z_score = stats.norm.ppf(1 - confidence_level)
        portfolio_var = z_score * portfolio_volatility * np.sqrt(1/252)  # Daily VaR
        
        return portfolio_var * total_value
    
    def calculate_position_risk(self, symbol: str, position_size: float, 
                              current_price: float) -> PositionRisk:
        """Calculate risk metrics for a single position"""
        market_value = position_size * current_price
        
        # Get volatility
        volatility = self.volatility_estimates.get(symbol, {}).get('ewma', 0.2)
        
        # Calculate VaR contribution
        position_var = abs(market_value) * volatility * stats.norm.ppf(1 - 0.95) / np.sqrt(252)
        
        # Calculate beta
        beta = self._calculate_beta(symbol)
        
        # Calculate correlation with portfolio
        correlation = self._calculate_portfolio_correlation(symbol)
        
        # Calculate liquidity score
        liquidity_score = self._calculate_liquidity_score(symbol)
        
        # Overall risk score
        risk_score = self._calculate_risk_score(volatility, beta, correlation, liquidity_score)
        
        return PositionRisk(
            symbol=symbol,
            position_size=position_size,
            market_value=market_value,
            var_contribution=position_var,
            beta=beta,
            correlation=correlation,
            liquidity_score=liquidity_score,
            risk_score=risk_score
        )
    
    def _calculate_beta(self, symbol: str) -> float:
        """Calculate beta relative to market (SPY)"""
        if symbol not in self.market_data or 'SPY' not in self.market_data:
            return 1.0
        
        try:
            symbol_returns = self.market_data[symbol]['returns'].dropna()
            market_returns = self.market_data['SPY']['returns'].dropna()
            
            # Align returns
            aligned_data = pd.DataFrame({
                'symbol': symbol_returns,
                'market': market_returns
            }).dropna()
            
            if len(aligned_data) < 30:
                return 1.0
            
            # Calculate beta using linear regression
            covariance = aligned_data['symbol'].cov(aligned_data['market'])
            market_variance = aligned_data['market'].var()
            
            if market_variance == 0:
                return 1.0
            
            beta = covariance / market_variance
            return beta
            
        except Exception as e:
            logger.error(f"Error calculating beta for {symbol}: {e}")
            return 1.0
    
    def _calculate_portfolio_correlation(self, symbol: str) -> float:
        """Calculate correlation with current portfolio"""
        if not self.positions or symbol not in self.market_data:
            return 0.0
        
        try:
            # Calculate portfolio returns
            portfolio_returns = []
            symbol_returns = self.market_data[symbol]['returns'].dropna()
            
            for date in symbol_returns.index:
                portfolio_return = 0
                total_value = 0
                
                for pos_symbol, position in self.positions.items():
                    if pos_symbol in self.market_data and date in self.market_data[pos_symbol].index:
                        return_val = self.market_data[pos_symbol].loc[date, 'returns']
                        if not np.isnan(return_val):
                            portfolio_return += position * return_val
                            total_value += abs(position)
                
                if total_value > 0:
                    portfolio_returns.append(portfolio_return / total_value)
                else:
                    portfolio_returns.append(0)
            
            if len(portfolio_returns) < 30:
                return 0.0
            
            # Calculate correlation
            correlation = np.corrcoef(symbol_returns.values, portfolio_returns)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating portfolio correlation for {symbol}: {e}")
            return 0.0
    
    def _calculate_liquidity_score(self, symbol: str) -> float:
        """Calculate liquidity score based on volume and spread"""
        if symbol not in self.market_data:
            return 0.5  # Default medium liquidity
        
        try:
            data = self.market_data[symbol]
            
            # Average daily volume (last 30 days)
            avg_volume = data['Volume'].tail(30).mean()
            
            # Volume score (normalized)
            volume_score = min(1.0, avg_volume / 1000000)  # 1M shares = perfect liquidity
            
            # Spread estimate (High-Low relative to Close)
            spread_estimate = ((data['High'] - data['Low']) / data['Close']).tail(30).mean()
            spread_score = max(0.0, 1.0 - spread_estimate * 10)  # Lower spread = higher score
            
            # Combined liquidity score
            liquidity_score = (volume_score * 0.7 + spread_score * 0.3)
            
            return max(0.0, min(1.0, liquidity_score))
            
        except Exception as e:
            logger.error(f"Error calculating liquidity score for {symbol}: {e}")
            return 0.5
    
    def _calculate_risk_score(self, volatility: float, beta: float, 
                            correlation: float, liquidity_score: float) -> float:
        """Calculate overall risk score for position"""
        # Normalize components
        vol_score = min(1.0, volatility / 0.5)  # 50% volatility = max risk
        beta_score = min(1.0, abs(beta) / 2.0)  # Beta of 2 = max risk
        corr_score = abs(correlation)  # High correlation = higher risk
        liquidity_risk = 1.0 - liquidity_score  # Low liquidity = higher risk
        
        # Weighted risk score
        risk_score = (vol_score * 0.4 + beta_score * 0.2 + 
                     corr_score * 0.2 + liquidity_risk * 0.2)
        
        return max(0.0, min(1.0, risk_score))
    
    async def assess_signal(self, signal: Dict) -> Dict[str, Any]:
        """Assess risk for a trading signal"""
        try:
            symbol = signal['symbol']
            action = signal['action']
            confidence = signal.get('confidence', 0.5)
            
            # Basic risk assessment
            risk_score = 0.0
            reasons = []
            
            # Check symbol volatility
            if symbol in self.volatility_estimates:
                volatility = self.volatility_estimates[symbol].get('ewma', 0.2)
                if volatility > 0.4:  # High volatility
                    risk_score += 0.3
                    reasons.append(f"High volatility: {volatility:.2%}")
            
            # Check confidence level
            if confidence < 0.6:
                risk_score += 0.2
                reasons.append(f"Low confidence: {confidence:.2f}")
            
            # Check market conditions
            if self._is_market_stressed():
                risk_score += 0.2
                reasons.append("Market stress detected")
            
            # Check position concentration
            if self._would_exceed_concentration(symbol):
                risk_score += 0.3
                reasons.append("Would exceed concentration limits")
            
            # Determine approval
            approved = risk_score < 0.5
            
            return {
                'approved': approved,
                'risk_score': risk_score,
                'reasons': reasons,
                'confidence_adjustment': max(0.1, confidence - risk_score)
            }
            
        except Exception as e:
            logger.error(f"Error assessing signal: {e}")
            return {
                'approved': False,
                'risk_score': 1.0,
                'reasons': [f"Assessment error: {e}"],
                'confidence_adjustment': 0.1
            }
    
    def _is_market_stressed(self) -> bool:
        """Check if market is in stressed conditions"""
        try:
            # Simple market stress indicators
            if 'SPY' in self.market_data:
                spy_data = self.market_data['SPY']
                recent_returns = spy_data['returns'].tail(5)
                
                # Check for high volatility or large negative returns
                if recent_returns.std() > 0.03 or recent_returns.mean() < -0.02:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking market stress: {e}")
            return False
    
    def _would_exceed_concentration(self, symbol: str) -> bool:
        """Check if adding position would exceed concentration limits"""
        try:
            # Calculate current concentration
            total_value = sum(abs(pos) for pos in self.positions.values())
            symbol_value = abs(self.positions.get(symbol, 0))
            
            if total_value == 0:
                return False
            
            current_concentration = symbol_value / total_value
            return current_concentration > self.risk_limits['max_sector_concentration']
            
        except Exception as e:
            logger.error(f"Error checking concentration: {e}")
            return False
    
    def calculate_position_size(self, signal: Dict, risk_assessment: Dict) -> float:
        """Calculate optimal position size"""
        try:
            symbol = signal['symbol']
            confidence = risk_assessment.get('confidence_adjustment', 0.5)
            
            # Base position size (percentage of portfolio)
            base_size = self.max_position_size * confidence
            
            # Adjust for volatility
            if symbol in self.volatility_estimates:
                volatility = self.volatility_estimates[symbol].get('ewma', 0.2)
                volatility_adjustment = max(0.5, 1.0 - volatility)
                base_size *= volatility_adjustment
            
            # Convert to dollar amount
            position_value = self.portfolio_value * base_size
            
            # Get current price (mock for now)
            current_price = signal.get('price', 100.0)
            
            # Calculate shares
            shares = position_value / current_price
            
            return max(1, int(shares))  # Minimum 1 share
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 1
    
    def check_risk_limits(self, total_pnl: float, positions: List[Dict]) -> List[Dict]:
        """Check all risk limits and return alerts"""
        alerts = []
        
        try:
            # Check daily loss limit
            if total_pnl < -self.max_daily_loss:
                alerts.append({
                    'severity': 'CRITICAL',
                    'message': f'Daily loss limit exceeded: ${total_pnl:.2f}',
                    'action': 'STOP_TRADING'
                })
            
            # Check drawdown
            if self.daily_pnl:
                returns = pd.Series(self.daily_pnl)
                max_dd = abs(self._calculate_maximum_drawdown(returns))
                
                if max_dd > self.max_drawdown:
                    alerts.append({
                        'severity': 'CRITICAL',
                        'message': f'Maximum drawdown exceeded: {max_dd:.2%}',
                        'action': 'STOP_TRADING'
                    })
            
            # Check position concentration
            if positions:
                total_value = sum(pos.get('market_value', 0) for pos in positions)
                
                for position in positions:
                    if total_value > 0:
                        concentration = position.get('market_value', 0) / total_value
                        if concentration > self.max_position_size * 1.5:  # 50% buffer
                            alerts.append({
                                'severity': 'WARNING',
                                'message': f'High concentration in {position.get("symbol")}: {concentration:.2%}',
                                'action': 'REDUCE_POSITION_SIZE'
                            })
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error checking risk limits: {e}")
            return [{
                'severity': 'ERROR',
                'message': f'Risk check error: {e}',
                'action': 'REVIEW_SYSTEM'
            }]
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get current risk metrics"""
        try:
            if not self.daily_pnl:
                return {}
            
            returns = pd.Series(self.daily_pnl)
            
            return {
                'var_95': self._calculate_historical_var(returns, 0.95),
                'var_99': self._calculate_historical_var(returns, 0.99),
                'expected_shortfall': self._calculate_expected_shortfall(returns, 0.95),
                'max_drawdown': abs(self._calculate_maximum_drawdown(returns)),
                'volatility': returns.std() * np.sqrt(252),
                'sharpe_ratio': self._calculate_sharpe_ratio(returns),
                'total_positions': len(self.positions),
                'portfolio_value': self.portfolio_value
            }
            
        except Exception as e:
            logger.error(f"Error getting risk metrics: {e}")
            return {}
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        try:
            if len(returns) < 2:
                return 0.0
            
            excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
            
            if returns.std() == 0:
                return 0.0
            
            return excess_returns.mean() / returns.std() * np.sqrt(252)
            
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0
    
    def reduce_position_size(self, factor: float):
        """Reduce position size limits"""
        self.max_position_size *= factor
        logger.info(f"Position size limit reduced by factor {factor} to {self.max_position_size:.2%}")
    
    def update_portfolio_value(self, value: float):
        """Update portfolio value"""
        self.portfolio_value = value
    
    def add_daily_pnl(self, pnl: float):
        """Add daily P&L for tracking"""
        self.daily_pnl.append(pnl)
        
        # Keep only last 252 days (1 year)
        if len(self.daily_pnl) > 252:
            self.daily_pnl = self.daily_pnl[-252:]
