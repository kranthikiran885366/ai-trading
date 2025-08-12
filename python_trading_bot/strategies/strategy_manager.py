#!/usr/bin/env python3
"""
Strategy Manager - Manages multiple trading strategies
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import yfinance as yf
import talib
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class Signal:
    """Trading signal data class"""
    symbol: str
    action: str  # BUY, SELL, HOLD
    confidence: float
    price: float
    timestamp: datetime
    strategy: str
    reasoning: str
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size: Optional[float] = None

class BaseStrategy(ABC):
    """Base class for all trading strategies"""
    
    def __init__(self, name: str, parameters: Dict[str, Any] = None):
        self.name = name
        self.parameters = parameters or {}
        self.is_active = True
        self.performance_metrics = {
            'total_signals': 0,
            'profitable_signals': 0,
            'win_rate': 0.0,
            'avg_return': 0.0,
            'sharpe_ratio': 0.0
        }
    
    @abstractmethod
    def generate_signal(self, symbol: str, data: pd.DataFrame) -> Optional[Signal]:
        """Generate trading signal for given symbol and data"""
        pass
    
    @abstractmethod
    def update_parameters(self, new_parameters: Dict[str, Any]):
        """Update strategy parameters"""
        pass
    
    def validate_data(self, data: pd.DataFrame, min_periods: int = 50) -> bool:
        """Validate if data is sufficient for strategy"""
        return len(data) >= min_periods and not data.empty
    
    def calculate_stop_loss(self, entry_price: float, action: str, atr: float = None) -> float:
        """Calculate stop loss price"""
        if atr is None:
            atr = entry_price * 0.02  # 2% default
        
        if action == 'BUY':
            return entry_price - (2 * atr)
        else:  # SELL
            return entry_price + (2 * atr)
    
    def calculate_take_profit(self, entry_price: float, action: str, atr: float = None) -> float:
        """Calculate take profit price"""
        if atr is None:
            atr = entry_price * 0.02  # 2% default
        
        if action == 'BUY':
            return entry_price + (3 * atr)
        else:  # SELL
            return entry_price - (3 * atr)

class MeanReversionStrategy(BaseStrategy):
    """Mean reversion strategy using Bollinger Bands and RSI"""
    
    def __init__(self, parameters: Dict[str, Any] = None):
        default_params = {
            'bb_period': 20,
            'bb_std': 2,
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'min_confidence': 0.6
        }
        if parameters:
            default_params.update(parameters)
        
        super().__init__("Mean_Reversion_Strategy", default_params)
    
    def generate_signal(self, symbol: str, data: pd.DataFrame) -> Optional[Signal]:
        """Generate mean reversion signal"""
        if not self.validate_data(data, max(self.parameters['bb_period'], self.parameters['rsi_period']) + 10):
            return None
        
        try:
            current_price = data['Close'].iloc[-1]
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(
                data['Close'].values, 
                timeperiod=self.parameters['bb_period'],
                nbdevup=self.parameters['bb_std'],
                nbdevdn=self.parameters['bb_std']
            )
            
            # RSI
            rsi = talib.RSI(data['Close'].values, timeperiod=self.parameters['rsi_period'])
            
            current_rsi = rsi[-1]
            current_bb_upper = bb_upper[-1]
            current_bb_lower = bb_lower[-1]
            
            # Calculate BB position
            bb_position = (current_price - current_bb_lower) / (current_bb_upper - current_bb_lower)
            
            action = None
            confidence = 0.0
            reasoning = ""
            
            # Buy signal: Price near lower BB and RSI oversold
            if (current_price <= current_bb_lower * 1.02 and 
                current_rsi <= self.parameters['rsi_oversold']):
                action = 'BUY'
                confidence = (1 - bb_position) * 0.5 + (1 - current_rsi / 100) * 0.5
                reasoning = f"Mean reversion buy: BB position {bb_position:.2f}, RSI {current_rsi:.1f}"
            
            # Sell signal: Price near upper BB and RSI overbought
            elif (current_price >= current_bb_upper * 0.98 and 
                  current_rsi >= self.parameters['rsi_overbought']):
                action = 'SELL'
                confidence = bb_position * 0.5 + (current_rsi / 100) * 0.5
                reasoning = f"Mean reversion sell: BB position {bb_position:.2f}, RSI {current_rsi:.1f}"
            
            if action and confidence >= self.parameters['min_confidence']:
                atr = self._calculate_atr(data)
                
                return Signal(
                    symbol=symbol,
                    action=action,
                    confidence=confidence,
                    price=current_price,
                    timestamp=datetime.now(),
                    strategy=self.name,
                    reasoning=reasoning,
                    stop_loss=self.calculate_stop_loss(current_price, action, atr),
                    take_profit=self.calculate_take_profit(current_price, action, atr)
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating mean reversion signal for {symbol}: {e}")
            return None
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        try:
            atr = talib.ATR(data['High'].values, data['Low'].values, data['Close'].values, period)
            return atr[-1] if not np.isnan(atr[-1]) else data['Close'].iloc[-1] * 0.02
        except:
            return data['Close'].iloc[-1] * 0.02
    
    def update_parameters(self, new_parameters: Dict[str, Any]):
        """Update strategy parameters"""
        self.parameters.update(new_parameters)

class MomentumStrategy(BaseStrategy):
    """Momentum strategy using MACD and moving averages"""
    
    def __init__(self, parameters: Dict[str, Any] = None):
        default_params = {
            'fast_ma': 12,
            'slow_ma': 26,
            'signal_ma': 9,
            'sma_period': 50,
            'min_confidence': 0.6,
            'volume_threshold': 1.5
        }
        if parameters:
            default_params.update(parameters)
        
        super().__init__("Momentum_Strategy", default_params)
    
    def generate_signal(self, symbol: str, data: pd.DataFrame) -> Optional[Signal]:
        """Generate momentum signal"""
        if not self.validate_data(data, self.parameters['slow_ma'] + 20):
            return None
        
        try:
            current_price = data['Close'].iloc[-1]
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(
                data['Close'].values,
                fastperiod=self.parameters['fast_ma'],
                slowperiod=self.parameters['slow_ma'],
                signalperiod=self.parameters['signal_ma']
            )
            
            # Moving averages
            sma = talib.SMA(data['Close'].values, timeperiod=self.parameters['sma_period'])
            
            # Volume analysis
            avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
            current_volume = data['Volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume
            
            current_macd = macd[-1]
            current_signal = macd_signal[-1]
            current_hist = macd_hist[-1]
            prev_hist = macd_hist[-2]
            current_sma = sma[-1]
            
            action = None
            confidence = 0.0
            reasoning = ""
            
            # Buy signal: MACD bullish crossover, price above SMA, good volume
            if (current_hist > 0 and prev_hist <= 0 and  # MACD histogram turning positive
                current_price > current_sma and  # Price above SMA
                volume_ratio >= self.parameters['volume_threshold']):  # Good volume
                
                action = 'BUY'
                macd_strength = min(1.0, abs(current_macd) / (current_price * 0.01))
                price_strength = (current_price - current_sma) / current_sma
                volume_strength = min(1.0, volume_ratio / 3.0)
                
                confidence = (macd_strength * 0.4 + price_strength * 0.3 + volume_strength * 0.3)
                reasoning = f"Momentum buy: MACD crossover, price above SMA, volume {volume_ratio:.1f}x"
            
            # Sell signal: MACD bearish crossover, price below SMA, good volume
            elif (current_hist < 0 and prev_hist >= 0 and  # MACD histogram turning negative
                  current_price < current_sma and  # Price below SMA
                  volume_ratio >= self.parameters['volume_threshold']):  # Good volume
                
                action = 'SELL'
                macd_strength = min(1.0, abs(current_macd) / (current_price * 0.01))
                price_strength = (current_sma - current_price) / current_sma
                volume_strength = min(1.0, volume_ratio / 3.0)
                
                confidence = (macd_strength * 0.4 + price_strength * 0.3 + volume_strength * 0.3)
                reasoning = f"Momentum sell: MACD crossover, price below SMA, volume {volume_ratio:.1f}x"
            
            if action and confidence >= self.parameters['min_confidence']:
                atr = self._calculate_atr(data)
                
                return Signal(
                    symbol=symbol,
                    action=action,
                    confidence=confidence,
                    price=current_price,
                    timestamp=datetime.now(),
                    strategy=self.name,
                    reasoning=reasoning,
                    stop_loss=self.calculate_stop_loss(current_price, action, atr),
                    take_profit=self.calculate_take_profit(current_price, action, atr)
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating momentum signal for {symbol}: {e}")
            return None
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        try:
            atr = talib.ATR(data['High'].values, data['Low'].values, data['Close'].values, period)
            return atr[-1] if not np.isnan(atr[-1]) else data['Close'].iloc[-1] * 0.02
        except:
            return data['Close'].iloc[-1] * 0.02
    
    def update_parameters(self, new_parameters: Dict[str, Any]):
        """Update strategy parameters"""
        self.parameters.update(new_parameters)

class BreakoutStrategy(BaseStrategy):
    """Breakout strategy using support/resistance levels"""
    
    def __init__(self, parameters: Dict[str, Any] = None):
        default_params = {
            'lookback_period': 20,
            'breakout_threshold': 0.02,  # 2% breakout
            'volume_confirmation': True,
            'volume_threshold': 1.5,
            'min_confidence': 0.6
        }
        if parameters:
            default_params.update(parameters)
        
        super().__init__("Breakout_Strategy", default_params)
    
    def generate_signal(self, symbol: str, data: pd.DataFrame) -> Optional[Signal]:
        """Generate breakout signal"""
        if not self.validate_data(data, self.parameters['lookback_period'] + 10):
            return None
        
        try:
            current_price = data['Close'].iloc[-1]
            lookback = self.parameters['lookback_period']
            
            # Calculate support and resistance levels
            recent_data = data.tail(lookback)
            resistance = recent_data['High'].max()
            support = recent_data['Low'].min()
            
            # Calculate breakout thresholds
            resistance_breakout = resistance * (1 + self.parameters['breakout_threshold'])
            support_breakout = support * (1 - self.parameters['breakout_threshold'])
            
            # Volume confirmation
            volume_confirmed = True
            if self.parameters['volume_confirmation']:
                avg_volume = data['Volume'].tail(20).mean()
                current_volume = data['Volume'].iloc[-1]
                volume_ratio = current_volume / avg_volume
                volume_confirmed = volume_ratio >= self.parameters['volume_threshold']
            
            action = None
            confidence = 0.0
            reasoning = ""
            
            # Bullish breakout
            if current_price > resistance_breakout and volume_confirmed:
                action = 'BUY'
                breakout_strength = (current_price - resistance) / resistance
                confidence = min(1.0, breakout_strength * 10)  # Scale breakout strength
                reasoning = f"Bullish breakout above {resistance:.2f}, strength: {breakout_strength:.2%}"
            
            # Bearish breakdown
            elif current_price < support_breakout and volume_confirmed:
                action = 'SELL'
                breakdown_strength = (support - current_price) / support
                confidence = min(1.0, breakdown_strength * 10)  # Scale breakdown strength
                reasoning = f"Bearish breakdown below {support:.2f}, strength: {breakdown_strength:.2%}"
            
            if action and confidence >= self.parameters['min_confidence']:
                atr = self._calculate_atr(data)
                
                return Signal(
                    symbol=symbol,
                    action=action,
                    confidence=confidence,
                    price=current_price,
                    timestamp=datetime.now(),
                    strategy=self.name,
                    reasoning=reasoning,
                    stop_loss=self.calculate_stop_loss(current_price, action, atr),
                    take_profit=self.calculate_take_profit(current_price, action, atr)
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating breakout signal for {symbol}: {e}")
            return None
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        try:
            atr = talib.ATR(data['High'].values, data['Low'].values, data['Close'].values, period)
            return atr[-1] if not np.isnan(atr[-1]) else data['Close'].iloc[-1] * 0.02
        except:
            return data['Close'].iloc[-1] * 0.02
    
    def update_parameters(self, new_parameters: Dict[str, Any]):
        """Update strategy parameters"""
        self.parameters.update(new_parameters)

class StrategyManager:
    """Manages multiple trading strategies"""
    
    def __init__(self, strategy_configs: List[str] = None):
        self.strategies = {}
        self.strategy_weights = {}
        self.strategy_performance = {}
        
        # Initialize strategies
        if strategy_configs:
            self._initialize_strategies(strategy_configs)
        else:
            # Default strategies
            self._initialize_default_strategies()
    
    def _initialize_strategies(self, strategy_configs: List[str]):
        """Initialize strategies based on configuration"""
        for config in strategy_configs:
            if config == 'mean_reversion':
                self.add_strategy(MeanReversionStrategy())
            elif config == 'momentum':
                self.add_strategy(MomentumStrategy())
            elif config == 'breakout':
                self.add_strategy(BreakoutStrategy())
    
    def _initialize_default_strategies(self):
        """Initialize default strategies"""
        self.add_strategy(MeanReversionStrategy())
        self.add_strategy(MomentumStrategy())
        self.add_strategy(BreakoutStrategy())
    
    def add_strategy(self, strategy: BaseStrategy, weight: float = 1.0):
        """Add a strategy to the manager"""
        self.strategies[strategy.name] = strategy
        self.strategy_weights[strategy.name] = weight
        self.strategy_performance[strategy.name] = {
            'signals_generated': 0,
            'successful_signals': 0,
            'total_return': 0.0,
            'win_rate': 0.0,
            'avg_confidence': 0.0
        }
        
        logger.info(f"Added strategy: {strategy.name}")
    
    def remove_strategy(self, strategy_name: str):
        """Remove a strategy"""
        if strategy_name in self.strategies:
            del self.strategies[strategy_name]
            del self.strategy_weights[strategy_name]
            del self.strategy_performance[strategy_name]
            logger.info(f"Removed strategy: {strategy_name}")
    
    def generate_signals(self, symbol: str, data: pd.DataFrame) -> List[Signal]:
        """Generate signals from all active strategies"""
        signals = []
        
        for strategy_name, strategy in self.strategies.items():
            if not strategy.is_active:
                continue
            
            try:
                signal = strategy.generate_signal(symbol, data)
                if signal:
                    signals.append(signal)
                    self.strategy_performance[strategy_name]['signals_generated'] += 1
                    
            except Exception as e:
                logger.error(f"Error generating signal from {strategy_name}: {e}")
        
        return signals
    
    def combine_signals(self, signals: List[Signal]) -> Optional[Signal]:
        """Combine multiple signals into a single signal"""
        if not signals:
            return None
        
        # Group signals by action
        buy_signals = [s for s in signals if s.action == 'BUY']
        sell_signals = [s for s in signals if s.action == 'SELL']
        
        # Calculate weighted confidence for each action
        buy_confidence = self._calculate_weighted_confidence(buy_signals)
        sell_confidence = self._calculate_weighted_confidence(sell_signals)
        
        # Determine final action
        if buy_confidence > sell_confidence and buy_confidence > 0.6:
            final_action = 'BUY'
            final_confidence = buy_confidence
            contributing_signals = buy_signals
        elif sell_confidence > buy_confidence and sell_confidence > 0.6:
            final_action = 'SELL'
            final_confidence = sell_confidence
            contributing_signals = sell_signals
        else:
            return None  # No clear consensus
        
        # Use the signal with highest confidence as base
        base_signal = max(contributing_signals, key=lambda x: x.confidence)
        
        # Create combined reasoning
        strategy_names = [s.strategy for s in contributing_signals]
        reasoning = f"Combined signal from {', '.join(strategy_names)} (confidence: {final_confidence:.2f})"
        
        return Signal(
            symbol=base_signal.symbol,
            action=final_action,
            confidence=final_confidence,
            price=base_signal.price,
            timestamp=datetime.now(),
            strategy="Combined_Strategy",
            reasoning=reasoning,
            stop_loss=base_signal.stop_loss,
            take_profit=base_signal.take_profit
        )
    
    def _calculate_weighted_confidence(self, signals: List[Signal]) -> float:
        """Calculate weighted confidence for a group of signals"""
        if not signals:
            return 0.0
        
        total_weight = 0.0
        weighted_confidence = 0.0
        
        for signal in signals:
            strategy_weight = self.strategy_weights.get(signal.strategy, 1.0)
            performance_weight = self._get_performance_weight(signal.strategy)
            
            combined_weight = strategy_weight * performance_weight
            weighted_confidence += signal.confidence * combined_weight
            total_weight += combined_weight
        
        return weighted_confidence / total_weight if total_weight > 0 else 0.0
    
    def _get_performance_weight(self, strategy_name: str) -> float:
        """Get performance-based weight for strategy"""
        if strategy_name not in self.strategy_performance:
            return 1.0
        
        perf = self.strategy_performance[strategy_name]
        
        # Base weight on win rate and number of signals
        if perf['signals_generated'] < 10:
            return 1.0  # Not enough data
        
        win_rate = perf['win_rate']
        signal_count = min(perf['signals_generated'], 100)  # Cap at 100
        
        # Weight based on win rate and experience
        performance_weight = (win_rate * 0.7 + (signal_count / 100) * 0.3)
        
        return max(0.1, min(2.0, performance_weight))  # Clamp between 0.1 and 2.0
    
    def update_strategy_performance(self, strategy_name: str, signal_result: Dict[str, Any]):
        """Update strategy performance metrics"""
        if strategy_name not in self.strategy_performance:
            return
        
        perf = self.strategy_performance[strategy_name]
        
        if signal_result.get('successful', False):
            perf['successful_signals'] += 1
        
        if 'return' in signal_result:
            perf['total_return'] += signal_result['return']
        
        # Update win rate
        if perf['signals_generated'] > 0:
            perf['win_rate'] = perf['successful_signals'] / perf['signals_generated']
    
    def get_strategy_rankings(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Get strategies ranked by performance"""
        rankings = []
        
        for name, perf in self.strategy_performance.items():
            score = 0.0
            
            if perf['signals_generated'] >= 10:
                score = (perf['win_rate'] * 0.6 + 
                        min(1.0, perf['signals_generated'] / 50) * 0.2 +
                        max(0.0, min(1.0, perf['total_return'] / 100)) * 0.2)
            
            rankings.append((name, {**perf, 'score': score}))
        
        return sorted(rankings, key=lambda x: x[1]['score'], reverse=True)
    
    def optimize_strategy_weights(self):
        """Optimize strategy weights based on performance"""
        rankings = self.get_strategy_rankings()
        
        total_score = sum(rank[1]['score'] for rank in rankings)
        
        if total_score > 0:
            for name, perf in rankings:
                # Adjust weights based on performance
                new_weight = (perf['score'] / total_score) * len(rankings)
                self.strategy_weights[name] = max(0.1, min(3.0, new_weight))
        
        logger.info("Strategy weights optimized based on performance")
    
    def get_status(self) -> Dict[str, Any]:
        """Get strategy manager status"""
        return {
            'active_strategies': len([s for s in self.strategies.values() if s.is_active]),
            'total_strategies': len(self.strategies),
            'strategy_weights': self.strategy_weights,
            'strategy_performance': self.strategy_performance,
            'rankings': self.get_strategy_rankings()
        }
    
    def start(self):
        """Start strategy manager"""
        logger.info("Strategy manager started")
    
    def stop(self):
        """Stop strategy manager"""
        logger.info("Strategy manager stopped")
