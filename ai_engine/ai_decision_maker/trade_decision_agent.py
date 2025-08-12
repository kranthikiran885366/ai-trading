#!/usr/bin/env python3
"""
Trade Decision Agent - Advanced AI agent for making trading decisions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
import json
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class DecisionType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE_LONG = "CLOSE_LONG"
    CLOSE_SHORT = "CLOSE_SHORT"

class ConfidenceLevel(Enum):
    VERY_LOW = 0.2
    LOW = 0.4
    MEDIUM = 0.6
    HIGH = 0.8
    VERY_HIGH = 0.9

@dataclass
class TradingSignal:
    """Trading signal from various sources"""
    source: str
    symbol: str
    action: str
    confidence: float
    price: float
    timestamp: datetime
    reasoning: str
    features: Dict[str, float]
    metadata: Dict[str, Any]

@dataclass
class MarketContext:
    """Market context information"""
    market_regime: str
    volatility: float
    trend_strength: float
    sentiment_score: float
    economic_indicators: Dict[str, float]
    sector_performance: Dict[str, float]
    correlation_risk: float

@dataclass
class TradeDecision:
    """Final trade decision"""
    symbol: str
    decision: DecisionType
    confidence: float
    position_size: float
    entry_price: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    reasoning: str
    risk_score: float
    expected_return: float
    holding_period: Optional[int]
    metadata: Dict[str, Any]

class TradeDecisionAgent:
    """Advanced AI agent for making trading decisions"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Decision models
        self.decision_model = None
        self.confidence_model = None
        self.risk_model = None
        
        # Feature processing
        self.scaler = StandardScaler()
        self.feature_columns = []
        
        # Decision history
        self.decision_history = []
        self.performance_metrics = {}
        
        # Thresholds and parameters
        self.min_confidence_threshold = self.config.get('min_confidence', 0.6)
        self.max_risk_score = self.config.get('max_risk_score', 0.7)
        self.position_sizing_method = self.config.get('position_sizing', 'kelly')
        
        # Market context weights
        self.context_weights = {
            'technical': 0.4,
            'fundamental': 0.2,
            'sentiment': 0.2,
            'market_regime': 0.2
        }
        
        # Initialize models
        self._initialize_models()
        
        logger.info("Trade Decision Agent initialized")
    
    def _initialize_models(self):
        """Initialize decision-making models"""
        try:
            # Decision model - predicts BUY/SELL/HOLD
            self.decision_model = GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=8,
                random_state=42
            )
            
            # Confidence model - predicts confidence in decision
            self.confidence_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
            # Risk model - predicts risk score
            self.risk_model = GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
            
            logger.info("Decision models initialized")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise
    
    def make_decision(self, signals: List[TradingSignal], 
                     market_context: MarketContext,
                     current_positions: Dict[str, Any] = None) -> List[TradeDecision]:
        """Make trading decisions based on signals and market context"""
        try:
            decisions = []
            
            # Group signals by symbol
            signals_by_symbol = self._group_signals_by_symbol(signals)
            
            for symbol, symbol_signals in signals_by_symbol.items():
                try:
                    decision = self._make_symbol_decision(
                        symbol, symbol_signals, market_context, current_positions
                    )
                    
                    if decision:
                        decisions.append(decision)
                        
                except Exception as e:
                    logger.error(f"Error making decision for {symbol}: {e}")
                    continue
            
            # Record decisions
            self._record_decisions(decisions)
            
            return decisions
            
        except Exception as e:
            logger.error(f"Error making trading decisions: {e}")
            return []
    
    def _group_signals_by_symbol(self, signals: List[TradingSignal]) -> Dict[str, List[TradingSignal]]:
        """Group signals by symbol"""
        grouped = {}
        for signal in signals:
            if signal.symbol not in grouped:
                grouped[signal.symbol] = []
            grouped[signal.symbol].append(signal)
        return grouped
    
    def _make_symbol_decision(self, symbol: str, signals: List[TradingSignal],
                            market_context: MarketContext,
                            current_positions: Dict[str, Any] = None) -> Optional[TradeDecision]:
        """Make decision for a specific symbol"""
        try:
            # Extract features from signals and context
            features = self._extract_decision_features(signals, market_context, symbol)
            
            if not features:
                return None
            
            # Predict decision
            decision_proba = self._predict_decision(features)
            decision_type = self._get_decision_type(decision_proba)
            
            # Predict confidence
            confidence = self._predict_confidence(features, decision_proba)
            
            # Predict risk
            risk_score = self._predict_risk(features)
            
            # Check if decision meets thresholds
            if confidence < self.min_confidence_threshold or risk_score > self.max_risk_score:
                logger.info(f"Decision for {symbol} rejected: confidence={confidence:.2f}, risk={risk_score:.2f}")
                return None
            
            # Calculate position size
            position_size = self._calculate_position_size(
                symbol, decision_type, confidence, risk_score, market_context
            )
            
            # Get current price
            current_price = self._get_current_price(symbol, signals)
            
            # Calculate stop loss and take profit
            stop_loss, take_profit = self._calculate_stop_take_levels(
                symbol, decision_type, current_price, risk_score, market_context
            )
            
            # Calculate expected return
            expected_return = self._calculate_expected_return(
                decision_type, current_price, take_profit, stop_loss, confidence
            )
            
            # Generate reasoning
            reasoning = self._generate_reasoning(signals, decision_type, confidence, risk_score)
            
            # Estimate holding period
            holding_period = self._estimate_holding_period(
                symbol, decision_type, market_context, signals
            )
            
            return TradeDecision(
                symbol=symbol,
                decision=decision_type,
                confidence=confidence,
                position_size=position_size,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reasoning=reasoning,
                risk_score=risk_score,
                expected_return=expected_return,
                holding_period=holding_period,
                metadata={
                    'signals_count': len(signals),
                    'market_regime': market_context.market_regime,
                    'volatility': market_context.volatility,
                    'sentiment': market_context.sentiment_score,
                    'decision_timestamp': datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Error making decision for {symbol}: {e}")
            return None
    
    def _extract_decision_features(self, signals: List[TradingSignal],
                                 market_context: MarketContext,
                                 symbol: str) -> Optional[Dict[str, float]]:
        """Extract features for decision making"""
        try:
            features = {}
            
            # Signal aggregation features
            if signals:
                buy_signals = [s for s in signals if s.action == 'BUY']
                sell_signals = [s for s in signals if s.action == 'SELL']
                
                features['buy_signal_count'] = len(buy_signals)
                features['sell_signal_count'] = len(sell_signals)
                features['total_signal_count'] = len(signals)
                
                # Average confidence by action
                features['avg_buy_confidence'] = np.mean([s.confidence for s in buy_signals]) if buy_signals else 0
                features['avg_sell_confidence'] = np.mean([s.confidence for s in sell_signals]) if sell_signals else 0
                features['avg_total_confidence'] = np.mean([s.confidence for s in signals])
                
                # Signal strength
                features['buy_strength'] = sum(s.confidence for s in buy_signals)
                features['sell_strength'] = sum(s.confidence for s in sell_signals)
                features['net_signal_strength'] = features['buy_strength'] - features['sell_strength']
                
                # Signal diversity (number of unique sources)
                unique_sources = len(set(s.source for s in signals))
                features['signal_diversity'] = unique_sources / len(signals) if signals else 0
                
                # Time-based features
                signal_times = [s.timestamp for s in signals]
                if len(signal_times) > 1:
                    time_span = (max(signal_times) - min(signal_times)).total_seconds() / 3600
                    features['signal_time_span_hours'] = time_span
                else:
                    features['signal_time_span_hours'] = 0
                
                # Feature aggregation from signals
                all_features = {}
                for signal in signals:
                    for key, value in signal.features.items():
                        if key not in all_features:
                            all_features[key] = []
                        all_features[key].append(value)
                
                # Statistical features from signal features
                for key, values in all_features.items():
                    if values:
                        features[f'{key}_mean'] = np.mean(values)
                        features[f'{key}_std'] = np.std(values)
                        features[f'{key}_max'] = np.max(values)
                        features[f'{key}_min'] = np.min(values)
            
            # Market context features
            features['market_regime_bull'] = 1 if market_context.market_regime == 'BULL' else 0
            features['market_regime_bear'] = 1 if market_context.market_regime == 'BEAR' else 0
            features['market_regime_sideways'] = 1 if market_context.market_regime == 'SIDEWAYS' else 0
            features['market_regime_volatile'] = 1 if market_context.market_regime == 'VOLATILE' else 0
            
            features['market_volatility'] = market_context.volatility
            features['trend_strength'] = market_context.trend_strength
            features['sentiment_score'] = market_context.sentiment_score
            features['correlation_risk'] = market_context.correlation_risk
            
            # Economic indicators
            for indicator, value in market_context.economic_indicators.items():
                features[f'econ_{indicator}'] = value
            
            # Sector performance
            for sector, performance in market_context.sector_performance.items():
                features[f'sector_{sector}'] = performance
            
            # Time-based features
            now = datetime.now()
            features['hour_of_day'] = now.hour
            features['day_of_week'] = now.weekday()
            features['is_market_open'] = 1 if 9 <= now.hour <= 16 and now.weekday() < 5 else 0
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting decision features: {e}")
            return None
    
    def _predict_decision(self, features: Dict[str, float]) -> np.ndarray:
        """Predict trading decision probabilities"""
        try:
            # Convert features to array
            feature_array = self._features_to_array(features)
            
            if self.decision_model and hasattr(self.decision_model, 'predict_proba'):
                # Use trained model
                probabilities = self.decision_model.predict_proba(feature_array.reshape(1, -1))[0]
            else:
                # Fallback to rule-based decision
                probabilities = self._rule_based_decision(features)
            
            return probabilities
            
        except Exception as e:
            logger.error(f"Error predicting decision: {e}")
            # Return neutral probabilities
            return np.array([0.33, 0.34, 0.33])  # [SELL, HOLD, BUY]
    
    def _predict_confidence(self, features: Dict[str, float], decision_proba: np.ndarray) -> float:
        """Predict confidence in the decision"""
        try:
            # Base confidence on decision probability
            base_confidence = np.max(decision_proba)
            
            # Adjust based on signal strength
            signal_strength = features.get('net_signal_strength', 0)
            signal_diversity = features.get('signal_diversity', 0)
            
            # Confidence adjustments
            confidence_adjustments = []
            
            # Signal strength adjustment
            if abs(signal_strength) > 2:
                confidence_adjustments.append(0.1)
            elif abs(signal_strength) > 1:
                confidence_adjustments.append(0.05)
            
            # Signal diversity adjustment
            if signal_diversity > 0.7:
                confidence_adjustments.append(0.1)
            elif signal_diversity > 0.5:
                confidence_adjustments.append(0.05)
            
            # Market regime adjustment
            if features.get('market_regime_volatile', 0) == 1:
                confidence_adjustments.append(-0.1)
            
            # Volatility adjustment
            volatility = features.get('market_volatility', 0.2)
            if volatility > 0.4:
                confidence_adjustments.append(-0.1)
            elif volatility < 0.1:
                confidence_adjustments.append(0.05)
            
            # Apply adjustments
            final_confidence = base_confidence + sum(confidence_adjustments)
            
            # Clamp between 0 and 1
            return max(0.0, min(1.0, final_confidence))
            
        except Exception as e:
            logger.error(f"Error predicting confidence: {e}")
            return 0.5
    
    def _predict_risk(self, features: Dict[str, float]) -> float:
        """Predict risk score for the decision"""
        try:
            risk_factors = []
            
            # Volatility risk
            volatility = features.get('market_volatility', 0.2)
            risk_factors.append(min(1.0, volatility / 0.5))
            
            # Market regime risk
            if features.get('market_regime_volatile', 0) == 1:
                risk_factors.append(0.8)
            elif features.get('market_regime_bear', 0) == 1:
                risk_factors.append(0.6)
            else:
                risk_factors.append(0.3)
            
            # Correlation risk
            correlation_risk = features.get('correlation_risk', 0.5)
            risk_factors.append(correlation_risk)
            
            # Signal consistency risk
            buy_conf = features.get('avg_buy_confidence', 0)
            sell_conf = features.get('avg_sell_confidence', 0)
            if abs(buy_conf - sell_conf) < 0.2:  # Conflicting signals
                risk_factors.append(0.7)
            else:
                risk_factors.append(0.3)
            
            # Time-based risk
            if features.get('is_market_open', 0) == 0:
                risk_factors.append(0.6)  # Higher risk outside market hours
            else:
                risk_factors.append(0.2)
            
            # Calculate weighted average
            risk_score = np.mean(risk_factors)
            
            return max(0.0, min(1.0, risk_score))
            
        except Exception as e:
            logger.error(f"Error predicting risk: {e}")
            return 0.5
    
    def _get_decision_type(self, probabilities: np.ndarray) -> DecisionType:
        """Convert probabilities to decision type"""
        try:
            # Assuming probabilities are [SELL, HOLD, BUY]
            decision_idx = np.argmax(probabilities)
            
            if decision_idx == 0:
                return DecisionType.SELL
            elif decision_idx == 1:
                return DecisionType.HOLD
            else:
                return DecisionType.BUY
                
        except Exception as e:
            logger.error(f"Error getting decision type: {e}")
            return DecisionType.HOLD
    
    def _calculate_position_size(self, symbol: str, decision_type: DecisionType,
                               confidence: float, risk_score: float,
                               market_context: MarketContext) -> float:
        """Calculate optimal position size"""
        try:
            if decision_type == DecisionType.HOLD:
                return 0.0
            
            # Base position size (percentage of portfolio)
            base_size = 0.05  # 5% base allocation
            
            # Adjust for confidence
            confidence_multiplier = confidence
            
            # Adjust for risk
            risk_multiplier = 1.0 - risk_score
            
            # Adjust for market regime
            regime_multiplier = 1.0
            if market_context.market_regime == 'BULL':
                regime_multiplier = 1.2
            elif market_context.market_regime == 'BEAR':
                regime_multiplier = 0.8
            elif market_context.market_regime == 'VOLATILE':
                regime_multiplier = 0.6
            
            # Adjust for volatility
            volatility_multiplier = max(0.5, 1.0 - market_context.volatility)
            
            # Calculate final position size
            position_size = (base_size * confidence_multiplier * 
                           risk_multiplier * regime_multiplier * volatility_multiplier)
            
            # Apply position sizing method
            if self.position_sizing_method == 'kelly':
                position_size = self._kelly_position_size(confidence, risk_score)
            elif self.position_sizing_method == 'fixed_fractional':
                position_size = min(position_size, 0.1)  # Max 10%
            
            return max(0.01, min(0.2, position_size))  # Between 1% and 20%
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.02  # Default 2%
    
    def _kelly_position_size(self, confidence: float, risk_score: float) -> float:
        """Calculate position size using Kelly criterion"""
        try:
            # Estimate win probability and win/loss ratio
            win_prob = confidence
            loss_prob = 1 - win_prob
            
            # Estimate win/loss ratio based on risk score
            win_loss_ratio = (1 - risk_score) / risk_score if risk_score > 0 else 1
            
            # Kelly formula: f = (bp - q) / b
            # where b = win/loss ratio, p = win probability, q = loss probability
            kelly_fraction = (win_loss_ratio * win_prob - loss_prob) / win_loss_ratio
            
            # Apply safety factor
            kelly_fraction *= 0.25  # Use quarter Kelly for safety
            
            return max(0.01, min(0.15, kelly_fraction))
            
        except Exception as e:
            logger.error(f"Error calculating Kelly position size: {e}")
            return 0.05
    
    def _calculate_stop_take_levels(self, symbol: str, decision_type: DecisionType,
                                  current_price: float, risk_score: float,
                                  market_context: MarketContext) -> Tuple[Optional[float], Optional[float]]:
        """Calculate stop loss and take profit levels"""
        try:
            if decision_type == DecisionType.HOLD:
                return None, None
            
            # Base risk percentage
            base_risk = 0.02  # 2%
            
            # Adjust risk based on volatility and risk score
            adjusted_risk = base_risk * (1 + market_context.volatility) * (1 + risk_score)
            
            # Risk-reward ratio
            risk_reward_ratio = 2.0  # 1:2 risk-reward
            
            if decision_type == DecisionType.BUY:
                stop_loss = current_price * (1 - adjusted_risk)
                take_profit = current_price * (1 + adjusted_risk * risk_reward_ratio)
            else:  # SELL
                stop_loss = current_price * (1 + adjusted_risk)
                take_profit = current_price * (1 - adjusted_risk * risk_reward_ratio)
            
            return stop_loss, take_profit
            
        except Exception as e:
            logger.error(f"Error calculating stop/take levels: {e}")
            return None, None
    
    def _calculate_expected_return(self, decision_type: DecisionType,
                                 current_price: float, take_profit: Optional[float],
                                 stop_loss: Optional[float], confidence: float) -> float:
        """Calculate expected return for the trade"""
        try:
            if decision_type == DecisionType.HOLD or not take_profit or not stop_loss:
                return 0.0
            
            if decision_type == DecisionType.BUY:
                potential_gain = (take_profit - current_price) / current_price
                potential_loss = (current_price - stop_loss) / current_price
            else:  # SELL
                potential_gain = (current_price - take_profit) / current_price
                potential_loss = (stop_loss - current_price) / current_price
            
            # Expected return = (win_prob * gain) - (loss_prob * loss)
            win_prob = confidence
            loss_prob = 1 - win_prob
            
            expected_return = (win_prob * potential_gain) - (loss_prob * potential_loss)
            
            return expected_return
            
        except Exception as e:
            logger.error(f"Error calculating expected return: {e}")
            return 0.0
    
    def _generate_reasoning(self, signals: List[TradingSignal],
                          decision_type: DecisionType, confidence: float,
                          risk_score: float) -> str:
        """Generate human-readable reasoning for the decision"""
        try:
            reasoning_parts = []
            
            # Decision summary
            reasoning_parts.append(f"Decision: {decision_type.value} with {confidence:.1%} confidence")
            
            # Signal analysis
            if signals:
                buy_signals = len([s for s in signals if s.action == 'BUY'])
                sell_signals = len([s for s in signals if s.action == 'SELL'])
                
                reasoning_parts.append(f"Based on {len(signals)} signals ({buy_signals} buy, {sell_signals} sell)")
                
                # Top signal sources
                sources = [s.source for s in signals]
                top_sources = list(set(sources))[:3]
                reasoning_parts.append(f"Key sources: {', '.join(top_sources)}")
            
            # Risk assessment
            if risk_score < 0.3:
                risk_level = "Low"
            elif risk_score < 0.6:
                risk_level = "Medium"
            else:
                risk_level = "High"
            
            reasoning_parts.append(f"Risk level: {risk_level} ({risk_score:.1%})")
            
            return ". ".join(reasoning_parts)
            
        except Exception as e:
            logger.error(f"Error generating reasoning: {e}")
            return f"Decision: {decision_type.value}"
    
    def _estimate_holding_period(self, symbol: str, decision_type: DecisionType,
                               market_context: MarketContext,
                               signals: List[TradingSignal]) -> Optional[int]:
        """Estimate optimal holding period in hours"""
        try:
            if decision_type == DecisionType.HOLD:
                return None
            
            # Base holding period based on market regime
            if market_context.market_regime == 'VOLATILE':
                base_period = 4  # 4 hours
            elif market_context.market_regime == 'TRENDING':
                base_period = 24  # 1 day
            else:
                base_period = 12  # 12 hours
            
            # Adjust based on volatility
            volatility_multiplier = 1 / (1 + market_context.volatility)
            
            # Adjust based on signal strength
            signal_strength = sum(s.confidence for s in signals) / len(signals) if signals else 0.5
            strength_multiplier = 0.5 + signal_strength
            
            holding_period = int(base_period * volatility_multiplier * strength_multiplier)
            
            return max(1, min(168, holding_period))  # Between 1 hour and 1 week
            
        except Exception as e:
            logger.error(f"Error estimating holding period: {e}")
            return 12  # Default 12 hours
    
    def _get_current_price(self, symbol: str, signals: List[TradingSignal]) -> float:
        """Get current price from signals or market data"""
        try:
            if signals:
                # Use most recent signal price
                latest_signal = max(signals, key=lambda s: s.timestamp)
                return latest_signal.price
            
            # Fallback to fetching current price
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='1d', interval='1m')
            
            if not data.empty:
                return data['Close'].iloc[-1]
            
            return 100.0  # Default fallback price
            
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return 100.0
    
    def _features_to_array(self, features: Dict[str, float]) -> np.ndarray:
        """Convert features dictionary to numpy array"""
        try:
            if not self.feature_columns:
                self.feature_columns = sorted(features.keys())
            
            # Ensure all expected features are present
            feature_array = []
            for col in self.feature_columns:
                feature_array.append(features.get(col, 0.0))
            
            return np.array(feature_array)
            
        except Exception as e:
            logger.error(f"Error converting features to array: {e}")
            return np.zeros(len(self.feature_columns) if self.feature_columns else 10)
    
    def _rule_based_decision(self, features: Dict[str, float]) -> np.ndarray:
        """Fallback rule-based decision making"""
        try:
            buy_score = 0.0
            sell_score = 0.0
            
            # Signal-based scoring
            buy_strength = features.get('buy_strength', 0)
            sell_strength = features.get('sell_strength', 0)
            
            buy_score += buy_strength * 0.4
            sell_score += sell_strength * 0.4
            
            # Market regime scoring
            if features.get('market_regime_bull', 0) == 1:
                buy_score += 0.2
            elif features.get('market_regime_bear', 0) == 1:
                sell_score += 0.2
            
            # Sentiment scoring
            sentiment = features.get('sentiment_score', 0)
            if sentiment > 0.1:
                buy_score += sentiment * 0.2
            elif sentiment < -0.1:
                sell_score += abs(sentiment) * 0.2
            
            # Normalize scores
            total_score = buy_score + sell_score
            if total_score > 0:
                buy_prob = buy_score / total_score
                sell_prob = sell_score / total_score
                hold_prob = max(0.1, 1 - buy_prob - sell_prob)
            else:
                buy_prob = sell_prob = 0.2
                hold_prob = 0.6
            
            # Normalize to sum to 1
            total = buy_prob + sell_prob + hold_prob
            return np.array([sell_prob/total, hold_prob/total, buy_prob/total])
            
        except Exception as e:
            logger.error(f"Error in rule-based decision: {e}")
            return np.array([0.33, 0.34, 0.33])
    
    def _record_decisions(self, decisions: List[TradeDecision]):
        """Record decisions for performance tracking"""
        try:
            for decision in decisions:
                self.decision_history.append({
                    'timestamp': datetime.now(),
                    'symbol': decision.symbol,
                    'decision': decision.decision.value,
                    'confidence': decision.confidence,
                    'risk_score': decision.risk_score,
                    'expected_return': decision.expected_return,
                    'position_size': decision.position_size
                })
            
            # Keep only recent decisions
            if len(self.decision_history) > 1000:
                self.decision_history = self.decision_history[-500:]
                
        except Exception as e:
            logger.error(f"Error recording decisions: {e}")
    
    def train_models(self, training_data: pd.DataFrame):
        """Train decision-making models"""
        try:
            logger.info("Training decision models...")
            
            # Prepare features and targets
            feature_cols = [col for col in training_data.columns 
                          if col not in ['decision', 'confidence', 'risk_score', 'symbol', 'timestamp']]
            
            X = training_data[feature_cols].fillna(0)
            
            # Train decision model
            if 'decision' in training_data.columns:
                y_decision = training_data['decision']
                self.decision_model.fit(X, y_decision)
                logger.info("Decision model trained")
            
            # Train confidence model
            if 'confidence' in training_data.columns:
                y_confidence = (training_data['confidence'] > 0.7).astype(int)
                self.confidence_model.fit(X, y_confidence)
                logger.info("Confidence model trained")
            
            # Train risk model
            if 'risk_score' in training_data.columns:
                y_risk = (training_data['risk_score'] > 0.5).astype(int)
                self.risk_model.fit(X, y_risk)
                logger.info("Risk model trained")
            
            # Store feature columns
            self.feature_columns = feature_cols
            
            # Fit scaler
            self.scaler.fit(X)
            
            logger.info("All models trained successfully")
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            raise
    
    def save_models(self, filepath: str):
        """Save trained models"""
        try:
            model_data = {
                'decision_model': self.decision_model,
                'confidence_model': self.confidence_model,
                'risk_model': self.risk_model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'config': self.config
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Models saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            raise
    
    def load_models(self, filepath: str):
        """Load trained models"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.decision_model = model_data['decision_model']
            self.confidence_model = model_data['confidence_model']
            self.risk_model = model_data['risk_model']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            
            logger.info(f"Models loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the decision agent"""
        try:
            if not self.decision_history:
                return {}
            
            decisions_df = pd.DataFrame(self.decision_history)
            
            metrics = {
                'total_decisions': len(decisions_df),
                'buy_decisions': len(decisions_df[decisions_df['decision'] == 'BUY']),
                'sell_decisions': len(decisions_df[decisions_df['decision'] == 'SELL']),
                'hold_decisions': len(decisions_df[decisions_df['decision'] == 'HOLD']),
                'avg_confidence': decisions_df['confidence'].mean(),
                'avg_risk_score': decisions_df['risk_score'].mean(),
                'avg_expected_return': decisions_df['expected_return'].mean(),
                'avg_position_size': decisions_df['position_size'].mean()
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {}

def main():
    """Test the Trade Decision Agent"""
    # Create sample signals
    signals = [
        TradingSignal(
            source='technical_analysis',
            symbol='AAPL',
            action='BUY',
            confidence=0.8,
            price=150.0,
            timestamp=datetime.now(),
            reasoning='Strong uptrend',
            features={'rsi': 30, 'macd': 0.5},
            metadata={}
        ),
        TradingSignal(
            source='sentiment_analysis',
            symbol='AAPL',
            action='BUY',
            confidence=0.7,
            price=150.0,
            timestamp=datetime.now(),
            reasoning='Positive sentiment',
            features={'sentiment_score': 0.6},
            metadata={}
        )
    ]
    
    # Create market context
    market_context = MarketContext(
        market_regime='BULL',
        volatility=0.2,
        trend_strength=0.8,
        sentiment_score=0.3,
        economic_indicators={'gdp_growth': 0.03, 'inflation': 0.02},
        sector_performance={'technology': 0.05},
        correlation_risk=0.4
    )
    
    # Initialize agent
    agent = TradeDecisionAgent()
    
    # Make decisions
    decisions = agent.make_decision(signals, market_context)
    
    print(f"Generated {len(decisions)} decisions:")
    for decision in decisions:
        print(f"Symbol: {decision.symbol}")
        print(f"Decision: {decision.decision.value}")
        print(f"Confidence: {decision.confidence:.2f}")
        print(f"Risk Score: {decision.risk_score:.2f}")
        print(f"Position Size: {decision.position_size:.2%}")
        print(f"Reasoning: {decision.reasoning}")
        print("-" * 50)

if __name__ == "__main__":
    main()
