#!/usr/bin/env python3
"""
AI Signal Filter - Advanced filtering and validation of trading signals
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
import json
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class SignalQuality(Enum):
    EXCELLENT = 5
    GOOD = 4
    AVERAGE = 3
    POOR = 2
    VERY_POOR = 1

class FilterResult(Enum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    MODIFY = "MODIFY"
    HOLD = "HOLD"

@dataclass
class TradingSignal:
    """Trading signal data structure"""
    id: str
    source: str
    symbol: str
    action: str
    confidence: float
    price: float
    timestamp: datetime
    reasoning: str
    features: Dict[str, float]
    metadata: Dict[str, Any]
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size: Optional[float] = None

@dataclass
class FilteredSignal:
    """Filtered signal with quality assessment"""
    original_signal: TradingSignal
    filter_result: FilterResult
    quality_score: float
    quality_grade: SignalQuality
    confidence_adjustment: float
    reasoning: str
    modifications: Dict[str, Any]
    risk_score: float
    expected_performance: float

class AISignalFilter:
    """Advanced AI-powered signal filtering system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Filter models
        self.quality_model = None
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.performance_predictor = None
        
        # Feature processing
        self.scaler = StandardScaler()
        self.feature_columns = []
        
        # Filter criteria
        self.min_confidence = self.config.get('min_confidence', 0.6)
        self.max_risk_score = self.config.get('max_risk_score', 0.7)
        self.min_quality_score = self.config.get('min_quality_score', 0.5)
        
        # Historical data for validation
        self.signal_history = []
        self.performance_history = []
        self.market_data_cache = {}
        
        # Filter statistics
        self.filter_stats = {
            'total_signals': 0,
            'accepted_signals': 0,
            'rejected_signals': 0,
            'modified_signals': 0,
            'avg_quality_score': 0.0
        }
        
        # Initialize models
        self._initialize_models()
        
        logger.info("AI Signal Filter initialized")
    
    def _initialize_models(self):
        """Initialize filtering models"""
        try:
            # Quality assessment model
            self.quality_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
            # Performance prediction model
            self.performance_predictor = RandomForestClassifier(
                n_estimators=150,
                max_depth=12,
                random_state=42
            )
            
            logger.info("Filter models initialized")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise
    
    def filter_signals(self, signals: List[TradingSignal],
                      market_context: Dict[str, Any] = None) -> List[FilteredSignal]:
        """Filter and validate trading signals"""
        try:
            filtered_signals = []
            
            for signal in signals:
                try:
                    # Apply comprehensive filtering
                    filtered_signal = self._filter_single_signal(signal, market_context)
                    
                    if filtered_signal:
                        filtered_signals.append(filtered_signal)
                        
                    # Update statistics
                    self._update_filter_stats(filtered_signal)
                    
                except Exception as e:
                    logger.error(f"Error filtering signal {signal.id}: {e}")
                    continue
            
            # Apply portfolio-level filters
            filtered_signals = self._apply_portfolio_filters(filtered_signals)
            
            # Sort by quality score
            filtered_signals.sort(key=lambda x: x.quality_score, reverse=True)
            
            return filtered_signals
            
        except Exception as e:
            logger.error(f"Error filtering signals: {e}")
            return []
    
    def _filter_single_signal(self, signal: TradingSignal,
                            market_context: Dict[str, Any] = None) -> Optional[FilteredSignal]:
        """Filter a single trading signal"""
        try:
            # Extract features for filtering
            features = self._extract_signal_features(signal, market_context)
            
            if not features:
                return None
            
            # Basic validation checks
            if not self._basic_validation(signal):
                return FilteredSignal(
                    original_signal=signal,
                    filter_result=FilterResult.REJECT,
                    quality_score=0.0,
                    quality_grade=SignalQuality.VERY_POOR,
                    confidence_adjustment=0.0,
                    reasoning="Failed basic validation",
                    modifications={},
                    risk_score=1.0,
                    expected_performance=0.0
                )
            
            # Quality assessment
            quality_score = self._assess_signal_quality(features, signal)
            quality_grade = self._get_quality_grade(quality_score)
            
            # Risk assessment
            risk_score = self._assess_signal_risk(features, signal)
            
            # Performance prediction
            expected_performance = self._predict_signal_performance(features, signal)
            
            # Anomaly detection
            is_anomaly = self._detect_signal_anomaly(features)
            
            # Market context validation
            context_score = self._validate_market_context(signal, market_context)
            
            # Source credibility check
            source_credibility = self._assess_source_credibility(signal.source)
            
            # Technical validation
            technical_score = self._validate_technical_aspects(signal)
            
            # Determine filter result
            filter_result, reasoning, modifications = self._determine_filter_result(
                signal, quality_score, risk_score, expected_performance,
                is_anomaly, context_score, source_credibility, technical_score
            )
            
            # Calculate confidence adjustment
            confidence_adjustment = self._calculate_confidence_adjustment(
                quality_score, risk_score, context_score, source_credibility
            )
            
            return FilteredSignal(
                original_signal=signal,
                filter_result=filter_result,
                quality_score=quality_score,
                quality_grade=quality_grade,
                confidence_adjustment=confidence_adjustment,
                reasoning=reasoning,
                modifications=modifications,
                risk_score=risk_score,
                expected_performance=expected_performance
            )
            
        except Exception as e:
            logger.error(f"Error filtering single signal: {e}")
            return None
    
    def _extract_signal_features(self, signal: TradingSignal,
                               market_context: Dict[str, Any] = None) -> Optional[Dict[str, float]]:
        """Extract features for signal filtering"""
        try:
            features = {}
            
            # Basic signal features
            features['confidence'] = signal.confidence
            features['price'] = signal.price
            features['has_stop_loss'] = 1.0 if signal.stop_loss else 0.0
            features['has_take_profit'] = 1.0 if signal.take_profit else 0.0
            features['has_position_size'] = 1.0 if signal.position_size else 0.0
            
            # Action encoding
            features['action_buy'] = 1.0 if signal.action == 'BUY' else 0.0
            features['action_sell'] = 1.0 if signal.action == 'SELL' else 0.0
            features['action_hold'] = 1.0 if signal.action == 'HOLD' else 0.0
            
            # Source features
            features['source_technical'] = 1.0 if 'technical' in signal.source.lower() else 0.0
            features['source_fundamental'] = 1.0 if 'fundamental' in signal.source.lower() else 0.0
            features['source_sentiment'] = 1.0 if 'sentiment' in signal.source.lower() else 0.0
            features['source_ai'] = 1.0 if 'ai' in signal.source.lower() else 0.0
            
            # Time-based features
            now = datetime.now()
            signal_age = (now - signal.timestamp).total_seconds() / 3600  # Hours
            features['signal_age_hours'] = signal_age
            features['is_recent'] = 1.0 if signal_age < 1 else 0.0
            features['hour_of_day'] = signal.timestamp.hour
            features['day_of_week'] = signal.timestamp.weekday()
            features['is_market_hours'] = 1.0 if 9 <= signal.timestamp.hour <= 16 else 0.0
            
            # Signal-specific features
            if signal.features:
                for key, value in signal.features.items():
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        features[f'signal_{key}'] = float(value)
            
            # Market context features
            if market_context:
                features['market_volatility'] = market_context.get('volatility', 0.2)
                features['market_trend'] = market_context.get('trend_strength', 0.0)
                features['market_sentiment'] = market_context.get('sentiment_score', 0.0)
                
                # Market regime
                regime = market_context.get('market_regime', 'SIDEWAYS')
                features['regime_bull'] = 1.0 if regime == 'BULL' else 0.0
                features['regime_bear'] = 1.0 if regime == 'BEAR' else 0.0
                features['regime_sideways'] = 1.0 if regime == 'SIDEWAYS' else 0.0
                features['regime_volatile'] = 1.0 if regime == 'VOLATILE' else 0.0
            
            # Historical performance features
            source_performance = self._get_source_performance(signal.source)
            features['source_win_rate'] = source_performance.get('win_rate', 0.5)
            features['source_avg_return'] = source_performance.get('avg_return', 0.0)
            features['source_signal_count'] = source_performance.get('signal_count', 0)
            
            # Symbol-specific features
            symbol_stats = self._get_symbol_statistics(signal.symbol)
            features['symbol_volatility'] = symbol_stats.get('volatility', 0.2)
            features['symbol_liquidity'] = symbol_stats.get('liquidity_score', 0.5)
            features['symbol_trend'] = symbol_stats.get('trend_strength', 0.0)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting signal features: {e}")
            return None
    
    def _basic_validation(self, signal: TradingSignal) -> bool:
        """Perform basic signal validation"""
        try:
            # Check required fields
            if not signal.symbol or not signal.action or not signal.source:
                return False
            
            # Check confidence range
            if not (0.0 <= signal.confidence <= 1.0):
                return False
            
            # Check price validity
            if signal.price <= 0:
                return False
            
            # Check action validity
            if signal.action not in ['BUY', 'SELL', 'HOLD']:
                return False
            
            # Check timestamp validity
            if signal.timestamp > datetime.now() + timedelta(minutes=5):
                return False
            
            # Check signal age (not too old)
            signal_age = (datetime.now() - signal.timestamp).total_seconds() / 3600
            if signal_age > 24:  # Older than 24 hours
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error in basic validation: {e}")
            return False
    
    def _assess_signal_quality(self, features: Dict[str, float], 
                             signal: TradingSignal) -> float:
        """Assess signal quality score"""
        try:
            quality_score = 0.0
            
            # Confidence component (30%)
            confidence_score = signal.confidence
            quality_score += confidence_score * 0.3
            
            # Completeness component (20%)
            completeness_score = 0.0
            if signal.stop_loss:
                completeness_score += 0.3
            if signal.take_profit:
                completeness_score += 0.3
            if signal.reasoning:
                completeness_score += 0.2
            if signal.features:
                completeness_score += 0.2
            
            quality_score += completeness_score * 0.2
            
            # Source credibility component (20%)
            source_credibility = self._assess_source_credibility(signal.source)
            quality_score += source_credibility * 0.2
            
            # Technical validity component (15%)
            technical_score = self._validate_technical_aspects(signal)
            quality_score += technical_score * 0.15
            
            # Timeliness component (15%)
            signal_age = features.get('signal_age_hours', 24)
            timeliness_score = max(0.0, 1.0 - signal_age / 24)  # Decay over 24 hours
            quality_score += timeliness_score * 0.15
            
            return max(0.0, min(1.0, quality_score))
            
        except Exception as e:
            logger.error(f"Error assessing signal quality: {e}")
            return 0.5
    
    def _assess_signal_risk(self, features: Dict[str, float], 
                          signal: TradingSignal) -> float:
        """Assess signal risk score"""
        try:
            risk_factors = []
            
            # Volatility risk
            symbol_volatility = features.get('symbol_volatility', 0.2)
            market_volatility = features.get('market_volatility', 0.2)
            avg_volatility = (symbol_volatility + market_volatility) / 2
            risk_factors.append(min(1.0, avg_volatility / 0.5))
            
            # Confidence risk (inverse)
            confidence_risk = 1.0 - signal.confidence
            risk_factors.append(confidence_risk)
            
            # Market regime risk
            if features.get('regime_volatile', 0) == 1.0:
                risk_factors.append(0.8)
            elif features.get('regime_bear', 0) == 1.0:
                risk_factors.append(0.6)
            else:
                risk_factors.append(0.3)
            
            # Time-based risk
            if features.get('is_market_hours', 0) == 0.0:
                risk_factors.append(0.7)  # Higher risk outside market hours
            else:
                risk_factors.append(0.3)
            
            # Source reliability risk
            source_win_rate = features.get('source_win_rate', 0.5)
            source_risk = 1.0 - source_win_rate
            risk_factors.append(source_risk)
            
            # Liquidity risk
            liquidity_score = features.get('symbol_liquidity', 0.5)
            liquidity_risk = 1.0 - liquidity_score
            risk_factors.append(liquidity_risk)
            
            # Calculate weighted average
            risk_score = np.mean(risk_factors)
            
            return max(0.0, min(1.0, risk_score))
            
        except Exception as e:
            logger.error(f"Error assessing signal risk: {e}")
            return 0.5
    
    def _predict_signal_performance(self, features: Dict[str, float], 
                                  signal: TradingSignal) -> float:
        """Predict expected signal performance"""
        try:
            # Use historical performance if available
            source_performance = self._get_source_performance(signal.source)
            base_performance = source_performance.get('avg_return', 0.0)
            
            # Adjust based on confidence
            confidence_adjustment = (signal.confidence - 0.5) * 0.1
            
            # Adjust based on market conditions
            market_adjustment = 0.0
            if 'regime_bull' in features and features['regime_bull'] == 1.0:
                market_adjustment = 0.02
            elif 'regime_bear' in features and features['regime_bear'] == 1.0:
                market_adjustment = -0.02
            
            # Adjust based on volatility
            volatility = features.get('market_volatility', 0.2)
            volatility_adjustment = -volatility * 0.1  # Higher vol = lower expected return
            
            expected_performance = (base_performance + confidence_adjustment + 
                                  market_adjustment + volatility_adjustment)
            
            return expected_performance
            
        except Exception as e:
            logger.error(f"Error predicting signal performance: {e}")
            return 0.0
    
    def _detect_signal_anomaly(self, features: Dict[str, float]) -> bool:
        """Detect if signal is anomalous"""
        try:
            # Convert features to array
            feature_array = self._features_to_array(features)
            
            if len(feature_array) == 0:
                return False
            
            # Use anomaly detector if trained
            if hasattr(self.anomaly_detector, 'decision_function'):
                try:
                    anomaly_score = self.anomaly_detector.decision_function([feature_array])[0]
                    return anomaly_score < -0.5  # Threshold for anomaly
                except:
                    pass
            
            # Fallback to statistical anomaly detection
            return self._statistical_anomaly_detection(features)
            
        except Exception as e:
            logger.error(f"Error detecting signal anomaly: {e}")
            return False
    
    def _statistical_anomaly_detection(self, features: Dict[str, float]) -> bool:
        """Statistical anomaly detection"""
        try:
            anomaly_indicators = 0
            
            # Check for extreme confidence values
            confidence = features.get('confidence', 0.5)
            if confidence > 0.95 or confidence < 0.1:
                anomaly_indicators += 1
            
            # Check for extreme volatility
            volatility = features.get('market_volatility', 0.2)
            if volatility > 0.6 or volatility < 0.05:
                anomaly_indicators += 1
            
            # Check for unusual timing
            hour = features.get('hour_of_day', 12)
            if hour < 6 or hour > 20:  # Very early or very late
                anomaly_indicators += 1
            
            # Check signal age
            signal_age = features.get('signal_age_hours', 1)
            if signal_age > 12:  # Very old signal
                anomaly_indicators += 1
            
            return anomaly_indicators >= 2
            
        except Exception as e:
            logger.error(f"Error in statistical anomaly detection: {e}")
            return False
    
    def _validate_market_context(self, signal: TradingSignal, 
                               market_context: Dict[str, Any] = None) -> float:
        """Validate signal against market context"""
        try:
            if not market_context:
                return 0.5  # Neutral score
            
            context_score = 0.5
            
            # Check signal alignment with market regime
            market_regime = market_context.get('market_regime', 'SIDEWAYS')
            
            if signal.action == 'BUY':
                if market_regime in ['BULL', 'SIDEWAYS']:
                    context_score += 0.2
                elif market_regime == 'BEAR':
                    context_score -= 0.3
            elif signal.action == 'SELL':
                if market_regime in ['BEAR', 'VOLATILE']:
                    context_score += 0.2
                elif market_regime == 'BULL':
                    context_score -= 0.3
            
            # Check volatility alignment
            market_volatility = market_context.get('volatility', 0.2)
            if market_volatility > 0.4 and signal.confidence > 0.8:
                context_score -= 0.1  # High confidence in high vol environment
            
            # Check sentiment alignment
            sentiment = market_context.get('sentiment_score', 0.0)
            if signal.action == 'BUY' and sentiment > 0.2:
                context_score += 0.1
            elif signal.action == 'SELL' and sentiment < -0.2:
                context_score += 0.1
            
            return max(0.0, min(1.0, context_score))
            
        except Exception as e:
            logger.error(f"Error validating market context: {e}")
            return 0.5
    
    def _assess_source_credibility(self, source: str) -> float:
        """Assess credibility of signal source"""
        try:
            # Get historical performance of source
            source_performance = self._get_source_performance(source)
            
            if not source_performance:
                return 0.5  # Default credibility
            
            # Base credibility on win rate and signal count
            win_rate = source_performance.get('win_rate', 0.5)
            signal_count = source_performance.get('signal_count', 0)
            avg_return = source_performance.get('avg_return', 0.0)
            
            # Calculate credibility score
            credibility = win_rate * 0.6  # Win rate is most important
            
            # Adjust for experience (signal count)
            if signal_count > 100:
                credibility += 0.2
            elif signal_count > 50:
                credibility += 0.1
            elif signal_count < 10:
                credibility -= 0.1
            
            # Adjust for average return
            if avg_return > 0.05:  # 5% average return
                credibility += 0.1
            elif avg_return < -0.02:  # -2% average return
                credibility -= 0.2
            
            return max(0.0, min(1.0, credibility))
            
        except Exception as e:
            logger.error(f"Error assessing source credibility: {e}")
            return 0.5
    
    def _validate_technical_aspects(self, signal: TradingSignal) -> float:
        """Validate technical aspects of the signal"""
        try:
            technical_score = 0.5
            
            # Check stop loss and take profit levels
            if signal.stop_loss and signal.take_profit:
                # Calculate risk-reward ratio
                if signal.action == 'BUY':
                    risk = signal.price - signal.stop_loss
                    reward = signal.take_profit - signal.price
                else:  # SELL
                    risk = signal.stop_loss - signal.price
                    reward = signal.price - signal.take_profit
                
                if risk > 0 and reward > 0:
                    risk_reward_ratio = reward / risk
                    if 1.5 <= risk_reward_ratio <= 4.0:  # Good risk-reward
                        technical_score += 0.3
                    elif risk_reward_ratio > 4.0:  # Too good to be true
                        technical_score -= 0.1
                    else:  # Poor risk-reward
                        technical_score -= 0.2
            
            # Check position sizing
            if signal.position_size:
                if 0.01 <= signal.position_size <= 0.1:  # Reasonable position size
                    technical_score += 0.1
                elif signal.position_size > 0.2:  # Too large
                    technical_score -= 0.2
            
            # Check reasoning quality
            if signal.reasoning:
                reasoning_length = len(signal.reasoning)
                if 20 <= reasoning_length <= 200:  # Good explanation
                    technical_score += 0.1
                elif reasoning_length < 10:  # Too brief
                    technical_score -= 0.1
            
            return max(0.0, min(1.0, technical_score))
            
        except Exception as e:
            logger.error(f"Error validating technical aspects: {e}")
            return 0.5
    
    def _determine_filter_result(self, signal: TradingSignal, quality_score: float,
                               risk_score: float, expected_performance: float,
                               is_anomaly: bool, context_score: float,
                               source_credibility: float, technical_score: float) -> Tuple[FilterResult, str, Dict[str, Any]]:
        """Determine the filter result for a signal"""
        try:
            modifications = {}
            
            # Rejection criteria
            if is_anomaly:
                return FilterResult.REJECT, "Signal detected as anomaly", modifications
            
            if quality_score < self.min_quality_score:
                return FilterResult.REJECT, f"Quality score too low: {quality_score:.2f}", modifications
            
            if risk_score > self.max_risk_score:
                return FilterResult.REJECT, f"Risk score too high: {risk_score:.2f}", modifications
            
            if signal.confidence < self.min_confidence:
                return FilterResult.REJECT, f"Confidence too low: {signal.confidence:.2f}", modifications
            
            if source_credibility < 0.3:
                return FilterResult.REJECT, f"Source credibility too low: {source_credibility:.2f}", modifications
            
            if expected_performance < -0.05:  # Expected loss > 5%
                return FilterResult.REJECT, f"Expected performance too poor: {expected_performance:.2%}", modifications
            
            # Modification criteria
            needs_modification = False
            reasoning_parts = []
            
            # Adjust confidence if needed
            if context_score < 0.4:
                confidence_adjustment = 0.8  # Reduce confidence
                modifications['confidence_multiplier'] = confidence_adjustment
                needs_modification = True
                reasoning_parts.append("Reduced confidence due to poor market context")
            
            # Adjust position size if needed
            if risk_score > 0.5:
                position_adjustment = 0.7  # Reduce position size
                modifications['position_size_multiplier'] = position_adjustment
                needs_modification = True
                reasoning_parts.append("Reduced position size due to high risk")
            
            # Adjust stop loss if needed
            if technical_score < 0.4 and signal.stop_loss:
                stop_loss_adjustment = 1.2  # Tighter stop loss
                modifications['stop_loss_multiplier'] = stop_loss_adjustment
                needs_modification = True
                reasoning_parts.append("Tightened stop loss due to technical concerns")
            
            if needs_modification:
                reasoning = "Signal modified: " + "; ".join(reasoning_parts)
                return FilterResult.MODIFY, reasoning, modifications
            
            # Accept criteria
            if (quality_score >= 0.7 and risk_score <= 0.4 and 
                expected_performance >= 0.02 and source_credibility >= 0.6):
                return FilterResult.ACCEPT, "High quality signal accepted", modifications
            
            # Default accept for signals that pass basic criteria
            return FilterResult.ACCEPT, "Signal accepted", modifications
            
        except Exception as e:
            logger.error(f"Error determining filter result: {e}")
            return FilterResult.REJECT, f"Error in filtering: {e}", {}
    
    def _calculate_confidence_adjustment(self, quality_score: float, risk_score: float,
                                       context_score: float, source_credibility: float) -> float:
        """Calculate confidence adjustment factor"""
        try:
            # Base adjustment on quality score
            adjustment = quality_score
            
            # Adjust for risk
            adjustment *= (1.0 - risk_score * 0.5)
            
            # Adjust for context
            adjustment *= (0.5 + context_score * 0.5)
            
            # Adjust for source credibility
            adjustment *= (0.5 + source_credibility * 0.5)
            
            return max(0.1, min(1.5, adjustment))
            
        except Exception as e:
            logger.error(f"Error calculating confidence adjustment: {e}")
            return 1.0
    
    def _apply_portfolio_filters(self, filtered_signals: List[FilteredSignal]) -> List[FilteredSignal]:
        """Apply portfolio-level filters"""
        try:
            if not filtered_signals:
                return filtered_signals
            
            # Remove duplicate signals for same symbol
            symbol_signals = {}
            for signal in filtered_signals:
                symbol = signal.original_signal.symbol
                if (symbol not in symbol_signals or 
                    signal.quality_score > symbol_signals[symbol].quality_score):
                    symbol_signals[symbol] = signal
            
            # Limit number of signals per action
            buy_signals = []
            sell_signals = []
            
            for signal in symbol_signals.values():
                if signal.original_signal.action == 'BUY':
                    buy_signals.append(signal)
                elif signal.original_signal.action == 'SELL':
                    sell_signals.append(signal)
            
            # Sort by quality and limit
            max_signals_per_action = self.config.get('max_signals_per_action', 5)
            
            buy_signals.sort(key=lambda x: x.quality_score, reverse=True)
            sell_signals.sort(key=lambda x: x.quality_score, reverse=True)
            
            final_signals = (buy_signals[:max_signals_per_action] + 
                           sell_signals[:max_signals_per_action])
            
            return final_signals
            
        except Exception as e:
            logger.error(f"Error applying portfolio filters: {e}")
            return filtered_signals
    
    def _get_quality_grade(self, quality_score: float) -> SignalQuality:
        """Convert quality score to grade"""
        if quality_score >= 0.9:
            return SignalQuality.EXCELLENT
        elif quality_score >= 0.75:
            return SignalQuality.GOOD
        elif quality_score >= 0.6:
            return SignalQuality.AVERAGE
        elif quality_score >= 0.4:
            return SignalQuality.POOR
        else:
            return SignalQuality.VERY_POOR
    
    def _get_source_performance(self, source: str) -> Dict[str, Any]:
        """Get historical performance of a signal source"""
        try:
            # Filter performance history by source
            source_history = [p for p in self.performance_history if p.get('source') == source]
            
            if not source_history:
                return {'win_rate': 0.5, 'avg_return': 0.0, 'signal_count': 0}
            
            # Calculate performance metrics
            total_signals = len(source_history)
            winning_signals = len([p for p in source_history if p.get('return', 0) > 0])
            win_rate = winning_signals / total_signals if total_signals > 0 else 0.5
            
            returns = [p.get('return', 0) for p in source_history]
            avg_return = np.mean(returns) if returns else 0.0
            
            return {
                'win_rate': win_rate,
                'avg_return': avg_return,
                'signal_count': total_signals
            }
            
        except Exception as e:
            logger.error(f"Error getting source performance: {e}")
            return {'win_rate': 0.5, 'avg_return': 0.0, 'signal_count': 0}
    
    def _get_symbol_statistics(self, symbol: str) -> Dict[str, Any]:
        """Get statistics for a symbol"""
        try:
            # Try to get from cache first
            if symbol in self.market_data_cache:
                cache_time = self.market_data_cache[symbol].get('timestamp', datetime.min)
                if (datetime.now() - cache_time).total_seconds() < 3600:  # 1 hour cache
                    return self.market_data_cache[symbol]['stats']
            
            # Fetch fresh data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='30d')
            
            if data.empty:
                return {'volatility': 0.2, 'liquidity_score': 0.5, 'trend_strength': 0.0}
            
            # Calculate statistics
            returns = data['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized
            
            # Liquidity score based on volume
            avg_volume = data['Volume'].mean()
            liquidity_score = min(1.0, avg_volume / 1000000)  # Normalize to millions
            
            # Trend strength
            price_change = (data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]
            trend_strength = np.tanh(price_change * 10)  # Normalize between -1 and 1
            
            stats = {
                'volatility': volatility,
                'liquidity_score': liquidity_score,
                'trend_strength': trend_strength
            }
            
            # Cache the results
            self.market_data_cache[symbol] = {
                'stats': stats,
                'timestamp': datetime.now()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting symbol statistics for {symbol}: {e}")
            return {'volatility': 0.2, 'liquidity_score': 0.5, 'trend_strength': 0.0}
    
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
            return np.array([])
    
    def _update_filter_stats(self, filtered_signal: Optional[FilteredSignal]):
        """Update filter statistics"""
        try:
            self.filter_stats['total_signals'] += 1
            
            if filtered_signal:
                if filtered_signal.filter_result == FilterResult.ACCEPT:
                    self.filter_stats['accepted_signals'] += 1
                elif filtered_signal.filter_result == FilterResult.REJECT:
                    self.filter_stats['rejected_signals'] += 1
                elif filtered_signal.filter_result == FilterResult.MODIFY:
                    self.filter_stats['modified_signals'] += 1
                
                # Update average quality score
                total_quality = (self.filter_stats['avg_quality_score'] * 
                               (self.filter_stats['total_signals'] - 1) + 
                               filtered_signal.quality_score)
                self.filter_stats['avg_quality_score'] = total_quality / self.filter_stats['total_signals']
            
        except Exception as e:
            logger.error(f"Error updating filter stats: {e}")
    
    def record_signal_performance(self, signal_id: str, performance_data: Dict[str, Any]):
        """Record performance of a filtered signal"""
        try:
            # Find the original signal
            original_signal = None
            for signal in self.signal_history:
                if signal.get('id') == signal_id:
                    original_signal = signal
                    break
            
            if original_signal:
                performance_record = {
                    'signal_id': signal_id,
                    'source': original_signal.get('source'),
                    'symbol': original_signal.get('symbol'),
                    'action': original_signal.get('action'),
                    'return': performance_data.get('return', 0.0),
                    'duration': performance_data.get('duration', 0),
                    'max_favorable_excursion': performance_data.get('mfe', 0.0),
                    'max_adverse_excursion': performance_data.get('mae', 0.0),
                    'timestamp': datetime.now()
                }
                
                self.performance_history.append(performance_record)
                
                # Keep only recent performance data
                if len(self.performance_history) > 1000:
                    self.performance_history = self.performance_history[-500:]
                
                logger.info(f"Recorded performance for signal {signal_id}: {performance_data.get('return', 0.0):.2%}")
            
        except Exception as e:
            logger.error(f"Error recording signal performance: {e}")
    
    def get_filter_statistics(self) -> Dict[str, Any]:
        """Get filter performance statistics"""
        try:
            stats = self.filter_stats.copy()
            
            # Calculate acceptance rate
            if stats['total_signals'] > 0:
                stats['acceptance_rate'] = stats['accepted_signals'] / stats['total_signals']
                stats['rejection_rate'] = stats['rejected_signals'] / stats['total_signals']
                stats['modification_rate'] = stats['modified_signals'] / stats['total_signals']
            else:
                stats['acceptance_rate'] = 0.0
                stats['rejection_rate'] = 0.0
                stats['modification_rate'] = 0.0
            
            # Add performance metrics if available
            if self.performance_history:
                returns = [p['return'] for p in self.performance_history]
                stats['avg_signal_return'] = np.mean(returns)
                stats['signal_win_rate'] = len([r for r in returns if r > 0]) / len(returns)
                stats['total_performance_records'] = len(self.performance_history)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting filter statistics: {e}")
            return self.filter_stats.copy()
    
    def train_filter_models(self, training_data: pd.DataFrame):
        """Train filter models with historical data"""
        try:
            logger.info("Training filter models...")
            
            if training_data.empty:
                logger.warning("No training data provided")
                return
            
            # Prepare features and targets
            feature_cols = [col for col in training_data.columns 
                          if col not in ['quality_score', 'performance', 'signal_id', 'timestamp']]
            
            X = training_data[feature_cols].fillna(0)
            
            # Train quality model
            if 'quality_score' in training_data.columns:
                y_quality = (training_data['quality_score'] > 0.7).astype(int)
                self.quality_model.fit(X, y_quality)
                logger.info("Quality model trained")
            
            # Train performance predictor
            if 'performance' in training_data.columns:
                y_performance = (training_data['performance'] > 0.02).astype(int)
                self.performance_predictor.fit(X, y_performance)
                logger.info("Performance predictor trained")
            
            # Train anomaly detector
            self.anomaly_detector.fit(X)
            logger.info("Anomaly detector trained")
            
            # Store feature columns
            self.feature_columns = feature_cols
            
            # Fit scaler
            self.scaler.fit(X)
            
            logger.info("All filter models trained successfully")
            
        except Exception as e:
            logger.error(f"Error training filter models: {e}")
            raise

def main():
    """Test the AI Signal Filter"""
    # Create sample signals
    signals = [
        TradingSignal(
            id="signal_1",
            source="technical_analysis",
            symbol="AAPL",
            action="BUY",
            confidence=0.8,
            price=150.0,
            timestamp=datetime.now(),
            reasoning="Strong uptrend with RSI oversold",
            features={"rsi": 30, "macd": 0.5, "volume_ratio": 1.5},
            metadata={},
            stop_loss=145.0,
            take_profit=160.0
        ),
        TradingSignal(
            id="signal_2",
            source="sentiment_analysis",
            symbol="GOOGL",
            action="SELL",
            confidence=0.9,
            price=2500.0,
            timestamp=datetime.now() - timedelta(hours=2),
            reasoning="Negative sentiment spike",
            features={"sentiment_score": -0.7},
            metadata={}
        ),
        TradingSignal(
            id="signal_3",
            source="unknown_source",
            symbol="TSLA",
            action="BUY",
            confidence=0.95,  # Suspiciously high
            price=800.0,
            timestamp=datetime.now() - timedelta(hours=10),  # Old signal
            reasoning="",
            features={},
            metadata={}
        )
    ]
    
    # Create market context
    market_context = {
        'market_regime': 'BULL',
        'volatility': 0.25,
        'sentiment_score': 0.1,
        'trend_strength': 0.6
    }
    
    # Initialize filter
    signal_filter = AISignalFilter()
    
    # Filter signals
    filtered_signals = signal_filter.filter_signals(signals, market_context)
    
    print(f"Filtered {len(signals)} signals, {len(filtered_signals)} passed filter")
    print("\nFiltered Signals:")
    print("-" * 80)
    
    for filtered_signal in filtered_signals:
        signal = filtered_signal.original_signal
        print(f"Signal ID: {signal.id}")
        print(f"Symbol: {signal.symbol} | Action: {signal.action}")
        print(f"Original Confidence: {signal.confidence:.2f}")
        print(f"Filter Result: {filtered_signal.filter_result.value}")
        print(f"Quality Score: {filtered_signal.quality_score:.2f} ({filtered_signal.quality_grade.name})")
        print(f"Risk Score: {filtered_signal.risk_score:.2f}")
        print(f"Confidence Adjustment: {filtered_signal.confidence_adjustment:.2f}")
        print(f"Expected Performance: {filtered_signal.expected_performance:.2%}")
        print(f"Reasoning: {filtered_signal.reasoning}")
        if filtered_signal.modifications:
            print(f"Modifications: {filtered_signal.modifications}")
        print("-" * 80)
    
    # Get filter statistics
    stats = signal_filter.get_filter_statistics()
    print(f"\nFilter Statistics:")
    print(f"Total Signals: {stats['total_signals']}")
    print(f"Accepted: {stats['accepted_signals']} ({stats['acceptance_rate']:.1%})")
    print(f"Rejected: {stats['rejected_signals']} ({stats['rejection_rate']:.1%})")
    print(f"Modified: {stats['modified_signals']} ({stats['modification_rate']:.1%})")
    print(f"Average Quality Score: {stats['avg_quality_score']:.2f}")

if __name__ == "__main__":
    main()
