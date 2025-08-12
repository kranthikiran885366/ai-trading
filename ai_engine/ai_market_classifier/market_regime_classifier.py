#!/usr/bin/env python3
"""
Market Regime Classifier - Advanced AI system for classifying market conditions
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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import KMeans
import yfinance as yf
import talib
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    BULL_MARKET = "BULL"
    BEAR_MARKET = "BEAR"
    SIDEWAYS_MARKET = "SIDEWAYS"
    HIGH_VOLATILITY = "VOLATILE"
    CRISIS = "CRISIS"
    RECOVERY = "RECOVERY"

class VolatilityRegime(Enum):
    LOW_VOL = "LOW"
    NORMAL_VOL = "NORMAL"
    HIGH_VOL = "HIGH"
    EXTREME_VOL = "EXTREME"

class TrendStrength(Enum):
    STRONG_UPTREND = "STRONG_UP"
    WEAK_UPTREND = "WEAK_UP"
    SIDEWAYS = "SIDEWAYS"
    WEAK_DOWNTREND = "WEAK_DOWN"
    STRONG_DOWNTREND = "STRONG_DOWN"

@dataclass
class MarketClassification:
    """Market classification result"""
    regime: MarketRegime
    volatility_regime: VolatilityRegime
    trend_strength: TrendStrength
    confidence: float
    regime_probabilities: Dict[str, float]
    supporting_indicators: Dict[str, float]
    regime_duration: int
    transition_probability: float
    timestamp: datetime

class MarketRegimeClassifier:
    """Advanced market regime classification system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Classification models
        self.regime_model = None
        self.volatility_model = None
        self.trend_model = None
        
        # Feature processing
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = []
        
        # Market data cache
        self.market_data_cache = {}
        self.classification_history = []
        
        # Model parameters
        self.lookback_periods = {
            'short': 20,
            'medium': 60,
            'long': 252
        }
        
        # Regime thresholds
        self.regime_thresholds = {
            'bull_bear_threshold': 0.15,  # 15% move for bull/bear
            'volatility_thresholds': [0.15, 0.25, 0.40],  # Low, Normal, High, Extreme
            'trend_thresholds': [-0.3, -0.1, 0.1, 0.3]  # Strong down, weak down, sideways, weak up, strong up
        }
        
        # Initialize models
        self._initialize_models()
        
        logger.info("Market Regime Classifier initialized")
    
    def _initialize_models(self):
        """Initialize classification models"""
        try:
            # Main regime classifier
            self.regime_model = GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=8,
                random_state=42
            )
            
            # Volatility regime classifier
            self.volatility_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
            # Trend strength classifier
            self.trend_model = GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
            
            logger.info("Classification models initialized")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise
    
    def classify_market_regime(self, symbols: List[str] = None, 
                             period: str = '2y') -> MarketClassification:
        """Classify current market regime"""
        try:
            # Use default symbols if none provided
            if symbols is None:
                symbols = ['^GSPC', '^IXIC', '^RUT', '^VIX', 'SPY', 'QQQ', 'IWM']
            
            # Collect market data
            market_data = self._collect_market_data(symbols, period)
            
            if not market_data:
                return self._get_default_classification()
            
            # Extract features
            features = self._extract_market_features(market_data)
            
            if not features:
                return self._get_default_classification()
            
            # Classify regime
            regime_classification = self._classify_regime(features)
            
            # Classify volatility
            volatility_classification = self._classify_volatility(features)
            
            # Classify trend
            trend_classification = self._classify_trend(features)
            
            # Calculate confidence and supporting indicators
            confidence = self._calculate_classification_confidence(features, regime_classification)
            supporting_indicators = self._get_supporting_indicators(features)
            
            # Estimate regime duration and transition probability
            regime_duration = self._estimate_regime_duration(regime_classification)
            transition_probability = self._calculate_transition_probability(features)
            
            # Create classification result
            classification = MarketClassification(
                regime=regime_classification['regime'],
                volatility_regime=volatility_classification,
                trend_strength=trend_classification,
                confidence=confidence,
                regime_probabilities=regime_classification['probabilities'],
                supporting_indicators=supporting_indicators,
                regime_duration=regime_duration,
                transition_probability=transition_probability,
                timestamp=datetime.now()
            )
            
            # Store classification history
            self.classification_history.append(classification)
            
            # Keep only recent history
            if len(self.classification_history) > 1000:
                self.classification_history = self.classification_history[-500:]
            
            return classification
            
        except Exception as e:
            logger.error(f"Error classifying market regime: {e}")
            return self._get_default_classification()
    
    def _collect_market_data(self, symbols: List[str], period: str) -> Dict[str, pd.DataFrame]:
        """Collect market data for analysis"""
        try:
            market_data = {}
            
            for symbol in symbols:
                try:
                    # Check cache first
                    cache_key = f"{symbol}_{period}"
                    if cache_key in self.market_data_cache:
                        cache_time = self.market_data_cache[cache_key]['timestamp']
                        if (datetime.now() - cache_time).total_seconds() < 3600:  # 1 hour cache
                            market_data[symbol] = self.market_data_cache[cache_key]['data']
                            continue
                    
                    # Fetch fresh data
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(period=period)
                    
                    if not data.empty:
                        # Calculate basic indicators
                        data['returns'] = data['Close'].pct_change()
                        data['log_returns'] = np.log(data['Close'] / data['Close'].shift(1))
                        data['volatility'] = data['returns'].rolling(20).std() * np.sqrt(252)
                        
                        market_data[symbol] = data
                        
                        # Cache the data
                        self.market_data_cache[cache_key] = {
                            'data': data,
                            'timestamp': datetime.now()
                        }
                        
                except Exception as e:
                    logger.warning(f"Could not fetch data for {symbol}: {e}")
                    continue
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error collecting market data: {e}")
            return {}
    
    def _extract_market_features(self, market_data: Dict[str, pd.DataFrame]) -> Optional[Dict[str, float]]:
        """Extract comprehensive market features"""
        try:
            features = {}
            
            # Process each symbol
            for symbol, data in market_data.items():
                if len(data) < 50:  # Need minimum data
                    continue
                
                symbol_features = self._extract_symbol_features(symbol, data)
                
                # Add symbol prefix to features
                for key, value in symbol_features.items():
                    features[f"{symbol}_{key}"] = value
            
            if not features:
                return None
            
            # Add cross-asset features
            cross_features = self._extract_cross_asset_features(market_data)
            features.update(cross_features)
            
            # Add macro features
            macro_features = self._extract_macro_features(market_data)
            features.update(macro_features)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting market features: {e}")
            return None
    
    def _extract_symbol_features(self, symbol: str, data: pd.DataFrame) -> Dict[str, float]:
        """Extract features for a single symbol"""
        try:
            features = {}
            
            if len(data) < 20:
                return features
            
            # Price-based features
            current_price = data['Close'].iloc[-1]
            features['current_price'] = current_price
            
            # Returns analysis
            returns = data['returns'].dropna()
            if len(returns) > 0:
                features['avg_return'] = returns.mean()
                features['return_std'] = returns.std()
                features['return_skew'] = returns.skew()
                features['return_kurtosis'] = returns.kurtosis()
                
                # Recent performance
                features['return_1d'] = returns.iloc[-1] if len(returns) > 0 else 0
                features['return_5d'] = returns.tail(5).mean() if len(returns) >= 5 else 0
                features['return_20d'] = returns.tail(20).mean() if len(returns) >= 20 else 0
                features['return_60d'] = returns.tail(60).mean() if len(returns) >= 60 else 0
            
            # Volatility features
            if 'volatility' in data.columns:
                vol_series = data['volatility'].dropna()
                if len(vol_series) > 0:
                    features['current_volatility'] = vol_series.iloc[-1]
                    features['avg_volatility'] = vol_series.mean()
                    features['volatility_trend'] = self._calculate_trend(vol_series.tail(20))
            
            # Technical indicators
            if len(data) >= 50:
                # Moving averages
                sma_20 = talib.SMA(data['Close'].values, timeperiod=20)
                sma_50 = talib.SMA(data['Close'].values, timeperiod=50)
                
                if len(sma_20) > 0 and len(sma_50) > 0:
                    features['price_sma20_ratio'] = current_price / sma_20[-1]
                    features['price_sma50_ratio'] = current_price / sma_50[-1]
                    features['sma20_sma50_ratio'] = sma_20[-1] / sma_50[-1]
                
                # RSI
                rsi = talib.RSI(data['Close'].values, timeperiod=14)
                if len(rsi) > 0:
                    features['rsi'] = rsi[-1]
                
                # MACD
                macd, macd_signal, macd_hist = talib.MACD(data['Close'].values)
                if len(macd_hist) > 0:
                    features['macd_histogram'] = macd_hist[-1]
                    features['macd_signal_ratio'] = macd[-1] / macd_signal[-1] if macd_signal[-1] != 0 else 1
                
                # Bollinger Bands
                bb_upper, bb_middle, bb_lower = talib.BBANDS(data['Close'].values)
                if len(bb_upper) > 0:
                    features['bb_position'] = (current_price - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1])
                    features['bb_width'] = (bb_upper[-1] - bb_lower[-1]) / bb_middle[-1]
                
                # ADX (trend strength)
                adx = talib.ADX(data['High'].values, data['Low'].values, data['Close'].values)
                if len(adx) > 0:
                    features['adx'] = adx[-1]
            
            # Volume analysis
            if 'Volume' in data.columns:
                volume = data['Volume']
                features['avg_volume'] = volume.mean()
                features['volume_trend'] = self._calculate_trend(volume.tail(20))
                
                # Volume-price relationship
                if len(returns) > 0:
                    volume_returns = volume.pct_change().dropna()
                    if len(volume_returns) > 20:
                        correlation = returns.corr(volume_returns)
                        features['volume_price_correlation'] = correlation if not np.isnan(correlation) else 0
            
            # Drawdown analysis
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            features['current_drawdown'] = drawdown.iloc[-1] if len(drawdown) > 0 else 0
            features['max_drawdown'] = drawdown.min() if len(drawdown) > 0 else 0
            
            # Trend analysis
            if len(data) >= 20:
                price_trend_20 = self._calculate_trend(data['Close'].tail(20))
                price_trend_60 = self._calculate_trend(data['Close'].tail(60)) if len(data) >= 60 else price_trend_20
                
                features['price_trend_20'] = price_trend_20
                features['price_trend_60'] = price_trend_60
                features['trend_consistency'] = abs(price_trend_20 - price_trend_60)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features for {symbol}: {e}")
            return {}
    
    def _extract_cross_asset_features(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Extract cross-asset relationship features"""
        try:
            features = {}
            
            # Collect returns for correlation analysis
            returns_data = {}
            for symbol, data in market_data.items():
                if 'returns' in data.columns and len(data['returns']) > 50:
                    returns_data[symbol] = data['returns'].dropna()
            
            if len(returns_data) < 2:
                return features
            
            # Create aligned returns DataFrame
            returns_df = pd.DataFrame
