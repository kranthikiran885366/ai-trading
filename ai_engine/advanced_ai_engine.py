#!/usr/bin/env python3
"""
Advanced AI Engine with 95% Success Rate Target
Multi-model ensemble with advanced prediction capabilities
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
import torch
import torch.nn as nn
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import json

logger = logging.getLogger(__name__)

@dataclass
class TradingSignal:
    symbol: str
    signal: str  # 'buy', 'sell', 'hold'
    confidence: float
    price: float
    target: Optional[float] = None
    stop_loss: Optional[float] = None
    timestamp: str = None
    features: Dict[str, float] = None

class AdvancedAIEngine:
    """Advanced AI Engine with ensemble models for 95% success rate"""
    
    def __init__(self, confidence_threshold: float = 0.85, ensemble_size: int = 5, target_success_rate: float = 0.95):
        self.confidence_threshold = confidence_threshold
        self.ensemble_size = ensemble_size
        self.target_success_rate = target_success_rate
        
        # Model components
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        
        # Performance tracking
        self.model_performance = {}
        self.prediction_history = []
        
        logger.info(f"Advanced AI Engine initialized with {ensemble_size} models")
    
    async def initialize(self):
        """Initialize all AI models"""
        try:
            await self._load_or_create_models()
            logger.info("AI Engine initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing AI Engine: {e}")
            raise
    
    async def _load_or_create_models(self):
        """Load existing models or create new ones"""
        try:
            # Try to load existing models
            self.models['random_forest'] = joblib.load('models/random_forest.pkl')
            self.models['gradient_boost'] = joblib.load('models/gradient_boost.pkl')
            self.models['neural_network'] = joblib.load('models/neural_network.pkl')
            self.scalers['main'] = joblib.load('models/scaler.pkl')
            logger.info("Loaded existing models")
        except FileNotFoundError:
            # Create new models
            await self._create_new_models()
            logger.info("Created new models")
    
    async def _create_new_models(self):
        """Create and train new models"""
        # Random Forest
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        # Gradient Boosting
        self.models['gradient_boost'] = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        # Neural Network
        self.models['neural_network'] = MLPClassifier(
            hidden_layer_sizes=(100, 50, 25),
            activation='relu',
            solver='adam',
            max_iter=1000,
            random_state=42
        )
        
        # Scaler
        self.scalers['main'] = StandardScaler()
        
        # Initialize with dummy data
        dummy_data = np.random.random((100, 20))
        dummy_labels = np.random.randint(0, 3, 100)
        
        self.scalers['main'].fit(dummy_data)
        scaled_data = self.scalers['main'].transform(dummy_data)
        
        for model_name, model in self.models.items():
            model.fit(scaled_data, dummy_labels)
    
    async def generate_signals(self, market_data: Dict, news_sentiment: Dict, portfolio_state: Dict) -> List[TradingSignal]:
        """Generate trading signals using ensemble models"""
        signals = []
        
        try:
            for symbol in market_data.keys():
                # Extract features
                features = await self._extract_features(symbol, market_data, news_sentiment, portfolio_state)
                
                # Generate prediction
                prediction = await self._predict_signal(symbol, features)
                
                if prediction['confidence'] >= self.confidence_threshold:
                    signal = TradingSignal(
                        symbol=symbol,
                        signal=prediction['signal'],
                        confidence=prediction['confidence'],
                        price=market_data[symbol].get('price', 0),
                        target=prediction.get('target'),
                        stop_loss=prediction.get('stop_loss'),
                        timestamp=pd.Timestamp.now().isoformat(),
                        features=features
                    )
                    signals.append(signal)
            
            logger.info(f"Generated {len(signals)} high-confidence signals")
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return []
    
    async def _extract_features(self, symbol: str, market_data: Dict, news_sentiment: Dict, portfolio_state: Dict) -> Dict[str, float]:
        """Extract features for ML models"""
        symbol_data = market_data.get(symbol, {})
        
        features = {
            # Price features
            'price': symbol_data.get('price', 0),
            'volume': symbol_data.get('volume', 0),
            'high': symbol_data.get('high', 0),
            'low': symbol_data.get('low', 0),
            'open': symbol_data.get('open', 0),
            
            # Technical indicators
            'rsi': symbol_data.get('rsi', 50),
            'macd': symbol_data.get('macd', 0),
            'bollinger_upper': symbol_data.get('bollinger_upper', 0),
            'bollinger_lower': symbol_data.get('bollinger_lower', 0),
            'sma_20': symbol_data.get('sma_20', 0),
            'sma_50': symbol_data.get('sma_50', 0),
            'ema_12': symbol_data.get('ema_12', 0),
            'ema_26': symbol_data.get('ema_26', 0),
            
            # Sentiment features
            'news_sentiment': news_sentiment.get(symbol, {}).get('score', 0),
            'news_volume': news_sentiment.get(symbol, {}).get('volume', 0),
            
            # Portfolio features
            'position_size': portfolio_state.get('positions', {}).get(symbol, {}).get('size', 0),
            'portfolio_exposure': portfolio_state.get('total_exposure', 0),
            'cash_ratio': portfolio_state.get('cash_ratio', 1.0),
            
            # Market features
            'market_volatility': symbol_data.get('volatility', 0),
            'correlation_spy': symbol_data.get('correlation_spy', 0),
            'beta': symbol_data.get('beta', 1.0),
        }
        
        return features
    
    async def _predict_signal(self, symbol: str, features: Dict[str, float]) -> Dict[str, Any]:
        """Predict trading signal using ensemble models"""
        try:
            # Convert features to array
            feature_array = np.array(list(features.values())).reshape(1, -1)
            scaled_features = self.scalers['main'].transform(feature_array)
            
            # Get predictions from all models
            predictions = {}
            confidences = {}
            
            for model_name, model in self.models.items():
                pred = model.predict(scaled_features)[0]
                prob = model.predict_proba(scaled_features)[0]
                
                predictions[model_name] = pred
                confidences[model_name] = np.max(prob)
            
            # Ensemble prediction
            signal_votes = {'buy': 0, 'sell': 0, 'hold': 0}
            total_confidence = 0
            
            for model_name, pred in predictions.items():
                signal_map = {0: 'hold', 1: 'buy', 2: 'sell'}
                signal = signal_map.get(pred, 'hold')
                signal_votes[signal] += confidences[model_name]
                total_confidence += confidences[model_name]
            
            # Determine final signal
            final_signal = max(signal_votes, key=signal_votes.get)
            final_confidence = signal_votes[final_signal] / total_confidence if total_confidence > 0 else 0
            
            # Calculate target and stop loss
            current_price = features.get('price', 0)
            target = None
            stop_loss = None
            
            if final_signal == 'buy':
                target = current_price * 1.02  # 2% target
                stop_loss = current_price * 0.99  # 1% stop loss
            elif final_signal == 'sell':
                target = current_price * 0.98  # 2% target
                stop_loss = current_price * 1.01  # 1% stop loss
            
            return {
                'signal': final_signal,
                'confidence': final_confidence,
                'target': target,
                'stop_loss': stop_loss,
                'model_predictions': predictions,
                'model_confidences': confidences
            }
            
        except Exception as e:
            logger.error(f"Error predicting signal for {symbol}: {e}")
            return {
                'signal': 'hold',
                'confidence': 0.0,
                'target': None,
                'stop_loss': None
            }
    
    async def retrain_models(self, training_data: pd.DataFrame):
        """Retrain models with new data"""
        try:
            logger.info("Starting model retraining...")
            
            # Prepare training data
            features = training_data.drop(['target', 'symbol', 'timestamp'], axis=1)
            targets = training_data['target']
            
            # Scale features
            scaled_features = self.scalers['main'].fit_transform(features)
            
            # Retrain each model
            for model_name, model in self.models.items():
                model.fit(scaled_features, targets)
                logger.info(f"Retrained {model_name}")
            
            # Save models
            await self._save_models()
            
            logger.info("Model retraining completed successfully")
            
        except Exception as e:
            logger.error(f"Error retraining models: {e}")
    
    async def _save_models(self):
        """Save trained models to disk"""
        try:
            for model_name, model in self.models.items():
                joblib.dump(model, f'models/{model_name}.pkl')
            
            joblib.dump(self.scalers['main'], 'models/scaler.pkl')
            logger.info("Models saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    async def adjust_confidence_threshold(self, adjustment: float):
        """Adjust confidence threshold based on performance"""
        old_threshold = self.confidence_threshold
        self.confidence_threshold = min(0.95, max(0.5, self.confidence_threshold + adjustment))
        
        logger.info(f"Adjusted confidence threshold from {old_threshold:.3f} to {self.confidence_threshold:.3f}")
    
    async def get_model_insights(self) -> Dict[str, Any]:
        """Get insights about model performance and predictions"""
        try:
            insights = {
                'confidence_threshold': self.confidence_threshold,
                'ensemble_size': len(self.models),
                'model_performance': self.model_performance,
                'feature_importance': self.feature_importance,
                'recent_predictions': len(self.prediction_history),
                'target_success_rate': self.target_success_rate,
                'models_status': {name: 'active' for name in self.models.keys()}
            }
            
            # Add model-specific insights
            if hasattr(self.models.get('random_forest'), 'feature_importances_'):
                insights['feature_importance'] = dict(zip(
                    [f'feature_{i}' for i in range(len(self.models['random_forest'].feature_importances_))],
                    self.models['random_forest'].feature_importances_
                ))
            
            return insights
            
        except Exception as e:
            logger.error(f"Error getting model insights: {e}")
            return {
                'confidence_threshold': self.confidence_threshold,
                'ensemble_size': len(self.models),
                'error': str(e)
            }
