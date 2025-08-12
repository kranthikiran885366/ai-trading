#!/usr/bin/env python3
"""
AI Risk Predictor - Advanced risk assessment and prediction system
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import os
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class RiskPrediction:
    """Risk prediction result"""
    var_1d: float           # 1-day Value at Risk
    var_5d: float           # 5-day Value at Risk
    expected_shortfall: float  # Expected Shortfall (CVaR)
    max_drawdown_risk: float   # Maximum Drawdown Risk
    volatility_forecast: float # Volatility Forecast
    risk_level: str         # Risk Level (LOW/MEDIUM/HIGH/CRITICAL)
    confidence: float       # Prediction Confidence
    risk_factors: Dict[str, float]  # Individual Risk Factors
    alerts: List[str]       # Risk Alerts
    timestamp: datetime

class RiskPredictorNN(nn.Module):
    """Neural Network for Risk Prediction"""
    
    def __init__(self, input_dim: int = 20, hidden_dims: List[int] = [64, 32, 16]):
        super(RiskPredictorNN, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        # Output layer for multiple risk metrics
        layers.append(nn.Linear(prev_dim, 5))  # VaR_1d, VaR_5d, ES, MaxDD, Vol
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class AIRiskPredictor:
    """Advanced AI-powered risk prediction system"""
    
    def __init__(self, model_path: str = "models/risk_predictor"):
        self.model_path = model_path
        self.model = None
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Risk thresholds
        self.risk_thresholds = {
            'var_1d': {'low': 0.01, 'medium': 0.02, 'high': 0.04, 'critical': 0.08},
            'var_5d': {'low': 0.02, 'medium': 0.04, 'high': 0.08, 'critical': 0.15},
            'expected_shortfall': {'low': 0.015, 'medium': 0.03, 'high': 0.06, 'critical': 0.12},
            'max_drawdown': {'low': 0.05, 'medium': 0.10, 'high': 0.20, 'critical': 0.35},
            'volatility': {'low': 0.01, 'medium': 0.02, 'high': 0.04, 'critical': 0.08}
        }
        
        # Feature names
        self.feature_names = [
            'returns_1d', 'returns_5d', 'returns_20d',
            'volatility_5d', 'volatility_20d', 'volatility_60d',
            'skewness', 'kurtosis', 'max_drawdown_current',
            'var_ratio', 'correlation_spy', 'beta',
            'rsi', 'macd_signal', 'bb_position',
            'volume_ratio', 'liquidity_score', 'market_cap_log',
            'sector_beta', 'momentum_score'
        ]
        
        # Prediction history
        self.prediction_history = []
        self.performance_metrics = {
            'total_predictions': 0,
            'accuracy_var': 0.0,
            'accuracy_es': 0.0,
            'accuracy_drawdown': 0.0,
            'avg_confidence': 0.0
        }
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize or load the risk prediction model"""
        try:
            if os.path.exists(f"{self.model_path}/risk_predictor.pth"):
                self.model = RiskPredictorNN(len(self.feature_names))
                self.model.load_state_dict(torch.load(f"{self.model_path}/risk_predictor.pth", map_location=self.device))
                self.model.to(self.device)
                self.model.eval()
                
                self.scaler = joblib.load(f"{self.model_path}/scaler.pkl")
                logger.info("Loaded existing risk prediction model")
            else:
                self._create_and_train_model()
                
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            self._create_and_train_model()
    
    def _create_and_train_model(self):
        """Create and train a new model"""
        logger.info("Creating and training new risk prediction model...")
        
        self.model = RiskPredictorNN(len(self.feature_names))
        self.model.to(self.device)
        
        # Train with synthetic data initially
        self.train_model()
    
    def predict_risk(self, market_data: pd.DataFrame, 
                    portfolio_data: Dict[str, Any] = None) -> RiskPrediction:
        """
        Predict various risk metrics
        
        Args:
            market_data: Historical market data
            portfolio_data: Portfolio context data
            
        Returns:
            RiskPrediction with comprehensive risk assessment
        """
        try:
            # Extract features
            features = self._extract_risk_features(market_data, portfolio_data)
            
            # Make prediction
            risk_metrics = self._predict_risk_metrics(features)
            
            # Calculate risk factors
            risk_factors = self._calculate_risk_factors(features, market_data)
            
            # Determine risk level
            risk_level = self._determine_risk_level(risk_metrics)
            
            # Calculate confidence
            confidence = self._calculate_prediction_confidence(features, risk_metrics)
            
            # Generate alerts
            alerts = self._generate_risk_alerts(risk_metrics, risk_factors)
            
            # Create prediction result
            prediction = RiskPrediction(
                var_1d=risk_metrics['var_1d'],
                var_5d=risk_metrics['var_5d'],
                expected_shortfall=risk_metrics['expected_shortfall'],
                max_drawdown_risk=risk_metrics['max_drawdown_risk'],
                volatility_forecast=risk_metrics['volatility_forecast'],
                risk_level=risk_level,
                confidence=confidence,
                risk_factors=risk_factors,
                alerts=alerts,
                timestamp=datetime.now()
            )
            
            # Update tracking
            self._update_tracking(prediction, features)
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error predicting risk: {e}")
            return RiskPrediction(
                var_1d=0.05, var_5d=0.1, expected_shortfall=0.075,
                max_drawdown_risk=0.15, volatility_forecast=0.03,
                risk_level="HIGH", confidence=0.0,
                risk_factors={}, alerts=[f"Error: {str(e)}"],
                timestamp=datetime.now()
            )
    
    def _extract_risk_features(self, market_data: pd.DataFrame, 
                              portfolio_data: Dict[str, Any] = None) -> np.ndarray:
        """Extract features for risk prediction"""
        if market_data.empty or len(market_data) < 60:
            # Return default features if insufficient data
            return np.zeros(len(self.feature_names))
        
        df = market_data.copy()
        features = []
        
        # Returns features
        df['returns'] = df['Close'].pct_change()
        features.append(df['returns'].iloc[-1])  # 1-day return
        features.append(df['returns'].iloc[-5:].mean())  # 5-day avg return
        features.append(df['returns'].iloc[-20:].mean())  # 20-day avg return
        
        # Volatility features
        features.append(df['returns'].iloc[-5:].std())   # 5-day volatility
        features.append(df['returns'].iloc[-20:].std())  # 20-day volatility
        features.append(df['returns'].iloc[-60:].std())  # 60-day volatility
        
        # Distribution features
        returns_20d = df['returns'].iloc[-20:].dropna()
        features.append(returns_20d.skew() if len(returns_20d) > 3 else 0.0)  # Skewness
        features.append(returns_20d.kurtosis() if len(returns_20d) > 3 else 0.0)  # Kurtosis
        
        # Drawdown features
        cumulative = (1 + df['returns']).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        features.append(drawdown.iloc[-1])  # Current drawdown
        
        # Variance ratio (mean reversion test)
        if len(returns_20d) >= 10:
            var_1 = returns_20d.var()
            var_5 = returns_20d.rolling(5).sum().var() / 5
            var_ratio = var_5 / var_1 if var_1 != 0 else 1.0
        else:
            var_ratio = 1.0
        features.append(var_ratio)
        
        # Market correlation (using SPY as proxy)
        features.append(0.7)  # Default correlation
        features.append(1.2)  # Default beta
        
        # Technical indicators
        if len(df) >= 14:
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            features.append(rsi.iloc[-1] / 100.0 if not np.isnan(rsi.iloc[-1]) else 0.5)
        else:
            features.append(0.5)
        
        # MACD
        if len(df) >= 26:
            ema12 = df['Close'].ewm(span=12).mean()
            ema26 = df['Close'].ewm(span=26).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9).mean()
            features.append((macd.iloc[-1] - signal.iloc[-1]) / df['Close'].iloc[-1])
        else:
            features.append(0.0)
        
        # Bollinger Bands position
        if len(df) >= 20:
            sma20 = df['Close'].rolling(20).mean()
            std20 = df['Close'].rolling(20).std()
            bb_upper = sma20 + (std20 * 2)
            bb_lower = sma20 - (std20 * 2)
            bb_position = (df['Close'].iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
            features.append(bb_position if not np.isnan(bb_position) else 0.5)
        else:
            features.append(0.5)
        
        # Volume features
        if 'Volume' in df.columns:
            vol_sma = df['Volume'].rolling(20).mean()
            vol_ratio = df['Volume'].iloc[-1] / vol_sma.iloc[-1] if vol_sma.iloc[-1] != 0 else 1.0
            features.append(vol_ratio)
        else:
            features.append(1.0)
        
        # Liquidity score (simplified)
        features.append(0.8)  # Default liquidity score
        
        # Market cap (log)
        market_cap = portfolio_data.get('market_cap', 1e9) if portfolio_data else 1e9
        features.append(np.log(market_cap) / 25.0)  # Normalized
        
        # Sector beta
        features.append(portfolio_data.get('sector_beta', 1.0) if portfolio_data else 1.0)
        
        # Momentum score
        if len(df) >= 20:
            momentum = (df['Close'].iloc[-1] / df['Close'].iloc[-20] - 1) * 100
            features.append(momentum / 20.0)  # Normalized
        else:
            features.append(0.0)
        
        # Ensure we have the right number of features
        while len(features) < len(self.feature_names):
            features.append(0.0)
        
        return np.array(features[:len(self.feature_names)], dtype=np.float32)
    
    def _predict_risk_metrics(self, features: np.ndarray) -> Dict[str, float]:
        """Predict risk metrics using the neural network"""
        try:
            if self.model is None:
                return self._get_default_risk_metrics()
            
            # Scale features
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # Convert to tensor
            features_tensor = torch.FloatTensor(features_scaled).to(self.device)
            
            # Predict
            with torch.no_grad():
                predictions = self.model(features_tensor).cpu().numpy()[0]
            
            # Ensure positive values and reasonable ranges
            var_1d = max(0.001, min(0.2, abs(predictions[0])))
            var_5d = max(var_1d, min(0.3, abs(predictions[1])))
            expected_shortfall = max(var_1d * 1.2, min(0.4, abs(predictions[2])))
            max_drawdown_risk = max(0.01, min(0.5, abs(predictions[3])))
            volatility_forecast = max(0.005, min(0.1, abs(predictions[4])))
            
            return {
                'var_1d': var_1d,
                'var_5d': var_5d,
                'expected_shortfall': expected_shortfall,
                'max_drawdown_risk': max_drawdown_risk,
                'volatility_forecast': volatility_forecast
            }
            
        except Exception as e:
            logger.error(f"Error in risk prediction: {e}")
            return self._get_default_risk_metrics()
    
    def _get_default_risk_metrics(self) -> Dict[str, float]:
        """Get default risk metrics when prediction fails"""
        return {
            'var_1d': 0.02,
            'var_5d': 0.045,
            'expected_shortfall': 0.03,
            'max_drawdown_risk': 0.1,
            'volatility_forecast': 0.025
        }
    
    def _calculate_risk_factors(self, features: np.ndarray, 
                               market_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate individual risk factor contributions"""
        risk_factors = {}
        
        # Volatility risk
        vol_5d = features[3] if len(features) > 3 else 0.02
        risk_factors['volatility_risk'] = min(1.0, vol_5d / 0.05)
        
        # Momentum risk
        momentum = features[-1] if len(features) > 0 else 0.0
        risk_factors['momentum_risk'] = min(1.0, abs(momentum) / 0.1)
        
        # Drawdown risk
        current_dd = features[8] if len(features) > 8 else 0.0
        risk_factors['drawdown_risk'] = min(1.0, abs(current_dd) / 0.2)
        
        # Distribution risk (skewness and kurtosis)
        skewness = features[6] if len(features) > 6 else 0.0
        kurtosis = features[7] if len(features) > 7 else 0.0
        risk_factors['distribution_risk'] = min(1.0, (abs(skewness) + abs(kurtosis)) / 5.0)
        
        # Correlation risk
        correlation = features[10] if len(features) > 10 else 0.7
        risk_factors['correlation_risk'] = min(1.0, abs(correlation) / 0.9)
        
        # Liquidity risk
        volume_ratio = features[15] if len(features) > 15 else 1.0
        risk_factors['liquidity_risk'] = min(1.0, max(0.0, (2.0 - volume_ratio) / 2.0))
        
        # Technical risk
        rsi = features[12] if len(features) > 12 else 0.5
        bb_pos = features[14] if len(features) > 14 else 0.5
        
        # Extreme RSI values indicate risk
        rsi_risk = max(0.0, abs(rsi - 0.5) - 0.3) / 0.2
        bb_risk = max(0.0, abs(bb_pos - 0.5) - 0.4) / 0.1
        risk_factors['technical_risk'] = min(1.0, max(rsi_risk, bb_risk))
        
        return risk_factors
    
    def _determine_risk_level(self, risk_metrics: Dict[str, float]) -> str:
        """Determine overall risk level"""
        risk_scores = []
        
        for metric, value in risk_metrics.items():
            if metric in self.risk_thresholds:
                thresholds = self.risk_thresholds[metric]
                
                if value >= thresholds['critical']:
                    risk_scores.append(4)
                elif value >= thresholds['high']:
                    risk_scores.append(3)
                elif value >= thresholds['medium']:
                    risk_scores.append(2)
                elif value >= thresholds['low']:
                    risk_scores.append(1)
                else:
                    risk_scores.append(0)
        
        if not risk_scores:
            return "MEDIUM"
        
        avg_risk_score = np.mean(risk_scores)
        
        if avg_risk_score >= 3.5:
            return "CRITICAL"
        elif avg_risk_score >= 2.5:
            return "HIGH"
        elif avg_risk_score >= 1.5:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _calculate_prediction_confidence(self, features: np.ndarray, 
                                       risk_metrics: Dict[str, float]) -> float:
        """Calculate confidence in risk predictions"""
        confidence_factors = []
        
        # Data quality factor
        non_zero_features = np.count_nonzero(features)
        data_quality = non_zero_features / len(features)
        confidence_factors.append(data_quality)
        
        # Feature consistency (low variance = high confidence)
        feature_std = np.std(features)
        consistency = max(0.0, 1.0 - feature_std)
        confidence_factors.append(consistency)
        
        # Prediction consistency (similar risk metrics = high confidence)
        risk_values = list(risk_metrics.values())
        risk_std = np.std(risk_values)
        risk_consistency = max(0.0, 1.0 - risk_std * 5)
        confidence_factors.append(risk_consistency)
        
        # Model confidence (based on training performance)
        model_confidence = 0.8  # Would be based on validation metrics
        confidence_factors.append(model_confidence)
        
        return np.mean(confidence_factors)
    
    def _generate_risk_alerts(self, risk_metrics: Dict[str, float], 
                             risk_factors: Dict[str, float]) -> List[str]:
        """Generate risk alerts based on predictions"""
        alerts = []
        
        # VaR alerts
        if risk_metrics['var_1d'] > self.risk_thresholds['var_1d']['high']:
            alerts.append(f"High 1-day VaR: {risk_metrics['var_1d']:.1%}")
        
        if risk_metrics['var_5d'] > self.risk_thresholds['var_5d']['high']:
            alerts.append(f"High 5-day VaR: {risk_metrics['var_5d']:.1%}")
        
        # Expected Shortfall alert
        if risk_metrics['expected_shortfall'] > self.risk_thresholds['expected_shortfall']['high']:
            alerts.append(f"High Expected Shortfall: {risk_metrics['expected_shortfall']:.1%}")
        
        # Drawdown alert
        if risk_metrics['max_drawdown_risk'] > self.risk_thresholds['max_drawdown']['medium']:
            alerts.append(f"Elevated Drawdown Risk: {risk_metrics['max_drawdown_risk']:.1%}")
        
        # Volatility alert
        if risk_metrics['volatility_forecast'] > self.risk_thresholds['volatility']['high']:
            alerts.append(f"High Volatility Forecast: {risk_metrics['volatility_forecast']:.1%}")
        
        # Risk factor alerts
        for factor, value in risk_factors.items():
            if value > 0.8:
                alerts.append(f"High {factor.replace('_', ' ')}: {value:.1%}")
        
        return alerts
    
    def _update_tracking(self, prediction: RiskPrediction, features: np.ndarray):
        """Update prediction tracking and performance metrics"""
        self.prediction_history.append({
            'timestamp': prediction.timestamp,
            'prediction': prediction,
            'features': features.tolist()
        })
        
        # Update performance metrics
        self.performance_metrics['total_predictions'] += 1
        
        # Calculate running averages
        recent_predictions = self.prediction_history[-100:]
        confidences = [p['prediction'].confidence for p in recent_predictions]
        self.performance_metrics['avg_confidence'] = np.mean(confidences)
        
        # Keep history manageable
        if len(self.prediction_history) > 1000:
            self.prediction_history = self.prediction_history[-500:]
    
    def train_model(self, training_data: pd.DataFrame = None):
        """Train the risk prediction model"""
        try:
            if training_data is None:
                training_data = self._create_synthetic_training_data()
            
            # Prepare features and targets
            X = training_data[self.feature_names].fillna(0).values
            y = training_data[['var_1d', 'var_5d', 'expected_shortfall', 
                              'max_drawdown_risk', 'volatility_forecast']].values
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Convert to tensors
            X_train_tensor = torch.FloatTensor(X_train_scaled).to(self.device)
            y_train_tensor = torch.FloatTensor(y_train).to(self.device)
            X_test_tensor = torch.FloatTensor(X_test_scaled).to(self.device)
            y_test_tensor = torch.FloatTensor(y_test).to(self.device)
            
            # Training setup
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
            
            # Training loop
            self.model.train()
            best_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(200):
                # Forward pass
                outputs = self.model(X_train_tensor)
                loss = criterion(outputs, y_train_tensor)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Validation
                if epoch % 10 == 0:
                    self.model.eval()
                    with torch.no_grad():
                        val_outputs = self.model(X_test_tensor)
                        val_loss = criterion(val_outputs, y_test_tensor)
                    
                    scheduler.step(val_loss)
                    
                    if val_loss < best_loss:
                        best_loss = val_loss
                        patience_counter = 0
                        # Save best model
                        torch.save(self.model.state_dict(), f"{self.model_path}/risk_predictor_best.pth")
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= 20:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break
                    
                    self.model.train()
                    
                    if epoch % 50 == 0:
                        logger.info(f"Epoch {epoch}, Train Loss: {loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Load best model
            self.model.load_state_dict(torch.load(f"{self.model_path}/risk_predictor_best.pth"))
            self.model.eval()
            
            # Save final model and scaler
            self.save_model()
            
            # Calculate final metrics
            with torch.no_grad():
                final_outputs = self.model(X_test_tensor).cpu().numpy()
                mse = mean_squared_error(y_test, final_outputs)
                mae = mean_absolute_error(y_test, final_outputs)
            
            logger.info(f"Risk prediction model trained successfully")
            logger.info(f"Final MSE: {mse:.6f}, MAE: {mae:.6f}")
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
    
    def _create_synthetic_training_data(self) -> pd.DataFrame:
        """Create synthetic training data for model training"""
        np.random.seed(42)
        n_samples = 5000
        
        data = []
        for _ in range(n_samples):
            # Generate random features
            features = {}
            
            # Returns features
            features['returns_1d'] = np.random.normal(0, 0.02)
            features['returns_5d'] = np.random.normal(0, 0.01)
            features['returns_20d'] = np.random.normal(0, 0.005)
            
            # Volatility features
            base_vol = np.random.gamma(2, 0.01)
            features['volatility_5d'] = base_vol * np.random.uniform(0.8, 1.2)
            features['volatility_20d'] = base_vol * np.random.uniform(0.9, 1.1)
            features['volatility_60d'] = base_vol * np.random.uniform(0.95, 1.05)
            
            # Distribution features
            features['skewness'] = np.random.normal(0, 1)
            features['kurtosis'] = np.random.gamma(2, 1)
            
            # Other features
            features['max_drawdown_current'] = -np.random.beta(1, 4) * 0.3
            features['var_ratio'] = np.random.gamma(2, 0.5)
            features['correlation_spy'] = np.random.beta(2, 2) * 0.8 + 0.1
            features['beta'] = np.random.gamma(2, 0.5)
            features['rsi'] = np.random.beta(2, 2)
            features['macd_signal'] = np.random.normal(0, 0.01)
            features['bb_position'] = np.random.beta(2, 2)
            features['volume_ratio'] = np.random.gamma(2, 0.5)
            features['liquidity_score'] = np.random.beta(3, 1)
            features['market_cap_log'] = np.random.normal(0.5, 0.2)
            features['sector_beta'] = np.random.gamma(2, 0.5)
            features['momentum_score'] = np.random.normal(0, 0.1)
            
            # Calculate target risk metrics based on features
            vol_factor = features['volatility_20d']
            dd_factor = abs(features['max_drawdown_current'])
            skew_factor = abs(features['skewness'])
            
            # VaR calculations
            var_1d = vol_factor * (1 + dd_factor + skew_factor * 0.1) * np.random.uniform(0.8, 1.2)
            var_5d = var_1d * np.sqrt(5) * np.random.uniform(0.9, 1.1)
            
            # Expected Shortfall (typically 1.2-1.5x VaR)
            expected_shortfall = var_1d * np.random.uniform(1.2, 1.5)
            
            # Max Drawdown Risk
            max_drawdown_risk = (dd_factor + vol_factor * 2) * np.random.uniform(0.8, 1.2)
            
            # Volatility Forecast
            volatility_forecast = vol_factor * np.random.uniform(0.9, 1.1)
            
            # Add targets to features
            features.update({
                'var_1d': max(0.001, min(0.2, var_1d)),
                'var_5d': max(0.002, min(0.3, var_5d)),
                'expected_shortfall': max(0.001, min(0.4, expected_shortfall)),
                'max_drawdown_risk': max(0.01, min(0.5, max_drawdown_risk)),
                'volatility_forecast': max(0.005, min(0.1, volatility_forecast))
            })
            
            data.append(features)
        
        return pd.DataFrame(data)
    
    def save_model(self):
        """Save the trained model and scaler"""
        try:
            os.makedirs(self.model_path, exist_ok=True)
            
            torch.save(self.model.state_dict(), f"{self.model_path}/risk_predictor.pth")
            joblib.dump(self.scaler, f"{self.model_path}/scaler.pkl")
            
            logger.info("Risk prediction model saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            'performance_metrics': self.performance_metrics.copy(),
            'risk_thresholds': self.risk_thresholds.copy(),
            'recent_predictions': len(self.prediction_history),
            'model_info': {
                'device': str(self.device),
                'feature_count': len(self.feature_names),
                'model_parameters': sum(p.numel() for p in self.model.parameters()) if self.model else 0
            }
        }
    
    def update_thresholds(self, new_thresholds: Dict[str, Dict[str, float]]):
        """Update risk thresholds"""
        for metric, thresholds in new_thresholds.items():
            if metric in self.risk_thresholds:
                self.risk_thresholds[metric].update(thresholds)
        
        logger.info(f"Updated risk thresholds: {new_thresholds}")
    
    def get_risk_statistics(self) -> Dict[str, Any]:
        """Get risk prediction statistics"""
        if not self.prediction_history:
            return {}
        
        recent = self.prediction_history[-50:]
        
        var_1d_values = [p['prediction'].var_1d for p in recent]
        risk_levels = [p['prediction'].risk_level for p in recent]
        confidences = [p['prediction'].confidence for p in recent]
        
        return {
            'recent_predictions_count': len(recent),
            'avg_var_1d': np.mean(var_1d_values),
            'max_var_1d': max(var_1d_values),
            'avg_confidence': np.mean(confidences),
            'risk_level_distribution': {
                level: risk_levels.count(level) for level in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
            },
            'alert_frequency': np.mean([len(p['prediction'].alerts) for p in recent])
        }

def main():
    """Test the risk predictor"""
    predictor = AIRiskPredictor()
    
    # Create test data
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    np.random.seed(42)
    
    # Simulate price data with some volatility clustering
    returns = np.random.normal(0.001, 0.02, len(dates))
    prices = [100]
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    test_data = pd.DataFrame({
        'Date': dates,
        'Close': prices[1:],
        'Volume': np.random.randint(1000000, 5000000, len(dates))
    })
    
    # Test portfolio data
    portfolio_data = {
        'market_cap': 1e10,
        'sector_beta': 1.2
    }
    
    # Predict risk
    risk_prediction = predictor.predict_risk(test_data, portfolio_data)
    
    print(f"Risk Level: {risk_prediction.risk_level}")
    print(f"1-Day VaR: {risk_prediction.var_1d:.2%}")
    print(f"5-Day VaR: {risk_prediction.var_5d:.2%}")
    print(f"Expected Shortfall: {risk_prediction.expected_shortfall:.2%}")
    print(f"Max Drawdown Risk: {risk_prediction.max_drawdown_risk:.2%}")
    print(f"Volatility Forecast: {risk_prediction.volatility_forecast:.2%}")
    print(f"Confidence: {risk_prediction.confidence:.2%}")
    print(f"Alerts: {risk_prediction.alerts}")
    
    # Get statistics
    stats = predictor.get_risk_statistics()
    print(f"\nStatistics: {stats}")

if __name__ == "__main__":
    main()
