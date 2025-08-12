#!/usr/bin/env python3
"""
AI Position Size Calculator - Intelligent position sizing with risk management
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import joblib
import os
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class PositionSizeResult:
    """Position sizing result"""
    position_size: float
    risk_amount: float
    confidence: float
    method_used: str
    risk_metrics: Dict[str, float]
    constraints_applied: List[str]
    reasoning: List[str]

class AIPositionSizer:
    """Advanced AI-powered position sizing system"""
    
    def __init__(self, model_path: str = "models/position_sizer"):
        self.model_path = model_path
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        
        # Default parameters
        self.default_params = {
            'max_position_size': 0.1,      # 10% max position
            'max_portfolio_risk': 0.02,    # 2% max portfolio risk
            'max_single_trade_risk': 0.01, # 1% max single trade risk
            'min_position_size': 0.001,    # 0.1% min position
            'kelly_multiplier': 0.25,      # Conservative Kelly
            'volatility_adjustment': True,
            'correlation_adjustment': True,
            'regime_adjustment': True
        }
        
        # Risk constraints
        self.risk_constraints = {
            'max_leverage': 3.0,
            'max_concentration': 0.2,      # 20% max in single asset
            'max_sector_exposure': 0.3,    # 30% max in single sector
            'min_diversification': 5       # Min 5 positions
        }
        
        # Performance tracking
        self.sizing_history = []
        self.performance_metrics = {
            'total_positions': 0,
            'avg_position_size': 0.0,
            'avg_risk_per_trade': 0.0,
            'max_position_size': 0.0,
            'sizing_accuracy': 0.0
        }
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize or load the position sizing model"""
        try:
            if os.path.exists(f"{self.model_path}/position_sizer.pkl"):
                self.model = joblib.load(f"{self.model_path}/position_sizer.pkl")
                self.scaler = joblib.load(f"{self.model_path}/scaler.pkl")
                self.feature_names = joblib.load(f"{self.model_path}/feature_names.pkl")
                logger.info("Loaded existing position sizing model")
            else:
                self._create_model()
                
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            self._create_model()
    
    def _create_model(self):
        """Create new position sizing model"""
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        
        # Define feature names
        self.feature_names = [
            'signal_confidence', 'risk_reward_ratio', 'volatility',
            'portfolio_correlation', 'market_regime_score', 'liquidity_score',
            'portfolio_heat', 'recent_performance', 'drawdown_level',
            'kelly_fraction', 'sharpe_ratio', 'win_rate'
        ]
        
        logger.info("Created new position sizing model")
    
    def calculate_position_size(self, 
                              signal: Dict[str, Any],
                              portfolio: Dict[str, Any],
                              market_data: Dict[str, Any]) -> PositionSizeResult:
        """
        Calculate optimal position size using multiple methods
        
        Args:
            signal: Trading signal information
            portfolio: Current portfolio state
            market_data: Market context data
            
        Returns:
            PositionSizeResult with sizing recommendation
        """
        try:
            # Extract features
            features = self._extract_features(signal, portfolio, market_data)
            
            # Calculate base position sizes using different methods
            kelly_size = self._calculate_kelly_position(signal, portfolio)
            risk_parity_size = self._calculate_risk_parity_position(signal, portfolio)
            volatility_size = self._calculate_volatility_adjusted_position(signal, market_data)
            ai_size = self._calculate_ai_position(features)
            
            # Combine methods with weights
            method_weights = {
                'kelly': 0.3,
                'risk_parity': 0.25,
                'volatility': 0.2,
                'ai': 0.25
            }
            
            base_size = (
                kelly_size * method_weights['kelly'] +
                risk_parity_size * method_weights['risk_parity'] +
                volatility_size * method_weights['volatility'] +
                ai_size * method_weights['ai']
            )
            
            # Apply constraints and adjustments
            adjusted_size, constraints_applied = self._apply_constraints(
                base_size, signal, portfolio, market_data
            )
            
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(
                adjusted_size, signal, portfolio, market_data
            )
            
            # Calculate confidence
            confidence = self._calculate_sizing_confidence(
                signal, portfolio, market_data, features
            )
            
            # Generate reasoning
            reasoning = self._generate_reasoning(
                adjusted_size, base_size, constraints_applied, risk_metrics
            )
            
            # Determine method used
            method_used = self._determine_primary_method(
                kelly_size, risk_parity_size, volatility_size, ai_size, method_weights
            )
            
            # Create result
            result = PositionSizeResult(
                position_size=adjusted_size,
                risk_amount=risk_metrics['risk_amount'],
                confidence=confidence,
                method_used=method_used,
                risk_metrics=risk_metrics,
                constraints_applied=constraints_applied,
                reasoning=reasoning
            )
            
            # Update tracking
            self._update_tracking(result, signal, portfolio)
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return PositionSizeResult(
                position_size=self.default_params['min_position_size'],
                risk_amount=0.0,
                confidence=0.0,
                method_used="ERROR",
                risk_metrics={},
                constraints_applied=[f"Error: {str(e)}"],
                reasoning=[f"Error occurred, using minimum position size"]
            )
    
    def _extract_features(self, signal: Dict[str, Any], 
                         portfolio: Dict[str, Any], 
                         market_data: Dict[str, Any]) -> np.ndarray:
        """Extract features for AI model"""
        features = []
        
        # Signal features
        features.append(signal.get('confidence', 0.5))
        features.append(signal.get('risk_reward_ratio', 1.0))
        
        # Market features
        features.append(market_data.get('volatility', 0.02))
        features.append(portfolio.get('correlation_score', 0.5))
        features.append(market_data.get('regime_score', 0.5))
        features.append(signal.get('liquidity_score', 0.8))
        
        # Portfolio features
        features.append(portfolio.get('heat_level', 0.0))
        features.append(portfolio.get('recent_performance', 0.0))
        features.append(portfolio.get('drawdown_level', 0.0))
        
        # Performance features
        features.append(self._calculate_kelly_fraction(signal, portfolio))
        features.append(portfolio.get('sharpe_ratio', 0.0))
        features.append(portfolio.get('win_rate', 0.5))
        
        return np.array(features, dtype=np.float32)
    
    def _calculate_kelly_position(self, signal: Dict[str, Any], portfolio: Dict[str, Any]) -> float:
        """Calculate Kelly Criterion position size"""
        try:
            win_rate = portfolio.get('win_rate', 0.55)
            avg_win = portfolio.get('avg_win', 0.02)
            avg_loss = portfolio.get('avg_loss', 0.01)
            
            if avg_loss <= 0:
                return self.default_params['min_position_size']
            
            # Kelly fraction
            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            
            # Apply multiplier for safety
            kelly_size = kelly_fraction * self.default_params['kelly_multiplier']
            
            # Adjust for signal confidence
            confidence_adj = signal.get('confidence', 0.5)
            kelly_size *= confidence_adj
            
            return max(self.default_params['min_position_size'], 
                      min(kelly_size, self.default_params['max_position_size']))
            
        except Exception as e:
            logger.error(f"Error calculating Kelly position: {e}")
            return self.default_params['min_position_size']
    
    def _calculate_risk_parity_position(self, signal: Dict[str, Any], portfolio: Dict[str, Any]) -> float:
        """Calculate risk parity position size"""
        try:
            # Target risk per position
            target_risk = self.default_params['max_single_trade_risk']
            
            # Asset volatility
            volatility = signal.get('volatility', 0.02)
            
            # Stop loss distance
            stop_loss_pct = signal.get('stop_loss_distance', 0.02)
            
            if stop_loss_pct <= 0 or volatility <= 0:
                return self.default_params['min_position_size']
            
            # Risk parity size
            risk_parity_size = target_risk / stop_loss_pct
            
            # Adjust for volatility
            vol_adjustment = 0.02 / volatility  # Normalize to 2% volatility
            risk_parity_size *= vol_adjustment
            
            return max(self.default_params['min_position_size'],
                      min(risk_parity_size, self.default_params['max_position_size']))
            
        except Exception as e:
            logger.error(f"Error calculating risk parity position: {e}")
            return self.default_params['min_position_size']
    
    def _calculate_volatility_adjusted_position(self, signal: Dict[str, Any], market_data: Dict[str, Any]) -> float:
        """Calculate volatility-adjusted position size"""
        try:
            base_size = 0.05  # 5% base position
            
            # Current volatility
            current_vol = market_data.get('volatility', 0.02)
            
            # Target volatility
            target_vol = 0.02  # 2% target
            
            # Volatility adjustment
            vol_adj = target_vol / current_vol if current_vol > 0 else 1.0
            
            # Market regime adjustment
            regime_score = market_data.get('regime_score', 0.5)
            regime_adj = 0.5 + regime_score * 0.5  # 0.5 to 1.0 multiplier
            
            vol_size = base_size * vol_adj * regime_adj
            
            return max(self.default_params['min_position_size'],
                      min(vol_size, self.default_params['max_position_size']))
            
        except Exception as e:
            logger.error(f"Error calculating volatility position: {e}")
            return self.default_params['min_position_size']
    
    def _calculate_ai_position(self, features: np.ndarray) -> float:
        """Calculate AI model position size"""
        try:
            if self.model is None:
                return 0.03  # Default 3%
            
            # Scale features
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # Predict position size
            predicted_size = self.model.predict(features_scaled)[0]
            
            # Ensure within bounds
            predicted_size = max(self.default_params['min_position_size'],
                               min(predicted_size, self.default_params['max_position_size']))
            
            return predicted_size
            
        except Exception as e:
            logger.error(f"Error calculating AI position: {e}")
            return 0.03
    
    def _apply_constraints(self, base_size: float, 
                          signal: Dict[str, Any],
                          portfolio: Dict[str, Any],
                          market_data: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Apply risk constraints to position size"""
        adjusted_size = base_size
        constraints_applied = []
        
        # Maximum position size constraint
        if adjusted_size > self.default_params['max_position_size']:
            adjusted_size = self.default_params['max_position_size']
            constraints_applied.append("max_position_size")
        
        # Minimum position size constraint
        if adjusted_size < self.default_params['min_position_size']:
            adjusted_size = self.default_params['min_position_size']
            constraints_applied.append("min_position_size")
        
        # Portfolio heat constraint
        current_heat = portfolio.get('heat_level', 0.0)
        max_heat = 0.1  # 10% max portfolio heat
        
        if current_heat + adjusted_size > max_heat:
            adjusted_size = max(0, max_heat - current_heat)
            constraints_applied.append("portfolio_heat")
        
        # Correlation constraint
        correlation = portfolio.get('correlation_score', 0.0)
        if correlation > 0.7:  # High correlation
            adjusted_size *= 0.7  # Reduce size
            constraints_applied.append("high_correlation")
        
        # Drawdown constraint
        drawdown = portfolio.get('drawdown_level', 0.0)
        if drawdown > 0.05:  # 5% drawdown
            drawdown_multiplier = max(0.3, 1.0 - drawdown * 2)
            adjusted_size *= drawdown_multiplier
            constraints_applied.append("drawdown_protection")
        
        # Market regime constraint
        regime_score = market_data.get('regime_score', 0.5)
        if regime_score < 0.3:  # Unfavorable regime
            adjusted_size *= 0.6
            constraints_applied.append("unfavorable_regime")
        
        # Volatility constraint
        volatility = market_data.get('volatility', 0.02)
        if volatility > 0.05:  # High volatility
            vol_multiplier = 0.02 / volatility
            adjusted_size *= vol_multiplier
            constraints_applied.append("high_volatility")
        
        return adjusted_size, constraints_applied
    
    def _calculate_risk_metrics(self, position_size: float,
                               signal: Dict[str, Any],
                               portfolio: Dict[str, Any],
                               market_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate risk metrics for the position"""
        portfolio_value = portfolio.get('total_value', 100000)
        
        # Position value
        position_value = portfolio_value * position_size
        
        # Risk amount (stop loss)
        stop_loss_pct = signal.get('stop_loss_distance', 0.02)
        risk_amount = position_value * stop_loss_pct
        
        # Portfolio risk percentage
        portfolio_risk_pct = risk_amount / portfolio_value
        
        # Expected return
        risk_reward = signal.get('risk_reward_ratio', 1.5)
        expected_return = risk_amount * risk_reward
        
        # Volatility-adjusted risk
        volatility = market_data.get('volatility', 0.02)
        vol_adjusted_risk = risk_amount * (volatility / 0.02)
        
        return {
            'position_value': position_value,
            'risk_amount': risk_amount,
            'portfolio_risk_pct': portfolio_risk_pct,
            'expected_return': expected_return,
            'vol_adjusted_risk': vol_adjusted_risk,
            'risk_reward_ratio': risk_reward,
            'stop_loss_pct': stop_loss_pct
        }
    
    def _calculate_sizing_confidence(self, signal: Dict[str, Any],
                                   portfolio: Dict[str, Any],
                                   market_data: Dict[str, Any],
                                   features: np.ndarray) -> float:
        """Calculate confidence in position sizing"""
        confidence_factors = []
        
        # Signal confidence
        signal_conf = signal.get('confidence', 0.5)
        confidence_factors.append(signal_conf)
        
        # Market regime confidence
        regime_conf = market_data.get('regime_confidence', 0.5)
        confidence_factors.append(regime_conf)
        
        # Portfolio stability
        drawdown = portfolio.get('drawdown_level', 0.0)
        stability = max(0.0, 1.0 - drawdown * 5)
        confidence_factors.append(stability)
        
        # Feature consistency (low variance = high confidence)
        feature_std = np.std(features)
        consistency = max(0.0, 1.0 - feature_std)
        confidence_factors.append(consistency)
        
        # Historical performance
        win_rate = portfolio.get('win_rate', 0.5)
        performance_conf = min(1.0, win_rate * 2)
        confidence_factors.append(performance_conf)
        
        return np.mean(confidence_factors)
    
    def _generate_reasoning(self, final_size: float, base_size: float,
                          constraints: List[str], risk_metrics: Dict[str, float]) -> List[str]:
        """Generate reasoning for position sizing decision"""
        reasoning = []
        
        # Base sizing
        reasoning.append(f"Base position size: {base_size:.1%}")
        reasoning.append(f"Final position size: {final_size:.1%}")
        
        # Risk metrics
        reasoning.append(f"Portfolio risk: {risk_metrics.get('portfolio_risk_pct', 0):.2%}")
        reasoning.append(f"Risk-reward ratio: {risk_metrics.get('risk_reward_ratio', 0):.1f}")
        
        # Constraints applied
        if constraints:
            reasoning.append(f"Constraints applied: {', '.join(constraints)}")
        
        # Size adjustment
        if abs(final_size - base_size) > 0.001:
            change_pct = (final_size - base_size) / base_size * 100
            reasoning.append(f"Size adjusted by {change_pct:+.1f}% due to constraints")
        
        return reasoning
    
    def _determine_primary_method(self, kelly: float, risk_parity: float,
                                volatility: float, ai: float, weights: Dict[str, float]) -> str:
        """Determine which method had the most influence"""
        weighted_contributions = {
            'kelly': kelly * weights['kelly'],
            'risk_parity': risk_parity * weights['risk_parity'],
            'volatility': volatility * weights['volatility'],
            'ai': ai * weights['ai']
        }
        
        return max(weighted_contributions, key=weighted_contributions.get)
    
    def _calculate_kelly_fraction(self, signal: Dict[str, Any], portfolio: Dict[str, Any]) -> float:
        """Calculate Kelly fraction for features"""
        win_rate = portfolio.get('win_rate', 0.55)
        avg_win = portfolio.get('avg_win', 0.02)
        avg_loss = portfolio.get('avg_loss', 0.01)
        
        if avg_loss <= 0:
            return 0.0
        
        return (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
    
    def _update_tracking(self, result: PositionSizeResult, 
                        signal: Dict[str, Any], portfolio: Dict[str, Any]):
        """Update performance tracking"""
        self.sizing_history.append({
            'timestamp': datetime.now(),
            'position_size': result.position_size,
            'risk_amount': result.risk_amount,
            'confidence': result.confidence,
            'method_used': result.method_used,
            'signal': signal,
            'portfolio_state': portfolio
        })
        
        # Update performance metrics
        self.performance_metrics['total_positions'] += 1
        
        # Calculate running averages
        recent_sizes = [h['position_size'] for h in self.sizing_history[-100:]]
        self.performance_metrics['avg_position_size'] = np.mean(recent_sizes)
        
        recent_risks = [h['risk_amount'] for h in self.sizing_history[-100:]]
        self.performance_metrics['avg_risk_per_trade'] = np.mean(recent_risks)
        
        self.performance_metrics['max_position_size'] = max(recent_sizes)
        
        # Keep history manageable
        if len(self.sizing_history) > 1000:
            self.sizing_history = self.sizing_history[-500:]
    
    def train_model(self, training_data: pd.DataFrame = None):
        """Train the AI position sizing model"""
        try:
            if training_data is None:
                # Create synthetic training data if none provided
                training_data = self._create_synthetic_training_data()
            
            # Prepare features and targets
            feature_columns = self.feature_names
            X = training_data[feature_columns].fillna(0)
            y = training_data['optimal_position_size']
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.model.fit(X_scaled, y)
            
            # Save model
            self.save_model()
            
            logger.info("Position sizing model trained successfully")
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
    
    def _create_synthetic_training_data(self) -> pd.DataFrame:
        """Create synthetic training data for model training"""
        np.random.seed(42)
        n_samples = 1000
        
        data = []
        for _ in range(n_samples):
            # Generate random features
            signal_confidence = np.random.beta(2, 2)
            risk_reward_ratio = np.random.gamma(2, 0.5) + 1.0
            volatility = np.random.gamma(2, 0.01)
            portfolio_correlation = np.random.beta(2, 3)
            market_regime_score = np.random.beta(2, 2)
            liquidity_score = np.random.beta(3, 1)
            portfolio_heat = np.random.beta(1, 4)
            recent_performance = np.random.normal(0, 0.02)
            drawdown_level = np.random.beta(1, 9)
            kelly_fraction = np.random.normal(0.1, 0.05)
            sharpe_ratio = np.random.normal(1.0, 0.5)
            win_rate = np.random.beta(3, 2)
            
            # Calculate optimal position size based on features
            base_size = 0.05  # 5% base
            
            # Adjust for confidence
            size_adj = signal_confidence * 0.8 + 0.2
            
            # Adjust for risk-reward
            rr_adj = min(2.0, risk_reward_ratio / 2.0)
            
            # Adjust for volatility
            vol_adj = 0.02 / max(0.005, volatility)
            
            # Adjust for other factors
            regime_adj = 0.5 + market_regime_score * 0.5
            drawdown_adj = max(0.3, 1.0 - drawdown_level * 2)
            
            optimal_size = base_size * size_adj * rr_adj * vol_adj * regime_adj * drawdown_adj
            optimal_size = max(0.001, min(0.1, optimal_size))
            
            data.append({
                'signal_confidence': signal_confidence,
                'risk_reward_ratio': risk_reward_ratio,
                'volatility': volatility,
                'portfolio_correlation': portfolio_correlation,
                'market_regime_score': market_regime_score,
                'liquidity_score': liquidity_score,
                'portfolio_heat': portfolio_heat,
                'recent_performance': recent_performance,
                'drawdown_level': drawdown_level,
                'kelly_fraction': kelly_fraction,
                'sharpe_ratio': sharpe_ratio,
                'win_rate': win_rate,
                'optimal_position_size': optimal_size
            })
        
        return pd.DataFrame(data)
    
    def save_model(self):
        """Save the trained model"""
        try:
            os.makedirs(self.model_path, exist_ok=True)
            
            joblib.dump(self.model, f"{self.model_path}/position_sizer.pkl")
            joblib.dump(self.scaler, f"{self.model_path}/scaler.pkl")
            joblib.dump(self.feature_names, f"{self.model_path}/feature_names.pkl")
            
            logger.info("Position sizing model saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            'performance_metrics': self.performance_metrics.copy(),
            'recent_sizing_stats': self._get_recent_stats(),
            'parameter_settings': self.default_params.copy(),
            'constraint_settings': self.risk_constraints.copy()
        }
    
    def _get_recent_stats(self) -> Dict[str, Any]:
        """Get recent sizing statistics"""
        if not self.sizing_history:
            return {}
        
        recent = self.sizing_history[-50:]  # Last 50 positions
        
        sizes = [h['position_size'] for h in recent]
        risks = [h['risk_amount'] for h in recent]
        confidences = [h['confidence'] for h in recent]
        
        return {
            'count': len(recent),
            'avg_size': np.mean(sizes),
            'median_size': np.median(sizes),
            'std_size': np.std(sizes),
            'avg_risk': np.mean(risks),
            'avg_confidence': np.mean(confidences),
            'size_range': [min(sizes), max(sizes)]
        }
    
    def update_parameters(self, new_params: Dict[str, Any]):
        """Update position sizing parameters"""
        self.default_params.update(new_params)
        logger.info(f"Updated position sizing parameters: {new_params}")
    
    def update_constraints(self, new_constraints: Dict[str, Any]):
        """Update risk constraints"""
        self.risk_constraints.update(new_constraints)
        logger.info(f"Updated risk constraints: {new_constraints}")

def main():
    """Test the position sizer"""
    sizer = AIPositionSizer()
    
    # Test signal
    test_signal = {
        'symbol': 'AAPL',
        'action': 'BUY',
        'confidence': 0.8,
        'risk_reward_ratio': 2.5,
        'volatility': 0.025,
        'stop_loss_distance': 0.03,
        'liquidity_score': 0.9
    }
    
    # Test portfolio
    test_portfolio = {
        'total_value': 100000,
        'heat_level': 0.05,
        'correlation_score': 0.3,
        'recent_performance': 0.02,
        'drawdown_level': 0.02,
        'win_rate': 0.6,
        'avg_win': 0.025,
        'avg_loss': 0.015,
        'sharpe_ratio': 1.2
    }
    
    # Test market data
    test_market = {
        'volatility': 0.02,
        'regime_score': 0.7,
        'regime_confidence': 0.8
    }
    
    # Calculate position size
    result = sizer.calculate_position_size(test_signal, test_portfolio, test_market)
    
    print(f"Position Size: {result.position_size:.2%}")
    print(f"Risk Amount: ${result.risk_amount:.2f}")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"Method Used: {result.method_used}")
    print(f"Constraints Applied: {result.constraints_applied}")
    print(f"Reasoning: {'; '.join(result.reasoning)}")
    
    # Get performance metrics
    metrics = sizer.get_performance_metrics()
    print(f"\nPerformance Metrics: {metrics}")

if __name__ == "__main__":
    main()
