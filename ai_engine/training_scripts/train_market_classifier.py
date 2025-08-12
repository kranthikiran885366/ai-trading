import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import yfinance as yf
import joblib
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class MarketClassifierTrainer:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        
    def prepare_market_data(self, symbols, period='5y', interval='1d'):
        """Prepare market regime classification data"""
        all_data = []
        
        for symbol in symbols:
            print(f"Processing {symbol}...")
            try:
                # Download data
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=period, interval=interval)
                
                if df.empty:
                    continue
                
                # Calculate market features
                features_df = self._calculate_market_features(df)
                
                # Create market regime labels
                labels = self._create_market_regime_labels(df)
                
                # Combine features and labels
                combined_df = pd.concat([features_df, labels], axis=1)
                combined_df['symbol'] = symbol
                
                all_data.append(combined_df)
                
            except Exception as e:
                print(f"Error processing {symbol}: {e}")
                continue
        
        if not all_data:
            raise ValueError("No data could be processed")
        
        # Combine all data
        final_df = pd.concat(all_data, ignore_index=True)
        final_df = final_df.dropna()
        
        print(f"Total samples: {len(final_df)}")
        return final_df
    
    def _calculate_market_features(self, df):
        """Calculate market regime features"""
        features = pd.DataFrame(index=df.index)
        
        # Price momentum features
        for period in [5, 10, 20, 50, 100]:
            features[f'return_{period}d'] = df['Close'].pct_change(period)
            features[f'volatility_{period}d'] = df['Close'].pct_change().rolling(period).std()
        
        # Moving average trends
        for period in [20, 50, 100, 200]:
            ma = df['Close'].rolling(period).mean()
            features[f'price_ma_{period}_ratio'] = df['Close'] / ma
            features[f'ma_{period}_slope'] = ma.pct_change(5)
        
        # Volatility regime indicators
        features['volatility_regime'] = self._calculate_volatility_regime(df)
        features['volatility_percentile'] = self._calculate_volatility_percentile(df)
        
        # Trend strength
        features['trend_strength'] = self._calculate_trend_strength(df)
        features['trend_consistency'] = self._calculate_trend_consistency(df)
        
        # Market breadth (using volume as proxy)
        features['volume_trend'] = df['Volume'].rolling(20).mean() / df['Volume'].rolling(50).mean()
        features['volume_volatility'] = df['Volume'].pct_change().rolling(20).std()
        
        # Support/Resistance levels
        features['support_strength'] = self._calculate_support_strength(df)
        features['resistance_strength'] = self._calculate_resistance_strength(df)
        
        # Market structure
        features['higher_highs'] = self._count_higher_highs(df)
        features['lower_lows'] = self._count_lower_lows(df)
        
        # Momentum oscillators
        features['rsi_regime'] = self._calculate_rsi_regime(df)
        features['macd_regime'] = self._calculate_macd_regime(df)
        
        # Cross-asset correlations (simplified)
        features['correlation_regime'] = self._estimate_correlation_regime(df)
        
        return features
    
    def _create_market_regime_labels(self, df, lookforward=20):
        """Create market regime labels"""
        labels = pd.Series(index=df.index, name='market_regime')
        
        # Calculate future volatility and returns
        future_volatility = df['Close'].pct_change().rolling(lookforward).std().shift(-lookforward)
        future_returns = (df['Close'].shift(-lookforward) / df['Close'] - 1)
        
        # Define regime thresholds
        vol_high_threshold = future_volatility.quantile(0.75)
        vol_low_threshold = future_volatility.quantile(0.25)
        return_threshold = 0.05
        
        # Classify regimes
        # 0: BULL (low volatility, positive returns)
        # 1: BEAR (high volatility, negative returns)  
        # 2: SIDEWAYS (low volatility, low returns)
        # 3: VOLATILE (high volatility, mixed returns)
        
        bull_mask = (future_volatility <= vol_low_threshold) & (future_returns > return_threshold)
        bear_mask = (future_volatility >= vol_high_threshold) & (future_returns < -return_threshold)
        sideways_mask = (future_volatility <= vol_low_threshold) & (abs(future_returns) <= return_threshold)
        volatile_mask = future_volatility >= vol_high_threshold
        
        labels[:] = 2  # Default to SIDEWAYS
        labels[bull_mask] = 0  # BULL
        labels[bear_mask] = 1  # BEAR
        labels[volatile_mask] = 3  # VOLATILE
        
        return labels
    
    def _calculate_volatility_regime(self, df, window=20):
        """Calculate volatility regime indicator"""
        volatility = df['Close'].pct_change().rolling(window).std()
        vol_ma = volatility.rolling(window*2).mean()
        return volatility / vol_ma
    
    def _calculate_volatility_percentile(self, df, window=252):
        """Calculate volatility percentile"""
        volatility = df['Close'].pct_change().rolling(20).std()
        return volatility.rolling(window).rank(pct=True)
    
    def _calculate_trend_strength(self, df, window=20):
        """Calculate trend strength using ADX-like calculation"""
        high_low = df['High'] - df['Low']
        high_close = abs(df['High'] - df['Close'].shift(1))
        low_close = abs(df['Low'] - df['Close'].shift(1))
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        
        plus_dm = np.where((df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']),
                          np.maximum(df['High'] - df['High'].shift(1), 0), 0)
        minus_dm = np.where((df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)),
                           np.maximum(df['Low'].shift(1) - df['Low'], 0), 0)
        
        plus_di = 100 * (pd.Series(plus_dm).rolling(window).mean() / 
                        pd.Series(true_range).rolling(window).mean())
        minus_di = 100 * (pd.Series(minus_dm).rolling(window).mean() / 
                         pd.Series(true_range).rolling(window).mean())
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window).mean()
        
        return adx
    
    def _calculate_trend_consistency(self, df, window=20):
        """Calculate trend consistency"""
        returns = df['Close'].pct_change()
        positive_days = (returns > 0).rolling(window).sum()
        return positive_days / window
    
    def _calculate_support_strength(self, df, window=20):
        """Calculate support level strength"""
        lows = df['Low'].rolling(window).min()
        touches = (df['Low'] <= lows * 1.02).rolling(window).sum()
        return touches / window
    
    def _calculate_resistance_strength(self, df, window=20):
        """Calculate resistance level strength"""
        highs = df['High'].rolling(window).max()
        touches = (df['High'] >= highs * 0.98).rolling(window).sum()
        return touches / window
    
    def _count_higher_highs(self, df, window=10):
        """Count higher highs in window"""
        higher_highs = (df['High'] > df['High'].shift(1)).rolling(window).sum()
        return higher_highs / window
    
    def _count_lower_lows(self, df, window=10):
        """Count lower lows in window"""
        lower_lows = (df['Low'] < df['Low'].shift(1)).rolling(window).sum()
        return lower_lows / window
    
    def _calculate_rsi_regime(self, df, window=14):
        """Calculate RSI-based regime indicator"""
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Classify RSI regime
        regime = pd.Series(index=df.index)
        regime[rsi < 30] = 0  # Oversold
        regime[rsi > 70] = 2  # Overbought
        regime[(rsi >= 30) & (rsi <= 70)] = 1  # Normal
        
        return regime
    
    def _calculate_macd_regime(self, df):
        """Calculate MACD-based regime indicator"""
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9).mean()
        
        regime = pd.Series(index=df.index)
        regime[macd > signal] = 1  # Bullish
        regime[macd <= signal] = 0  # Bearish
        
        return regime
    
    def _estimate_correlation_regime(self, df):
        """Estimate correlation regime (simplified)"""
        # This is a simplified version - in practice, you'd use multiple assets
        volatility = df['Close'].pct_change().rolling(20).std()
        volume_change = df['Volume'].pct_change().rolling(20).std()
        
        # High correlation between volatility and volume suggests stress
        correlation = volatility.rolling(50).corr(volume_change)
        return correlation.fillna(0)
    
    def train_classifier(self, data, target_col='market_regime'):
        """Train market regime classifier"""
        # Prepare features and target
        features = data.drop([target_col, 'symbol'], axis=1, errors='ignore')
        target = data[target_col]
        
        # Store feature columns
        self.feature_columns = features.columns.tolist()
        
        # Handle missing values
        features = features.fillna(features.median())
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Encode labels
        target_encoded = self.label_encoder.fit_transform(target)
        
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Try different models
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            )
        }
        
        best_score = 0
        best_model = None
        best_model_name = None
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            # Cross-validation
            scores = []
            for train_idx, val_idx in tscv.split(features_scaled):
                X_train, X_val = features_scaled[train_idx], features_scaled[val_idx]
                y_train, y_val = target_encoded[train_idx], target_encoded[val_idx]
                
                model.fit(X_train, y_train)
                score = model.score(X_val, y_val)
                scores.append(score)
            
            avg_score = np.mean(scores)
            print(f"{name} CV Score: {avg_score:.4f} (+/- {np.std(scores)*2:.4f})")
            
            if avg_score > best_score:
                best_score = avg_score
                best_model = model
                best_model_name = name
        
        print(f"Best model: {best_model_name} with score: {best_score:.4f}")
        
        # Train best model on full dataset
        self.model = best_model
        self.model.fit(features_scaled, target_encoded)
        
        return self.model
    
    def evaluate_classifier(self, X_test, y_test):
        """Evaluate classifier performance"""
        # Scale test features
        X_test_scaled = self.scaler.transform(X_test)
        
        # Predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)
        
        # Encode test labels
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test_encoded, y_pred,
                                  target_names=self.label_encoder.classes_))
        
        # Confusion matrix
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test_encoded, y_pred))
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nTop 10 Most Important Features:")
            print(importance_df.head(10))
        
        return y_pred, y_pred_proba
    
    def save_classifier(self, model_path='ai_models/market_classifier'):
        """Save the trained classifier"""
        os.makedirs(model_path, exist_ok=True)
        
        # Save model and preprocessors
        joblib.dump(self.model, f'{model_path}/model.pkl')
        joblib.dump(self.scaler, f'{model_path}/scaler.pkl')
        joblib.dump(self.label_encoder, f'{model_path}/label_encoder.pkl')
        joblib.dump(self.feature_columns, f'{model_path}/feature_columns.pkl')
        
        # Save metadata
        metadata = {
            'model_type': type(self.model).__name__,
            'num_features': len(self.feature_columns),
            'num_classes': len(self.label_encoder.classes_),
            'classes': self.label_encoder.classes_.tolist(),
            'training_date': datetime.now().isoformat()
        }
        
        import json
        with open(f'{model_path}/metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Classifier saved to {model_path}")
    
    def load_classifier(self, model_path='ai_models/market_classifier'):
        """Load trained classifier"""
        self.model = joblib.load(f'{model_path}/model.pkl')
        self.scaler = joblib.load(f'{model_path}/scaler.pkl')
        self.label_encoder = joblib.load(f'{model_path}/label_encoder.pkl')
        self.feature_columns = joblib.load(f'{model_path}/feature_columns.pkl')
        
        print(f"Classifier loaded from {model_path}")
    
    def predict_market_regime(self, features_df):
        """Predict market regime for new data"""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Ensure features match training data
        features_df = features_df[self.feature_columns]
        features_df = features_df.fillna(features_df.median())
        
        # Scale features
        features_scaled = self.scaler.transform(features_df)
        
        # Predict
        prediction = self.model.predict(features_scaled)
        prediction_proba = self.model.predict_proba(features_scaled)
        
        # Decode prediction
        regime = self.label_encoder.inverse_transform(prediction)[0]
        confidence = np.max(prediction_proba)
        
        # Get probabilities for each class
        class_probabilities = {}
        for i, class_name in enumerate(self.label_encoder.classes_):
            class_probabilities[class_name] = float(prediction_proba[0][i])
        
        return {
            'regime': regime,
            'confidence': float(confidence),
            'probabilities': class_probabilities
        }

def main():
    """Main training function"""
    # Initialize trainer
    trainer = MarketClassifierTrainer()
    
    # Define market symbols for training
    symbols = [
        # US Indices
        'SPY', 'QQQ', 'IWM', 'DIA',
        # Sectors
        'XLF', 'XLK', 'XLE', 'XLV', 'XLI',
        # International
        'EFA', 'EEM', 'VGK', 'FXI',
        # Commodities
        'GLD', 'SLV', 'USO', 'UNG',
        # Bonds
        'TLT', 'IEF', 'HYG', 'LQD'
    ]
    
    print("Preparing market data...")
    data = trainer.prepare_market_data(symbols, period='5y', interval='1d')
    
    print(f"Data shape: {data.shape}")
    print(f"Market regime distribution:")
    print(data['market_regime'].value_counts())
    
    # Split data chronologically
    split_date = data.index[int(len(data) * 0.8)]
    train_data = data[data.index <= split_date]
    test_data = data[data.index > split_date]
    
    print("Training classifier...")
    trainer.train_classifier(train_data)
    
    print("Evaluating classifier...")
    test_features = test_data.drop(['market_regime', 'symbol'], axis=1, errors='ignore')
    test_target = test_data['market_regime']
    trainer.evaluate_classifier(test_features, test_target)
    
    print("Saving classifier...")
    trainer.save_classifier()
    
    print("Training completed!")

if __name__ == "__main__":
    main()
