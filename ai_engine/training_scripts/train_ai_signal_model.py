import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import yfinance as yf
import joblib
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class AISignalModelTrainer:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        self.lookback_window = 60  # 60 time periods for LSTM
        
    def prepare_data(self, symbols, period='2y', interval='1h'):
        """Prepare training data from multiple symbols"""
        all_data = []
        
        for symbol in symbols:
            print(f"Processing {symbol}...")
            try:
                # Download data
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=period, interval=interval)
                
                if df.empty:
                    continue
                
                # Calculate features
                features_df = self._calculate_features(df)
                
                # Create labels
                labels = self._create_labels(df)
                
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
    
    def _calculate_features(self, df):
        """Calculate comprehensive technical features"""
        features = pd.DataFrame(index=df.index)
        
        # Price features
        features['open'] = df['Open']
        features['high'] = df['High']
        features['low'] = df['Low']
        features['close'] = df['Close']
        features['volume'] = df['Volume']
        
        # Returns
        features['returns_1'] = df['Close'].pct_change(1)
        features['returns_5'] = df['Close'].pct_change(5)
        features['returns_10'] = df['Close'].pct_change(10)
        
        # Moving averages
        for period in [5, 10, 20, 50]:
            features[f'sma_{period}'] = df['Close'].rolling(period).mean()
            features[f'ema_{period}'] = df['Close'].ewm(span=period).mean()
            features[f'price_sma_{period}_ratio'] = df['Close'] / features[f'sma_{period}']
        
        # Volatility
        features['volatility_10'] = df['Close'].rolling(10).std()
        features['volatility_20'] = df['Close'].rolling(20).std()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        features['macd'] = exp1 - exp2
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_histogram'] = features['macd'] - features['macd_signal']
        
        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        bb_ma = df['Close'].rolling(bb_period).mean()
        bb_std_dev = df['Close'].rolling(bb_period).std()
        features['bb_upper'] = bb_ma + (bb_std_dev * bb_std)
        features['bb_lower'] = bb_ma - (bb_std_dev * bb_std)
        features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / bb_ma
        features['bb_position'] = (df['Close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
        
        # Volume indicators
        features['volume_sma_10'] = df['Volume'].rolling(10).mean()
        features['volume_ratio'] = df['Volume'] / features['volume_sma_10']
        
        # Price patterns
        features['doji'] = self._detect_doji(df)
        features['hammer'] = self._detect_hammer(df)
        
        # Support/Resistance
        features['support'] = df['Low'].rolling(20).min()
        features['resistance'] = df['High'].rolling(20).max()
        features['support_distance'] = (df['Close'] - features['support']) / df['Close']
        features['resistance_distance'] = (features['resistance'] - df['Close']) / df['Close']
        
        # Time features
        features['hour'] = df.index.hour
        features['day_of_week'] = df.index.dayofweek
        
        return features
    
    def _create_labels(self, df, future_periods=5, threshold=0.02):
        """Create trading labels based on future price movements"""
        labels = pd.Series(index=df.index, name='signal')
        
        # Calculate future returns
        future_returns = df['Close'].shift(-future_periods) / df['Close'] - 1
        
        # Create labels: 0=HOLD, 1=BUY, 2=SELL
        labels[:] = 0  # Default to HOLD
        labels[future_returns > threshold] = 1  # BUY
        labels[future_returns < -threshold] = 2  # SELL
        
        return labels
    
    def _detect_doji(self, df):
        """Detect Doji candlestick pattern"""
        body = abs(df['Close'] - df['Open'])
        range_val = df['High'] - df['Low']
        return (body / range_val < 0.1).astype(int)
    
    def _detect_hammer(self, df):
        """Detect Hammer candlestick pattern"""
        body = abs(df['Close'] - df['Open'])
        upper_shadow = df['High'] - np.maximum(df['Open'], df['Close'])
        lower_shadow = np.minimum(df['Open'], df['Close']) - df['Low']
        
        hammer_condition = (
            (lower_shadow > 2 * body) & 
            (upper_shadow < 0.5 * body) & 
            (body > 0)
        )
        return hammer_condition.astype(int)
    
    def create_sequences(self, data, target_col='signal'):
        """Create sequences for LSTM training"""
        features = data.drop([target_col, 'symbol'], axis=1, errors='ignore')
        targets = data[target_col]
        
        # Store feature columns
        self.feature_columns = features.columns.tolist()
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Create sequences
        X, y = [], []
        for i in range(self.lookback_window, len(features_scaled)):
            X.append(features_scaled[i-self.lookback_window:i])
            y.append(targets.iloc[i])
        
        X = np.array(X)
        y = np.array(y)
        
        # Encode labels
        y = self.label_encoder.fit_transform(y)
        
        return X, y
    
    def build_model(self, input_shape, num_classes=3):
        """Build advanced neural network model"""
        model = Sequential([
            # CNN layers for pattern recognition
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
            BatchNormalization(),
            Conv1D(filters=64, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Dropout(0.2),
            
            # LSTM layers for sequence learning
            LSTM(100, return_sequences=True),
            Dropout(0.3),
            LSTM(50, return_sequences=False),
            Dropout(0.3),
            
            # Dense layers for classification
            Dense(50, activation='relu'),
            BatchNormalization(),
            Dropout(0.4),
            Dense(25, activation='relu'),
            Dropout(0.2),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, X, y, validation_split=0.2, epochs=100, batch_size=32):
        """Train the model with advanced techniques"""
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                'best_signal_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Build model
        self.model = self.build_model((X.shape[1], X.shape[2]), len(np.unique(y)))
        
        print("Model Architecture:")
        self.model.summary()
        
        # Train model
        history = self.model.fit(
            X, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
            shuffle=False  # Important for time series
        )
        
        return history
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        # Predictions
        y_pred_proba = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                  target_names=['HOLD', 'BUY', 'SELL']))
        
        # Confusion matrix
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        # Calculate accuracy by class
        for i, class_name in enumerate(['HOLD', 'BUY', 'SELL']):
            class_mask = y_test == i
            if np.sum(class_mask) > 0:
                class_accuracy = np.mean(y_pred[class_mask] == y_test[class_mask])
                print(f"{class_name} Accuracy: {class_accuracy:.4f}")
        
        return y_pred, y_pred_proba
    
    def save_model(self, model_path='ai_models/signal_model'):
        """Save the trained model and preprocessors"""
        os.makedirs(model_path, exist_ok=True)
        
        # Save model
        self.model.save(f'{model_path}/model.h5')
        
        # Save preprocessors
        joblib.dump(self.scaler, f'{model_path}/scaler.pkl')
        joblib.dump(self.label_encoder, f'{model_path}/label_encoder.pkl')
        joblib.dump(self.feature_columns, f'{model_path}/feature_columns.pkl')
        
        # Save metadata
        metadata = {
            'lookback_window': self.lookback_window,
            'num_features': len(self.feature_columns),
            'num_classes': len(self.label_encoder.classes_),
            'classes': self.label_encoder.classes_.tolist(),
            'training_date': datetime.now().isoformat()
        }
        
        import json
        with open(f'{model_path}/metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path='ai_models/signal_model'):
        """Load trained model and preprocessors"""
        self.model = tf.keras.models.load_model(f'{model_path}/model.h5')
        self.scaler = joblib.load(f'{model_path}/scaler.pkl')
        self.label_encoder = joblib.load(f'{model_path}/label_encoder.pkl')
        self.feature_columns = joblib.load(f'{model_path}/feature_columns.pkl')
        
        print(f"Model loaded from {model_path}")
    
    def predict_signal(self, features_df):
        """Predict trading signal for new data"""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Ensure features match training data
        features_df = features_df[self.feature_columns]
        
        # Scale features
        features_scaled = self.scaler.transform(features_df)
        
        # Create sequence
        if len(features_scaled) < self.lookback_window:
            raise ValueError(f"Need at least {self.lookback_window} data points")
        
        sequence = features_scaled[-self.lookback_window:].reshape(1, self.lookback_window, -1)
        
        # Predict
        prediction_proba = self.model.predict(sequence, verbose=0)
        prediction = np.argmax(prediction_proba, axis=1)[0]
        
        # Decode prediction
        signal = self.label_encoder.inverse_transform([prediction])[0]
        confidence = np.max(prediction_proba)
        
        return {
            'signal': signal,
            'confidence': float(confidence),
            'probabilities': {
                'HOLD': float(prediction_proba[0][0]),
                'BUY': float(prediction_proba[0][1]),
                'SELL': float(prediction_proba[0][2])
            }
        }

def main():
    """Main training function"""
    # Initialize trainer
    trainer = AISignalModelTrainer()
    
    # Define symbols to train on
    symbols = [
        'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA',
        'NVDA', 'META', 'NFLX', 'AMD', 'CRM',
        'SPY', 'QQQ', 'IWM', 'GLD', 'SLV'
    ]
    
    print("Preparing training data...")
    data = trainer.prepare_data(symbols, period='2y', interval='1h')
    
    print("Creating sequences...")
    X, y = trainer.create_sequences(data)
    
    print(f"Training data shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Label distribution: {np.bincount(y)}")
    
    # Split data
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print("Training model...")
    history = trainer.train_model(X_train, y_train, epochs=50, batch_size=64)
    
    print("Evaluating model...")
    trainer.evaluate_model(X_test, y_test)
    
    print("Saving model...")
    trainer.save_model()
    
    print("Training completed!")

if __name__ == "__main__":
    main()
