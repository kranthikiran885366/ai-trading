import pandas as pd
import numpy as np
import talib
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import yfinance as yf
import requests
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class FeatureExtractor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.price_scaler = MinMaxScaler()
        self.volume_scaler = MinMaxScaler()
        
    def extract_technical_features(self, df):
        """Extract comprehensive technical indicators"""
        features = pd.DataFrame(index=df.index)
        
        # Price-based features
        features['open'] = df['Open']
        features['high'] = df['High']
        features['low'] = df['Low']
        features['close'] = df['Close']
        features['volume'] = df['Volume']
        
        # Basic price features
        features['hl_pct'] = (df['High'] - df['Low']) / df['Close'] * 100
        features['pc_pct'] = (df['Close'] - df['Open']) / df['Open'] * 100
        
        # Moving Averages
        for period in [5, 10, 20, 50, 100, 200]:
            features[f'sma_{period}'] = talib.SMA(df['Close'], timeperiod=period)
            features[f'ema_{period}'] = talib.EMA(df['Close'], timeperiod=period)
            features[f'price_sma_{period}_ratio'] = df['Close'] / features[f'sma_{period}']
            features[f'price_ema_{period}_ratio'] = df['Close'] / features[f'ema_{period}']
        
        # Momentum Indicators
        features['rsi_14'] = talib.RSI(df['Close'], timeperiod=14)
        features['rsi_21'] = talib.RSI(df['Close'], timeperiod=21)
        features['rsi_7'] = talib.RSI(df['Close'], timeperiod=7)
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(df['Close'])
        features['macd'] = macd
        features['macd_signal'] = macd_signal
        features['macd_histogram'] = macd_hist
        features['macd_signal_ratio'] = macd / macd_signal
        
        # Stochastic
        slowk, slowd = talib.STOCH(df['High'], df['Low'], df['Close'])
        features['stoch_k'] = slowk
        features['stoch_d'] = slowd
        features['stoch_k_d_diff'] = slowk - slowd
        
        # Williams %R
        features['williams_r'] = talib.WILLR(df['High'], df['Low'], df['Close'])
        
        # Commodity Channel Index
        features['cci'] = talib.CCI(df['High'], df['Low'], df['Close'])
        
        # Average True Range
        features['atr_14'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
        features['atr_21'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=21)
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(df['Close'])
        features['bb_upper'] = bb_upper
        features['bb_middle'] = bb_middle
        features['bb_lower'] = bb_lower
        features['bb_width'] = (bb_upper - bb_lower) / bb_middle
        features['bb_position'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower)
        
        # Parabolic SAR
        features['sar'] = talib.SAR(df['High'], df['Low'])
        features['sar_signal'] = np.where(df['Close'] > features['sar'], 1, -1)
        
        # Volume Indicators
        features['obv'] = talib.OBV(df['Close'], df['Volume'])
        features['ad'] = talib.AD(df['High'], df['Low'], df['Close'], df['Volume'])
        features['adosc'] = talib.ADOSC(df['High'], df['Low'], df['Close'], df['Volume'])
        
        # Volume Moving Averages
        features['volume_sma_10'] = talib.SMA(df['Volume'], timeperiod=10)
        features['volume_sma_20'] = talib.SMA(df['Volume'], timeperiod=20)
        features['volume_ratio'] = df['Volume'] / features['volume_sma_20']
        
        # Price Patterns
        features['doji'] = talib.CDLDOJI(df['Open'], df['High'], df['Low'], df['Close'])
        features['hammer'] = talib.CDLHAMMER(df['Open'], df['High'], df['Low'], df['Close'])
        features['hanging_man'] = talib.CDLHANGINGMAN(df['Open'], df['High'], df['Low'], df['Close'])
        features['shooting_star'] = talib.CDLSHOOTINGSTAR(df['Open'], df['High'], df['Low'], df['Close'])
        features['engulfing'] = talib.CDLENGULFING(df['Open'], df['High'], df['Low'], df['Close'])
        
        # Trend Indicators
        features['adx'] = talib.ADX(df['High'], df['Low'], df['Close'])
        features['plus_di'] = talib.PLUS_DI(df['High'], df['Low'], df['Close'])
        features['minus_di'] = talib.MINUS_DI(df['High'], df['Low'], df['Close'])
        
        # Volatility
        features['volatility_10'] = df['Close'].rolling(10).std()
        features['volatility_20'] = df['Close'].rolling(20).std()
        features['volatility_ratio'] = features['volatility_10'] / features['volatility_20']
        
        # Price Change Features
        for period in [1, 2, 3, 5, 10]:
            features[f'price_change_{period}'] = df['Close'].pct_change(period)
            features[f'high_change_{period}'] = df['High'].pct_change(period)
            features[f'low_change_{period}'] = df['Low'].pct_change(period)
            features[f'volume_change_{period}'] = df['Volume'].pct_change(period)
        
        # Support and Resistance Levels
        features['support_level'] = df['Low'].rolling(20).min()
        features['resistance_level'] = df['High'].rolling(20).max()
        features['support_distance'] = (df['Close'] - features['support_level']) / df['Close']
        features['resistance_distance'] = (features['resistance_level'] - df['Close']) / df['Close']
        
        # Market Structure
        features['higher_high'] = (df['High'] > df['High'].shift(1)).astype(int)
        features['lower_low'] = (df['Low'] < df['Low'].shift(1)).astype(int)
        features['inside_bar'] = ((df['High'] < df['High'].shift(1)) & 
                                 (df['Low'] > df['Low'].shift(1))).astype(int)
        
        return features
    
    def extract_market_microstructure_features(self, df):
        """Extract market microstructure features"""
        features = pd.DataFrame(index=df.index)
        
        # Bid-Ask Spread (simulated from OHLC)
        features['spread'] = (df['High'] - df['Low']) / df['Close']
        features['spread_ma'] = features['spread'].rolling(20).mean()
        features['spread_ratio'] = features['spread'] / features['spread_ma']
        
        # Price Impact
        features['price_impact'] = abs(df['Close'] - df['Open']) / df['Volume']
        
        # Tick Direction
        features['tick_direction'] = np.sign(df['Close'].diff())
        features['tick_runs'] = self._calculate_runs(features['tick_direction'])
        
        # Volatility Clustering
        returns = df['Close'].pct_change()
        features['volatility_clustering'] = returns.rolling(20).std() / returns.rolling(60).std()
        
        return features
    
    def extract_sentiment_features(self, symbol, news_data=None):
        """Extract sentiment-based features"""
        features = {}
        
        if news_data:
            # News sentiment analysis
            from textblob import TextBlob
            
            sentiments = []
            for article in news_data:
                blob = TextBlob(article.get('title', '') + ' ' + article.get('description', ''))
                sentiments.append(blob.sentiment.polarity)
            
            features['news_sentiment'] = np.mean(sentiments) if sentiments else 0
            features['news_sentiment_std'] = np.std(sentiments) if sentiments else 0
            features['news_count'] = len(sentiments)
        else:
            features['news_sentiment'] = 0
            features['news_sentiment_std'] = 0
            features['news_count'] = 0
        
        # Social media sentiment (placeholder for real implementation)
        features['social_sentiment'] = 0
        features['social_volume'] = 0
        
        return features
    
    def extract_macro_features(self):
        """Extract macroeconomic features"""
        features = {}
        
        try:
            # VIX (Fear Index)
            vix = yf.download('^VIX', period='5d', interval='1d')
            features['vix'] = vix['Close'].iloc[-1] if not vix.empty else 20
            features['vix_change'] = vix['Close'].pct_change().iloc[-1] if len(vix) > 1 else 0
            
            # Dollar Index
            dxy = yf.download('DX-Y.NYB', period='5d', interval='1d')
            features['dxy'] = dxy['Close'].iloc[-1] if not dxy.empty else 100
            features['dxy_change'] = dxy['Close'].pct_change().iloc[-1] if len(dxy) > 1 else 0
            
            # 10-Year Treasury Yield
            tnx = yf.download('^TNX', period='5d', interval='1d')
            features['tnx'] = tnx['Close'].iloc[-1] if not tnx.empty else 2.5
            features['tnx_change'] = tnx['Close'].pct_change().iloc[-1] if len(tnx) > 1 else 0
            
            # Gold
            gold = yf.download('GC=F', period='5d', interval='1d')
            features['gold'] = gold['Close'].iloc[-1] if not gold.empty else 2000
            features['gold_change'] = gold['Close'].pct_change().iloc[-1] if len(gold) > 1 else 0
            
        except Exception as e:
            print(f"Error fetching macro data: {e}")
            features = {
                'vix': 20, 'vix_change': 0,
                'dxy': 100, 'dxy_change': 0,
                'tnx': 2.5, 'tnx_change': 0,
                'gold': 2000, 'gold_change': 0
            }
        
        return features
    
    def extract_time_features(self, df):
        """Extract time-based features"""
        features = pd.DataFrame(index=df.index)
        
        # Time components
        features['hour'] = df.index.hour
        features['day_of_week'] = df.index.dayofweek
        features['day_of_month'] = df.index.day
        features['month'] = df.index.month
        features['quarter'] = df.index.quarter
        
        # Market session indicators
        features['pre_market'] = ((features['hour'] >= 4) & (features['hour'] < 9.5)).astype(int)
        features['market_open'] = ((features['hour'] >= 9.5) & (features['hour'] < 16)).astype(int)
        features['after_market'] = ((features['hour'] >= 16) & (features['hour'] < 20)).astype(int)
        
        # Special trading days
        features['monday'] = (features['day_of_week'] == 0).astype(int)
        features['friday'] = (features['day_of_week'] == 4).astype(int)
        features['month_end'] = (features['day_of_month'] > 25).astype(int)
        features['quarter_end'] = ((features['month'] % 3 == 0) & (features['month_end'] == 1)).astype(int)
        
        return features
    
    def create_feature_matrix(self, symbol, period='1y', interval='1h', include_sentiment=True):
        """Create complete feature matrix for a symbol"""
        try:
            # Download price data
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                raise ValueError(f"No data available for {symbol}")
            
            # Extract all features
            tech_features = self.extract_technical_features(df)
            micro_features = self.extract_market_microstructure_features(df)
            time_features = self.extract_time_features(df)
            
            # Combine features
            feature_matrix = pd.concat([tech_features, micro_features, time_features], axis=1)
            
            # Add macro features (same for all symbols)
            macro_features = self.extract_macro_features()
            for key, value in macro_features.items():
                feature_matrix[key] = value
            
            # Add sentiment features if requested
            if include_sentiment:
                sentiment_features = self.extract_sentiment_features(symbol)
                for key, value in sentiment_features.items():
                    feature_matrix[key] = value
            
            # Forward fill and backward fill NaN values
            feature_matrix = feature_matrix.fillna(method='ffill').fillna(method='bfill')
            
            # Remove any remaining NaN rows
            feature_matrix = feature_matrix.dropna()
            
            return feature_matrix
            
        except Exception as e:
            print(f"Error creating feature matrix for {symbol}: {e}")
            return pd.DataFrame()
    
    def _calculate_runs(self, series):
        """Calculate runs of consecutive same values"""
        runs = []
        current_run = 1
        
        for i in range(1, len(series)):
            if series.iloc[i] == series.iloc[i-1]:
                current_run += 1
            else:
                runs.append(current_run)
                current_run = 1
        runs.append(current_run)
        
        return pd.Series(runs, index=series.index[-len(runs):])
    
    def normalize_features(self, features, fit_scaler=True):
        """Normalize features using StandardScaler"""
        if fit_scaler:
            normalized = self.scaler.fit_transform(features)
        else:
            normalized = self.scaler.transform(features)
        
        return pd.DataFrame(normalized, index=features.index, columns=features.columns)
    
    def get_feature_importance(self, features, target):
        """Calculate feature importance using mutual information"""
        from sklearn.feature_selection import mutual_info_regression
        
        # Remove NaN values
        clean_data = pd.concat([features, target], axis=1).dropna()
        X = clean_data.iloc[:, :-1]
        y = clean_data.iloc[:, -1]
        
        # Calculate mutual information
        mi_scores = mutual_info_regression(X, y)
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': mi_scores
        }).sort_values('importance', ascending=False)
        
        return importance_df

if __name__ == "__main__":
    # Test the feature extractor
    extractor = FeatureExtractor()
    
    # Test with a popular stock
    symbol = "AAPL"
    features = extractor.create_feature_matrix(symbol, period='6mo', interval='1h')
    
    print(f"Feature matrix shape: {features.shape}")
    print(f"Features: {list(features.columns)}")
    print(f"Latest features:\n{features.tail()}")
