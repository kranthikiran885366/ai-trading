import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yfinance as yf
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

class NewsSentimentAnalyzer:
    def __init__(self):
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.finbert_analyzer = None
        self.news_sources = {
            'alpha_vantage': {
                'url': 'https://www.alphavantage.co/query',
                'api_key': None  # Set your API key
            },
            'newsapi': {
                'url': 'https://newsapi.org/v2/everything',
                'api_key': None  # Set your API key
            },
            'finnhub': {
                'url': 'https://finnhub.io/api/v1/news',
                'api_key': None  # Set your API key
            }
        }
        self.initialize_finbert()
        
    def initialize_finbert(self):
        """Initialize FinBERT model for financial sentiment analysis"""
        try:
            model_name = "ProsusAI/finbert"
            self.finbert_analyzer = pipeline(
                "sentiment-analysis",
                model=model_name,
                tokenizer=model_name,
                device=0 if torch.cuda.is_available() else -1
            )
            print("FinBERT model loaded successfully")
        except Exception as e:
            print(f"Could not load FinBERT model: {e}")
            self.finbert_analyzer = None
    
    def set_api_keys(self, alpha_vantage_key=None, newsapi_key=None, finnhub_key=None):
        """Set API keys for news sources"""
        if alpha_vantage_key:
            self.news_sources['alpha_vantage']['api_key'] = alpha_vantage_key
        if newsapi_key:
            self.news_sources['newsapi']['api_key'] = newsapi_key
        if finnhub_key:
            self.news_sources['finnhub']['api_key'] = finnhub_key
    
    def fetch_news_alpha_vantage(self, symbols, limit=50):
        """Fetch news from Alpha Vantage"""
        if not self.news_sources['alpha_vantage']['api_key']:
            return []
        
        all_news = []
        
        for symbol in symbols:
            try:
                params = {
                    'function': 'NEWS_SENTIMENT',
                    'tickers': symbol,
                    'apikey': self.news_sources['alpha_vantage']['api_key'],
                    'limit': limit
                }
                
                response = requests.get(self.news_sources['alpha_vantage']['url'], params=params)
                data = response.json()
                
                if 'feed' in data:
                    for article in data['feed']:
                        news_item = {
                            'symbol': symbol,
                            'title': article.get('title', ''),
                            'summary': article.get('summary', ''),
                            'url': article.get('url', ''),
                            'time_published': article.get('time_published', ''),
                            'source': 'alpha_vantage',
                            'sentiment_score': article.get('overall_sentiment_score', 0),
                            'sentiment_label': article.get('overall_sentiment_label', 'neutral')
                        }
                        all_news.append(news_item)
                
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                print(f"Error fetching Alpha Vantage news for {symbol}: {e}")
                continue
        
        return all_news
    
    def fetch_news_newsapi(self, symbols, days_back=7):
        """Fetch news from NewsAPI"""
        if not self.news_sources['newsapi']['api_key']:
            return []
        
        all_news = []
        from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        for symbol in symbols:
            try:
                # Get company name for better search
                ticker = yf.Ticker(symbol)
                info = ticker.info
                company_name = info.get('longName', symbol)
                
                params = {
                    'q': f'"{company_name}" OR "{symbol}"',
                    'from': from_date,
                    'sortBy': 'publishedAt',
                    'apiKey': self.news_sources['newsapi']['api_key'],
                    'language': 'en',
                    'pageSize': 50
                }
                
                response = requests.get(self.news_sources['newsapi']['url'], params=params)
                data = response.json()
                
                if data.get('status') == 'ok' and 'articles' in data:
                    for article in data['articles']:
                        news_item = {
                            'symbol': symbol,
                            'title': article.get('title', ''),
                            'summary': article.get('description', ''),
                            'content': article.get('content', ''),
                            'url': article.get('url', ''),
                            'time_published': article.get('publishedAt', ''),
                            'source': 'newsapi',
                            'author': article.get('author', ''),
                            'source_name': article.get('source', {}).get('name', '')
                        }
                        all_news.append(news_item)
                
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                print(f"Error fetching NewsAPI news for {symbol}: {e}")
                continue
        
        return all_news
    
    def fetch_news_finnhub(self, symbols):
        """Fetch news from Finnhub"""
        if not self.news_sources['finnhub']['api_key']:
            return []
        
        all_news = []
        
        for symbol in symbols:
            try:
                params = {
                    'symbol': symbol,
                    'token': self.news_sources['finnhub']['api_key']
                }
                
                response = requests.get(self.news_sources['finnhub']['url'], params=params)
                data = response.json()
                
                if isinstance(data, list):
                    for article in data:
                        news_item = {
                            'symbol': symbol,
                            'title': article.get('headline', ''),
                            'summary': article.get('summary', ''),
                            'url': article.get('url', ''),
                            'time_published': datetime.fromtimestamp(article.get('datetime', 0)).isoformat(),
                            'source': 'finnhub',
                            'category': article.get('category', ''),
                            'image': article.get('image', '')
                        }
                        all_news.append(news_item)
                
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                print(f"Error fetching Finnhub news for {symbol}: {e}")
                continue
        
        return all_news
    
    def fetch_all_news(self, symbols, days_back=7):
        """Fetch news from all available sources"""
        all_news = []
        
        # Fetch from all sources in parallel
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            
            # Alpha Vantage
            if self.news_sources['alpha_vantage']['api_key']:
                futures.append(executor.submit(self.fetch_news_alpha_vantage, symbols))
            
            # NewsAPI
            if self.news_sources['newsapi']['api_key']:
                futures.append(executor.submit(self.fetch_news_newsapi, symbols, days_back))
            
            # Finnhub
            if self.news_sources['finnhub']['api_key']:
                futures.append(executor.submit(self.fetch_news_finnhub, symbols))
            
            # Collect results
            for future in as_completed(futures):
                try:
                    news_data = future.result()
                    all_news.extend(news_data)
                except Exception as e:
                    print(f"Error in news fetching thread: {e}")
        
        # Remove duplicates based on title and URL
        seen = set()
        unique_news = []
        for news in all_news:
            identifier = (news.get('title', ''), news.get('url', ''))
            if identifier not in seen:
                seen.add(identifier)
                unique_news.append(news)
        
        print(f"Fetched {len(unique_news)} unique news articles")
        return unique_news
    
    def analyze_sentiment_textblob(self, text):
        """Analyze sentiment using TextBlob"""
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity  # -1 to 1
            subjectivity = blob.sentiment.subjectivity  # 0 to 1
            
            # Convert to label
            if polarity > 0.1:
                label = 'positive'
            elif polarity < -0.1:
                label = 'negative'
            else:
                label = 'neutral'
            
            return {
                'polarity': polarity,
                'subjectivity': subjectivity,
                'label': label,
                'confidence': abs(polarity)
            }
        except:
            return {'polarity': 0, 'subjectivity': 0, 'label': 'neutral', 'confidence': 0}
    
    def analyze_sentiment_vader(self, text):
        """Analyze sentiment using VADER"""
        try:
            scores = self.vader_analyzer.polarity_scores(text)
            
            # Determine label based on compound score
            compound = scores['compound']
            if compound >= 0.05:
                label = 'positive'
            elif compound <= -0.05:
                label = 'negative'
            else:
                label = 'neutral'
            
            return {
                'compound': compound,
                'positive': scores['pos'],
                'negative': scores['neg'],
                'neutral': scores['neu'],
                'label': label,
                'confidence': abs(compound)
            }
        except:
            return {'compound': 0, 'positive': 0, 'negative': 0, 'neutral': 1, 'label': 'neutral', 'confidence': 0}
    
    def analyze_sentiment_finbert(self, text):
        """Analyze sentiment using FinBERT"""
        if not self.finbert_analyzer:
            return {'label': 'neutral', 'confidence': 0}
        
        try:
            # Truncate text to model's max length
            max_length = 512
            if len(text) > max_length:
                text = text[:max_length]
            
            result = self.finbert_analyzer(text)[0]
            
            # Map FinBERT labels to our format
            label_mapping = {
                'LABEL_0': 'negative',
                'LABEL_1': 'neutral', 
                'LABEL_2': 'positive'
            }
            
            label = label_mapping.get(result['label'], result['label'].lower())
            confidence = result['score']
            
            return {
                'label': label,
                'confidence': confidence,
                'raw_label': result['label']
            }
        except Exception as e:
            print(f"FinBERT analysis error: {e}")
            return {'label': 'neutral', 'confidence': 0}
    
    def analyze_comprehensive_sentiment(self, text):
        """Analyze sentiment using all available methods"""
        # Clean text
        text = self.clean_text(text)
        
        if not text:
            return self.get_neutral_sentiment()
        
        # Analyze with all methods
        textblob_result = self.analyze_sentiment_textblob(text)
        vader_result = self.analyze_sentiment_vader(text)
        finbert_result = self.analyze_sentiment_finbert(text)
        
        # Combine results with weighted average
        weights = {
            'textblob': 0.2,
            'vader': 0.3,
            'finbert': 0.5  # Give more weight to financial-specific model
        }
        
        # Convert labels to numeric scores
        label_to_score = {'negative': -1, 'neutral': 0, 'positive': 1}
        
        textblob_score = label_to_score[textblob_result['label']] * textblob_result['confidence']
        vader_score = label_to_score[vader_result['label']] * vader_result['confidence']
        finbert_score = label_to_score[finbert_result['label']] * finbert_result['confidence']
        
        # Weighted average
        combined_score = (
            textblob_score * weights['textblob'] +
            vader_score * weights['vader'] +
            finbert_score * weights['finbert']
        )
        
        # Determine final label
        if combined_score > 0.1:
            final_label = 'positive'
        elif combined_score < -0.1:
            final_label = 'negative'
        else:
            final_label = 'neutral'
        
        return {
            'combined_score': combined_score,
            'final_label': final_label,
            'confidence': abs(combined_score),
            'textblob': textblob_result,
            'vader': vader_result,
            'finbert': finbert_result
        }
    
    def clean_text(self, text):
        """Clean and preprocess text"""
        if not text:
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\$$\$$,]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def get_neutral_sentiment(self):
        """Return neutral sentiment structure"""
        return {
            'combined_score': 0,
            'final_label': 'neutral',
            'confidence': 0,
            'textblob': {'polarity': 0, 'subjectivity': 0, 'label': 'neutral', 'confidence': 0},
            'vader': {'compound': 0, 'positive': 0, 'negative': 0, 'neutral': 1, 'label': 'neutral', 'confidence': 0},
            'finbert': {'label': 'neutral', 'confidence': 0}
        }
    
    def analyze_news_batch(self, news_articles):
        """Analyze sentiment for a batch of news articles"""
        results = []
        
        for article in news_articles:
            # Combine title and summary for analysis
            text_to_analyze = f"{article.get('title', '')} {article.get('summary', '')}"
            
            # Analyze sentiment
            sentiment = self.analyze_comprehensive_sentiment(text_to_analyze)
            
            # Add to article
            article_result = article.copy()
            article_result['sentiment_analysis'] = sentiment
            article_result['analyzed_at'] = datetime.now().isoformat()
            
            results.append(article_result)
        
        return results
    
    def calculate_symbol_sentiment(self, news_articles, symbol, time_decay=True):
        """Calculate overall sentiment for a symbol"""
        symbol_articles = [article for article in news_articles if article.get('symbol') == symbol]
        
        if not symbol_articles:
            return self.get_neutral_sentiment()
        
        sentiments = []
        weights = []
        
        current_time = datetime.now()
        
        for article in symbol_articles:
            sentiment = article.get('sentiment_analysis', {})
            if not sentiment:
                continue
            
            score = sentiment.get('combined_score', 0)
            confidence = sentiment.get('confidence', 0)
            
            # Calculate time-based weight
            weight = confidence
            if time_decay:
                try:
                    article_time = datetime.fromisoformat(article.get('time_published', '').replace('Z', '+00:00'))
                    hours_old = (current_time - article_time).total_seconds() / 3600
                    time_weight = np.exp(-hours_old / 24)  # Exponential decay over 24 hours
                    weight *= time_weight
                except:
                    weight *= 0.5  # Reduce weight for articles without valid timestamps
            
            sentiments.append(score)
            weights.append(weight)
        
        if not sentiments:
            return self.get_neutral_sentiment()
        
        # Calculate weighted average
        weights = np.array(weights)
        sentiments = np.array(sentiments)
        
        if weights.sum() == 0:
            weighted_sentiment = 0
        else:
            weighted_sentiment = np.average(sentiments, weights=weights)
        
        # Determine label
        if weighted_sentiment > 0.1:
            label = 'positive'
        elif weighted_sentiment < -0.1:
            label = 'negative'
        else:
            label = 'neutral'
        
        return {
            'symbol': symbol,
            'sentiment_score': weighted_sentiment,
            'sentiment_label': label,
            'confidence': abs(weighted_sentiment),
            'article_count': len(symbol_articles),
            'analyzed_articles': len(sentiments),
            'calculation_time': current_time.isoformat()
        }
    
    def get_market_sentiment_overview(self, symbols, days_back=7):
        """Get comprehensive market sentiment overview"""
        print(f"Fetching news for {len(symbols)} symbols...")
        
        # Fetch news
        all_news = self.fetch_all_news(symbols, days_back)
        
        if not all_news:
            print("No news articles found")
            return {}
        
        print(f"Analyzing sentiment for {len(all_news)} articles...")
        
        # Analyze sentiment
        analyzed_news = self.analyze_news_batch(all_news)
        
        # Calculate sentiment for each symbol
        symbol_sentiments = {}
        for symbol in symbols:
            symbol_sentiment = self.calculate_symbol_sentiment(analyzed_news, symbol)
            symbol_sentiments[symbol] = symbol_sentiment
        
        # Calculate overall market sentiment
        all_scores = [s['sentiment_score'] for s in symbol_sentiments.values() if s['article_count'] > 0]
        
        if all_scores:
            market_sentiment = np.mean(all_scores)
            market_label = 'positive' if market_sentiment > 0.1 else 'negative' if market_sentiment < -0.1 else 'neutral'
        else:
            market_sentiment = 0
            market_label = 'neutral'
        
        return {
            'market_overview': {
                'overall_sentiment': market_sentiment,
                'overall_label': market_label,
                'total_articles': len(analyzed_news),
                'symbols_analyzed': len([s for s in symbol_sentiments.values() if s['article_count'] > 0]),
                'analysis_time': datetime.now().isoformat()
            },
            'symbol_sentiments': symbol_sentiments,
            'analyzed_articles': analyzed_news
        }
    
    def get_sentiment_signals(self, symbols, threshold=0.3):
        """Generate trading signals based on sentiment analysis"""
        sentiment_overview = self.get_market_sentiment_overview(symbols)
        
        signals = []
        
        for symbol, sentiment_data in sentiment_overview['symbol_sentiments'].items():
            if sentiment_data['article_count'] < 2:  # Need minimum articles
                continue
            
            score = sentiment_data['sentiment_score']
            confidence = sentiment_data['confidence']
            
            # Generate signal if sentiment is strong enough
            if abs(score) > threshold and confidence > 0.5:
                signal_type = 'BUY' if score > 0 else 'SELL'
                
                signal = {
                    'symbol': symbol,
                    'signal': signal_type,
                    'sentiment_score': score,
                    'confidence': confidence,
                    'article_count': sentiment_data['article_count'],
                    'reasoning': f"Strong {sentiment_data['sentiment_label']} sentiment ({score:.3f})",
                    'timestamp': datetime.now().isoformat()
                }
                
                signals.append(signal)
        
        return signals

def main():
    """Test the sentiment analyzer"""
    analyzer = NewsSentimentAnalyzer()
    
    # Set your API keys here
    analyzer.set_api_keys(
        alpha_vantage_key="YOUR_ALPHA_VANTAGE_KEY",
        newsapi_key="YOUR_NEWSAPI_KEY",
        finnhub_key="YOUR_FINNHUB_KEY"
    )
    
    # Test symbols
    symbols = ['AAPL', 'GOOGL', 'TSLA', 'MSFT', 'AMZN']
    
    # Get sentiment overview
    sentiment_overview = analyzer.get_market_sentiment_overview(symbols, days_back=3)
    
    print("Market Sentiment Overview:")
    print(f"Overall Sentiment: {sentiment_overview['market_overview']['overall_sentiment']:.3f}")
    print(f"Overall Label: {sentiment_overview['market_overview']['overall_label']}")
    print(f"Total Articles: {sentiment_overview['market_overview']['total_articles']}")
    
    print("\nSymbol Sentiments:")
    for symbol, data in sentiment_overview['symbol_sentiments'].items():
        print(f"{symbol}: {data['sentiment_score']:.3f} ({data['sentiment_label']}) - {data['article_count']} articles")
    
    # Get sentiment signals
    signals = analyzer.get_sentiment_signals(symbols)
    print(f"\nGenerated {len(signals)} sentiment signals:")
    for signal in signals:
        print(f"{signal['symbol']}: {signal['signal']} (confidence: {signal['confidence']:.3f})")

if __name__ == "__main__":
    main()
