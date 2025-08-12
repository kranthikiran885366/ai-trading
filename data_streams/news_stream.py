#!/usr/bin/env python3
"""
Real-time News Stream with Sentiment Analysis
"""

import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json
from dataclasses import dataclass
from transformers import pipeline
import feedparser
import re

logger = logging.getLogger(__name__)

@dataclass
class NewsItem:
    """News item structure"""
    title: str
    content: str
    source: str
    url: str
    timestamp: datetime
    symbols: List[str]
    sentiment_score: float
    sentiment_label: str
    relevance_score: float

class NewsStream:
    """Real-time news streaming with sentiment analysis"""
    
    def __init__(self, api_keys: Dict[str, str], kafka_producer=None):
        self.api_keys = api_keys
        self.kafka_producer = kafka_producer
        
        # News storage
        self.news_items = []
        self.sentiment_cache = {}
        
        # Sentiment analyzer
        self.sentiment_analyzer = None
        
        # News sources
        self.news_sources = {
            'newsapi': 'https://newsapi.org/v2/everything',
            'alpha_vantage': 'https://www.alphavantage.co/query',
            'rss_feeds': [
                'https://feeds.finance.yahoo.com/rss/2.0/headline',
                'https://www.cnbc.com/id/100003114/device/rss/rss.html',
                'https://feeds.bloomberg.com/markets/news.rss'
            ]
        }
        
        # Symbol keywords for relevance detection
        self.symbol_keywords = {
            'AAPL': ['apple', 'iphone', 'ipad', 'mac', 'tim cook'],
            'GOOGL': ['google', 'alphabet', 'android', 'youtube', 'sundar pichai'],
            'MSFT': ['microsoft', 'windows', 'azure', 'office', 'satya nadella'],
            'AMZN': ['amazon', 'aws', 'prime', 'alexa', 'jeff bezos'],
            'TSLA': ['tesla', 'elon musk', 'electric vehicle', 'model 3', 'model y'],
            'NVDA': ['nvidia', 'gpu', 'ai chip', 'jensen huang'],
            'META': ['meta', 'facebook', 'instagram', 'whatsapp', 'mark zuckerberg'],
            'BTCUSDT': ['bitcoin', 'btc', 'cryptocurrency', 'crypto'],
            'ETHUSDT': ['ethereum', 'eth', 'smart contract', 'defi']
        }
        
        self.is_running = False
        
        logger.info("News Stream initialized")
    
    async def start(self):
        """Start news streaming"""
        try:
            logger.info("Starting news streams...")
            self.is_running = True
            
            # Initialize sentiment analyzer
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                device=-1  # Use CPU
            )
            
            # Start news fetching tasks
            tasks = [
                self._fetch_newsapi_news(),
                self._fetch_alpha_vantage_news(),
                self._fetch_rss_news(),
                self._process_news_loop()
            ]
            
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            logger.error(f"Error starting news streams: {e}")
    
    async def stop(self):
        """Stop news streaming"""
        logger.info("Stopping news streams...")
        self.is_running = False
    
    async def _fetch_newsapi_news(self):
        """Fetch news from NewsAPI"""
        try:
            if not self.api_keys.get('newsapi'):
                logger.warning("NewsAPI key not provided")
                return
            
            while self.is_running:
                try:
                    # Financial keywords
                    keywords = "stock market OR trading OR finance OR economy OR earnings"
                    
                    params = {
                        'q': keywords,
                        'language': 'en',
                        'sortBy': 'publishedAt',
                        'pageSize': 50,
                        'apiKey': self.api_keys['newsapi']
                    }
                    
                    async with aiohttp.ClientSession() as session:
                        async with session.get(self.news_sources['newsapi'], params=params) as response:
                            if response.status == 200:
                                data = await response.json()
                                
                                for article in data.get('articles', []):
                                    await self._process_news_article(article, 'newsapi')
                            else:
                                logger.error(f"NewsAPI error: {response.status}")
                    
                    # Wait 10 minutes before next fetch
                    await asyncio.sleep(600)
                    
                except Exception as e:
                    logger.error(f"Error fetching NewsAPI news: {e}")
                    await asyncio.sleep(300)
                    
        except Exception as e:
            logger.error(f"Error in NewsAPI fetching: {e}")
    
    async def _fetch_alpha_vantage_news(self):
        """Fetch news from Alpha Vantage"""
        try:
            if not self.api_keys.get('alpha_vantage'):
                logger.warning("Alpha Vantage key not provided")
                return
            
            while self.is_running:
                try:
                    # Fetch news for each symbol
                    for symbol in self.symbol_keywords.keys():
                        if not symbol.endswith('USDT'):  # Skip crypto for Alpha Vantage
                            params = {
                                'function': 'NEWS_SENTIMENT',
                                'tickers': symbol,
                                'apikey': self.api_keys['alpha_vantage']
                            }
                            
                            async with aiohttp.ClientSession() as session:
                                async with session.get(self.news_sources['alpha_vantage'], params=params) as response:
                                    if response.status == 200:
                                        data = await response.json()
                                        
                                        for article in data.get('feed', []):
                                            await self._process_alpha_vantage_article(article, symbol)
                                    else:
                                        logger.error(f"Alpha Vantage error: {response.status}")
                            
                            # Rate limiting
                            await asyncio.sleep(12)  # Alpha Vantage allows 5 calls per minute
                    
                    # Wait 30 minutes before next full cycle
                    await asyncio.sleep(1800)
                    
                except Exception as e:
                    logger.error(f"Error fetching Alpha Vantage news: {e}")
                    await asyncio.sleep(600)
                    
        except Exception as e:
            logger.error(f"Error in Alpha Vantage fetching: {e}")
    
    async def _fetch_rss_news(self):
        """Fetch news from RSS feeds"""
        try:
            while self.is_running:
                try:
                    for feed_url in self.news_sources['rss_feeds']:
                        try:
                            # Parse RSS feed
                            feed = feedparser.parse(feed_url)
                            
                            for entry in feed.entries:
                                article = {
                                    'title': entry.title,
                                    'description': entry.get('description', ''),
                                    'url': entry.link,
                                    'publishedAt': entry.get('published', ''),
                                    'source': {'name': feed.feed.get('title', 'RSS')}
                                }
                                
                                await self._process_news_article(article, 'rss')
                                
                        except Exception as e:
                            logger.error(f"Error parsing RSS feed {feed_url}: {e}")
                    
                    # Wait 15 minutes before next RSS fetch
                    await asyncio.sleep(900)
                    
                except Exception as e:
                    logger.error(f"Error fetching RSS news: {e}")
                    await asyncio.sleep(300)
                    
        except Exception as e:
            logger.error(f"Error in RSS fetching: {e}")
    
    async def _process_news_article(self, article: Dict[str, Any], source: str):
        """Process a news article"""
        try:
            title = article.get('title', '')
            content = article.get('description', '') or article.get('content', '')
            url = article.get('url', '')
            
            if not title or not content:
                return
            
            # Parse timestamp
            timestamp_str = article.get('publishedAt', '')
            try:
                if timestamp_str:
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                else:
                    timestamp = datetime.now()
            except:
                timestamp = datetime.now()
            
            # Skip old news (older than 24 hours)
            if (datetime.now() - timestamp.replace(tzinfo=None)).total_seconds() > 86400:
                return
            
            # Detect relevant symbols
            relevant_symbols = self._detect_relevant_symbols(title + ' ' + content)
            
            if not relevant_symbols:
                return
            
            # Analyze sentiment
            sentiment_score, sentiment_label = await self._analyze_sentiment(title + ' ' + content)
            
            # Calculate relevance score
            relevance_score = self._calculate_relevance_score(title + ' ' + content, relevant_symbols)
            
            # Create news item
            news_item = NewsItem(
                title=title,
                content=content,
                source=source,
                url=url,
                timestamp=timestamp,
                symbols=relevant_symbols,
                sentiment_score=sentiment_score,
                sentiment_label=sentiment_label,
                relevance_score=relevance_score
            )
            
            # Store news item
            self.news_items.append(news_item)
            
            # Keep only recent news (last 1000 items)
            if len(self.news_items) > 1000:
                self.news_items = self.news_items[-500:]
            
            # Send to Kafka
            if self.kafka_producer:
                message = {
                    'type': 'news',
                    'title': title,
                    'symbols': relevant_symbols,
                    'sentiment_score': sentiment_score,
                    'sentiment_label': sentiment_label,
                    'relevance_score': relevance_score,
                    'timestamp': timestamp.isoformat(),
                    'source': source
                }
                
                self.kafka_producer.send('news_stream', message)
            
            logger.debug(f"Processed news: {title[:50]}... (sentiment: {sentiment_label})")
            
        except Exception as e:
            logger.error(f"Error processing news article: {e}")
    
    async def _process_alpha_vantage_article(self, article: Dict[str, Any], symbol: str):
        """Process Alpha Vantage news article"""
        try:
            title = article.get('title', '')
            content = article.get('summary', '')
            url = article.get('url', '')
            
            # Alpha Vantage provides sentiment
            sentiment_score = float(article.get('overall_sentiment_score', 0))
            sentiment_label = article.get('overall_sentiment_label', 'neutral')
            
            timestamp_str = article.get('time_published', '')
            try:
                timestamp = datetime.strptime(timestamp_str, '%Y%m%dT%H%M%S')
            except:
                timestamp = datetime.now()
            
            # Create news item
            news_item = NewsItem(
                title=title,
                content=content,
                source='alpha_vantage',
                url=url,
                timestamp=timestamp,
                symbols=[symbol],
                sentiment_score=sentiment_score,
                sentiment_label=sentiment_label,
                relevance_score=1.0  # Alpha Vantage pre-filters by symbol
            )
            
            self.news_items.append(news_item)
            
            # Send to Kafka
            if self.kafka_producer:
                message = {
                    'type': 'news',
                    'title': title,
                    'symbols': [symbol],
                    'sentiment_score': sentiment_score,
                    'sentiment_label': sentiment_label,
                    'relevance_score': 1.0,
                    'timestamp': timestamp.isoformat(),
                    'source': 'alpha_vantage'
                }
                
                self.kafka_producer.send('news_stream', message)
            
        except Exception as e:
            logger.error(f"Error processing Alpha Vantage article: {e}")
    
    def _detect_relevant_symbols(self, text: str) -> List[str]:
        """Detect relevant symbols in text"""
        try:
            text_lower = text.lower()
            relevant_symbols = []
            
            for symbol, keywords in self.symbol_keywords.items():
                for keyword in keywords:
                    if keyword.lower() in text_lower:
                        if symbol not in relevant_symbols:
                            relevant_symbols.append(symbol)
                        break
            
            return relevant_symbols
            
        except Exception as e:
            logger.error(f"Error detecting relevant symbols: {e}")
            return []
    
    async def _analyze_sentiment(self, text: str) -> tuple[float, str]:
        """Analyze sentiment of text"""
        try:
            # Check cache first
            text_hash = hash(text)
            if text_hash in self.sentiment_cache:
                return self.sentiment_cache[text_hash]
            
            if not self.sentiment_analyzer:
                return 0.0, 'neutral'
            
            # Truncate text if too long
            if len(text) > 512:
                text = text[:512]
            
            # Analyze sentiment
            result = self.sentiment_analyzer(text)[0]
            
            # Convert to score and label
            label = result['label'].lower()
            score = result['score']
            
            # Convert to standardized format
            if label == 'positive':
                sentiment_score = score
                sentiment_label = 'positive'
            elif label == 'negative':
                sentiment_score = -score
                sentiment_label = 'negative'
            else:
                sentiment_score = 0.0
                sentiment_label = 'neutral'
            
            # Cache result
            self.sentiment_cache[text_hash] = (sentiment_score, sentiment_label)
            
            # Keep cache size manageable
            if len(self.sentiment_cache) > 1000:
                # Remove oldest entries
                keys_to_remove = list(self.sentiment_cache.keys())[:500]
                for key in keys_to_remove:
                    del self.sentiment_cache[key]
            
            return sentiment_score, sentiment_label
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return 0.0, 'neutral'
    
    def _calculate_relevance_score(self, text: str, symbols: List[str]) -> float:
        """Calculate relevance score for symbols"""
        try:
            text_lower = text.lower()
            total_score = 0.0
            
            for symbol in symbols:
                symbol_score = 0.0
                keywords = self.symbol_keywords.get(symbol, [])
                
                for keyword in keywords:
                    # Count keyword occurrences
                    count = text_lower.count(keyword.lower())
                    symbol_score += count * 0.1
                
                # Bonus for symbol mention in title
                if any(keyword.lower() in text_lower[:100] for keyword in keywords):
                    symbol_score += 0.3
                
                total_score += min(symbol_score, 1.0)  # Cap at 1.0 per symbol
            
            return min(total_score / len(symbols), 1.0) if symbols else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating relevance score: {e}")
            return 0.5
    
    async def _process_news_loop(self):
        """Process news items and generate insights"""
        try:
            while self.is_running:
                # Clean old news items
                cutoff_time = datetime.now() - timedelta(hours=24)
                self.news_items = [
                    item for item in self.news_items 
                    if item.timestamp > cutoff_time
                ]
                
                # Generate sentiment summaries
                await self._generate_sentiment_summaries()
                
                # Wait 5 minutes
                await asyncio.sleep(300)
                
        except Exception as e:
            logger.error(f"Error in news processing loop: {e}")
    
    async def _generate_sentiment_summaries(self):
        """Generate sentiment summaries for symbols"""
        try:
            symbol_sentiments = {}
            
            # Calculate sentiment for each symbol
            for symbol in self.symbol_keywords.keys():
                relevant_news = [
                    item for item in self.news_items 
                    if symbol in item.symbols and 
                    (datetime.now() - item.timestamp).total_seconds() < 3600  # Last hour
                ]
                
                if relevant_news:
                    # Weighted average sentiment
                    total_weight = 0
                    weighted_sentiment = 0
                    
                    for news_item in relevant_news:
                        weight = news_item.relevance_score
                        weighted_sentiment += news_item.sentiment_score * weight
                        total_weight += weight
                    
                    avg_sentiment = weighted_sentiment / total_weight if total_weight > 0 else 0
                    
                    symbol_sentiments[symbol] = {
                        'sentiment_score': avg_sentiment,
                        'news_count': len(relevant_news),
                        'latest_news': relevant_news[-3:],  # Last 3 news items
                        'timestamp': datetime.now()
                    }
            
            self.symbol_sentiments = symbol_sentiments
            
        except Exception as e:
            logger.error(f"Error generating sentiment summaries: {e}")
    
    async def get_sentiment_analysis(self) -> Dict[str, Any]:
        """Get sentiment analysis for all symbols"""
        try:
            return getattr(self, 'symbol_sentiments', {})
            
        except Exception as e:
            logger.error(f"Error getting sentiment analysis: {e}")
            return {}
    
    def get_recent_news(self, symbol: str = None, limit: int = 10) -> List[NewsItem]:
        """Get recent news items"""
        try:
            if symbol:
                filtered_news = [
                    item for item in self.news_items 
                    if symbol in item.symbols
                ]
            else:
                filtered_news = self.news_items
            
            # Sort by timestamp (newest first)
            filtered_news.sort(key=lambda x: x.timestamp, reverse=True)
            
            return filtered_news[:limit]
            
        except Exception as e:
            logger.error(f"Error getting recent news: {e}")
            return []
    
    def get_news_stats(self) -> Dict[str, Any]:
        """Get news statistics"""
        try:
            total_news = len(self.news_items)
            
            # Count by sentiment
            positive_count = sum(1 for item in self.news_items if item.sentiment_label == 'positive')
            negative_count = sum(1 for item in self.news_items if item.sentiment_label == 'negative')
            neutral_count = total_news - positive_count - negative_count
            
            # Count by source
            source_counts = {}
            for item in self.news_items:
                source_counts[item.source] = source_counts.get(item.source, 0) + 1
            
            return {
                'total_news': total_news,
                'sentiment_distribution': {
                    'positive': positive_count,
                    'negative': negative_count,
                    'neutral': neutral_count
                },
                'source_distribution': source_counts,
                'cache_size': len(self.sentiment_cache),
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting news stats: {e}")
            return {}
