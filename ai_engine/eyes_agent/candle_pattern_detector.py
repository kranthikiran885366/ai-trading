#!/usr/bin/env python3
"""
Candle Pattern Detector - Advanced candlestick pattern recognition system
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
import talib
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class PatternType(Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"
    REVERSAL = "REVERSAL"
    CONTINUATION = "CONTINUATION"

class PatternStrength(Enum):
    WEAK = 1
    MODERATE = 2
    STRONG = 3
    VERY_STRONG = 4

@dataclass
class CandlePattern:
    """Candlestick pattern detection result"""
    name: str
    pattern_type: PatternType
    strength: PatternStrength
    confidence: float
    start_index: int
    end_index: int
    description: str
    trading_signal: str
    target_price: Optional[float]
    stop_loss: Optional[float]
    probability: float
    volume_confirmation: bool
    trend_context: str

class CandlePatternDetector:
    """Advanced candlestick pattern recognition and analysis system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Pattern detection parameters
        self.min_pattern_length = self.config.get('min_pattern_length', 1)
        self.max_pattern_length = self.config.get('max_pattern_length', 5)
        self.volume_threshold = self.config.get('volume_threshold', 1.5)
        self.trend_lookback = self.config.get('trend_lookback', 20)
        
        # Pattern definitions and their characteristics
        self.pattern_definitions = self._initialize_pattern_definitions()
        
        # Historical pattern performance
        self.pattern_performance = {}
        self.detection_history = []
        
        logger.info("Candle Pattern Detector initialized")
    
    def _initialize_pattern_definitions(self) -> Dict[str, Dict[str, Any]]:
        """Initialize candlestick pattern definitions"""
        return {
            'doji': {
                'function': self._detect_doji,
                'type': PatternType.NEUTRAL,
                'min_length': 1,
                'max_length': 1,
                'description': 'Indecision pattern with equal open and close',
                'reliability': 0.6
            },
            'hammer': {
                'function': self._detect_hammer,
                'type': PatternType.BULLISH,
                'min_length': 1,
                'max_length': 1,
                'description': 'Bullish reversal pattern with long lower shadow',
                'reliability': 0.7
            },
            'hanging_man': {
                'function': self._detect_hanging_man,
                'type': PatternType.BEARISH,
                'min_length': 1,
                'max_length': 1,
                'description': 'Bearish reversal pattern with long lower shadow',
                'reliability': 0.7
            },
            'shooting_star': {
                'function': self._detect_shooting_star,
                'type': PatternType.BEARISH,
                'min_length': 1,
                'max_length': 1,
                'description': 'Bearish reversal pattern with long upper shadow',
                'reliability': 0.75
            },
            'inverted_hammer': {
                'function': self._detect_inverted_hammer,
                'type': PatternType.BULLISH,
                'min_length': 1,
                'max_length': 1,
                'description': 'Bullish reversal pattern with long upper shadow',
                'reliability': 0.65
            },
            'engulfing_bullish': {
                'function': self._detect_bullish_engulfing,
                'type': PatternType.BULLISH,
                'min_length': 2,
                'max_length': 2,
                'description': 'Bullish reversal pattern where second candle engulfs first',
                'reliability': 0.8
            },
            'engulfing_bearish': {
                'function': self._detect_bearish_engulfing,
                'type': PatternType.BEARISH,
                'min_length': 2,
                'max_length': 2,
                'description': 'Bearish reversal pattern where second candle engulfs first',
                'reliability': 0.8
            },
            'morning_star': {
                'function': self._detect_morning_star,
                'type': PatternType.BULLISH,
                'min_length': 3,
                'max_length': 3,
                'description': 'Three-candle bullish reversal pattern',
                'reliability': 0.85
            },
            'evening_star': {
                'function': self._detect_evening_star,
                'type': PatternType.BEARISH,
                'min_length': 3,
                'max_length': 3,
                'description': 'Three-candle bearish reversal pattern',
                'reliability': 0.85
            },
            'three_white_soldiers': {
                'function': self._detect_three_white_soldiers,
                'type': PatternType.BULLISH,
                'min_length': 3,
                'max_length': 3,
                'description': 'Three consecutive bullish candles',
                'reliability': 0.8
            },
            'three_black_crows': {
                'function': self._detect_three_black_crows,
                'type': PatternType.BEARISH,
                'min_length': 3,
                'max_length': 3,
                'description': 'Three consecutive bearish candles',
                'reliability': 0.8
            },
            'harami_bullish': {
                'function': self._detect_bullish_harami,
                'type': PatternType.BULLISH,
                'min_length': 2,
                'max_length': 2,
                'description': 'Bullish reversal pattern with small candle inside large one',
                'reliability': 0.7
            },
            'harami_bearish': {
                'function': self._detect_bearish_harami,
                'type': PatternType.BEARISH,
                'min_length': 2,
                'max_length': 2,
                'description': 'Bearish reversal pattern with small candle inside large one',
                'reliability': 0.7
            },
            'dark_cloud_cover': {
                'function': self._detect_dark_cloud_cover,
                'type': PatternType.BEARISH,
                'min_length': 2,
                'max_length': 2,
                'description': 'Bearish reversal pattern with bearish candle covering bullish one',
                'reliability': 0.75
            },
            'piercing_pattern': {
                'function': self._detect_piercing_pattern,
                'type': PatternType.BULLISH,
                'min_length': 2,
                'max_length': 2,
                'description': 'Bullish reversal pattern with bullish candle piercing bearish one',
                'reliability': 0.75
            }
        }
    
    def detect_patterns(self, data: pd.DataFrame, symbol: str = None) -> List[CandlePattern]:
        """Detect all candlestick patterns in the given data"""
        try:
            if len(data) < 3:
                logger.warning("Insufficient data for pattern detection")
                return []
            
            patterns = []
            
            # Ensure required columns exist
            required_columns = ['Open', 'High', 'Low', 'Close']
            if not all(col in data.columns for col in required_columns):
                logger.error("Missing required OHLC columns")
                return []
            
            # Add volume if available
            has_volume = 'Volume' in data.columns
            
            # Calculate trend context
            trend_context = self._analyze_trend_context(data)
            
            # Detect each pattern type
            for pattern_name, pattern_info in self.pattern_definitions.items():
                try:
                    pattern_results = pattern_info['function'](data, has_volume)
                    
                    for result in pattern_results:
                        # Enhance pattern with additional analysis
                        enhanced_pattern = self._enhance_pattern(
                            result, pattern_name, pattern_info, data, trend_context
                        )
                        
                        if enhanced_pattern:
                            patterns.append(enhanced_pattern)
                            
                except Exception as e:
                    logger.error(f"Error detecting {pattern_name}: {e}")
                    continue
            
            # Sort patterns by confidence and recency
            patterns.sort(key=lambda x: (x.confidence, x.end_index), reverse=True)
            
            # Store detection history
            self._store_detection_history(symbol, patterns)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting patterns: {e}")
            return []
    
    def _detect_doji(self, data: pd.DataFrame, has_volume: bool) -> List[Dict[str, Any]]:
        """Detect Doji patterns"""
        patterns = []
        
        try:
            for i in range(len(data)):
                open_price = data['Open'].iloc[i]
                close_price = data['Close'].iloc[i]
                high_price = data['High'].iloc[i]
                low_price = data['Low'].iloc[i]
                
                # Calculate body and shadow sizes
                body_size = abs(close_price - open_price)
                total_range = high_price - low_price
                
                if total_range == 0:
                    continue
                
                # Doji criteria: very small body relative to total range
                body_ratio = body_size / total_range
                
                if body_ratio <= 0.1:  # Body is less than 10% of total range
                    patterns.append({
                        'start_index': i,
                        'end_index': i,
                        'confidence': 1.0 - body_ratio,  # Higher confidence for smaller body
                        'volume_confirmation': self._check_volume_confirmation(data, i, has_volume)
                    })
                    
        except Exception as e:
            logger.error(f"Error detecting doji: {e}")
        
        return patterns
    
    def _detect_hammer(self, data: pd.DataFrame, has_volume: bool) -> List[Dict[str, Any]]:
        """Detect Hammer patterns"""
        patterns = []
        
        try:
            for i in range(len(data)):
                open_price = data['Open'].iloc[i]
                close_price = data['Close'].iloc[i]
                high_price = data['High'].iloc[i]
                low_price = data['Low'].iloc[i]
                
                # Calculate components
                body_size = abs(close_price - open_price)
                lower_shadow = min(open_price, close_price) - low_price
                upper_shadow = high_price - max(open_price, close_price)
                total_range = high_price - low_price
                
                if total_range == 0:
                    continue
                
                # Hammer criteria
                if (lower_shadow >= 2 * body_size and  # Long lower shadow
                    upper_shadow <= body_size * 0.1 and  # Very small upper shadow
                    body_size > 0):  # Has a body
                    
                    confidence = min(1.0, lower_shadow / (body_size * 3))
                    
                    patterns.append({
                        'start_index': i,
                        'end_index': i,
                        'confidence': confidence,
                        'volume_confirmation': self._check_volume_confirmation(data, i, has_volume)
                    })
                    
        except Exception as e:
            logger.error(f"Error detecting hammer: {e}")
        
        return patterns
    
    def _detect_hanging_man(self, data: pd.DataFrame, has_volume: bool) -> List[Dict[str, Any]]:
        """Detect Hanging Man patterns"""
        patterns = []
        
        try:
            for i in range(len(data)):
                open_price = data['Open'].iloc[i]
                close_price = data['Close'].iloc[i]
                high_price = data['High'].iloc[i]
                low_price = data['Low'].iloc[i]
                
                # Calculate components
                body_size = abs(close_price - open_price)
                lower_shadow = min(open_price, close_price) - low_price
                upper_shadow = high_price - max(open_price, close_price)
                
                # Hanging man criteria (similar to hammer but in uptrend context)
                if (lower_shadow >= 2 * body_size and
                    upper_shadow <= body_size * 0.1 and
                    body_size > 0):
                    
                    # Check if we're in an uptrend (simplified check)
                    in_uptrend = self._check_uptrend_context(data, i)
                    
                    if in_uptrend:
                        confidence = min(1.0, lower_shadow / (body_size * 3))
                        
                        patterns.append({
                            'start_index': i,
                            'end_index': i,
                            'confidence': confidence,
                            'volume_confirmation': self._check_volume_confirmation(data, i, has_volume)
                        })
                        
        except Exception as e:
            logger.error(f"Error detecting hanging man: {e}")
        
        return patterns
    
    def _detect_shooting_star(self, data: pd.DataFrame, has_volume: bool) -> List[Dict[str, Any]]:
        """Detect Shooting Star patterns"""
        patterns = []
        
        try:
            for i in range(len(data)):
                open_price = data['Open'].iloc[i]
                close_price = data['Close'].iloc[i]
                high_price = data['High'].iloc[i]
                low_price = data['Low'].iloc[i]
                
                # Calculate components
                body_size = abs(close_price - open_price)
                lower_shadow = min(open_price, close_price) - low_price
                upper_shadow = high_price - max(open_price, close_price)
                
                # Shooting star criteria
                if (upper_shadow >= 2 * body_size and  # Long upper shadow
                    lower_shadow <= body_size * 0.1 and  # Very small lower shadow
                    body_size > 0):  # Has a body
                    
                    confidence = min(1.0, upper_shadow / (body_size * 3))
                    
                    patterns.append({
                        'start_index': i,
                        'end_index': i,
                        'confidence': confidence,
                        'volume_confirmation': self._check_volume_confirmation(data, i, has_volume)
                    })
                    
        except Exception as e:
            logger.error(f"Error detecting shooting star: {e}")
        
        return patterns
    
    def _detect_inverted_hammer(self, data: pd.DataFrame, has_volume: bool) -> List[Dict[str, Any]]:
        """Detect Inverted Hammer patterns"""
        patterns = []
        
        try:
            for i in range(len(data)):
                open_price = data['Open'].iloc[i]
                close_price = data['Close'].iloc[i]
                high_price = data['High'].iloc[i]
                low_price = data['Low'].iloc[i]
                
                # Calculate components
                body_size = abs(close_price - open_price)
                lower_shadow = min(open_price, close_price) - low_price
                upper_shadow = high_price - max(open_price, close_price)
                
                # Inverted hammer criteria (similar to shooting star but in downtrend context)
                if (upper_shadow >= 2 * body_size and
                    lower_shadow <= body_size * 0.1 and
                    body_size > 0):
                    
                    # Check if we're in a downtrend
                    in_downtrend = self._check_downtrend_context(data, i)
                    
                    if in_downtrend:
                        confidence = min(1.0, upper_shadow / (body_size * 3))
                        
                        patterns.append({
                            'start_index': i,
                            'end_index': i,
                            'confidence': confidence,
                            'volume_confirmation': self._check_volume_confirmation(data, i, has_volume)
                        })
                        
        except Exception as e:
            logger.error(f"Error detecting inverted hammer: {e}")
        
        return patterns
    
    def _detect_bullish_engulfing(self, data: pd.DataFrame, has_volume: bool) -> List[Dict[str, Any]]:
        """Detect Bullish Engulfing patterns"""
        patterns = []
        
        try:
            for i in range(1, len(data)):
                # First candle (bearish)
                open1 = data['Open'].iloc[i-1]
                close1 = data['Close'].iloc[i-1]
                
                # Second candle (bullish)
                open2 = data['Open'].iloc[i]
                close2 = data['Close'].iloc[i]
                
                # Bullish engulfing criteria
                if (close1 < open1 and  # First candle is bearish
                    close2 > open2 and  # Second candle is bullish
                    open2 < close1 and  # Second opens below first close
                    close2 > open1):    # Second closes above first open
                    
                    # Calculate engulfing strength
                    first_body = abs(close1 - open1)
                    second_body = abs(close2 - open2)
                    
                    if first_body > 0:
                        engulfing_ratio = second_body / first_body
                        confidence = min(1.0, engulfing_ratio / 2)
                        
                        patterns.append({
                            'start_index': i-1,
                            'end_index': i,
                            'confidence': confidence,
                            'volume_confirmation': self._check_volume_confirmation(data, i, has_volume)
                        })
                        
        except Exception as e:
            logger.error(f"Error detecting bullish engulfing: {e}")
        
        return patterns
    
    def _detect_bearish_engulfing(self, data: pd.DataFrame, has_volume: bool) -> List[Dict[str, Any]]:
        """Detect Bearish Engulfing patterns"""
        patterns = []
        
        try:
            for i in range(1, len(data)):
                # First candle (bullish)
                open1 = data['Open'].iloc[i-1]
                close1 = data['Close'].iloc[i-1]
                
                # Second candle (bearish)
                open2 = data['Open'].iloc[i]
                close2 = data['Close'].iloc[i]
                
                # Bearish engulfing criteria
                if (close1 > open1 and  # First candle is bullish
                    close2 < open2 and  # Second candle is bearish
                    open2 > close1 and  # Second opens above first close
                    close2 < open1):    # Second closes below first open
                    
                    # Calculate engulfing strength
                    first_body = abs(close1 - open1)
                    second_body = abs(close2 - open2)
                    
                    if first_body > 0:
                        engulfing_ratio = second_body / first_body
                        confidence = min(1.0, engulfing_ratio / 2)
                        
                        patterns.append({
                            'start_index': i-1,
                            'end_index': i,
                            'confidence': confidence,
                            'volume_confirmation': self._check_volume_confirmation(data, i, has_volume)
                        })
                        
        except Exception as e:
            logger.error(f"Error detecting bearish engulfing: {e}")
        
        return patterns
    
    def _detect_morning_star(self, data: pd.DataFrame, has_volume: bool) -> List[Dict[str, Any]]:
        """Detect Morning Star patterns"""
        patterns = []
        
        try:
            for i in range(2, len(data)):
                # Three candles
                open1, close1 = data['Open'].iloc[i-2], data['Close'].iloc[i-2]
                open2, close2 = data['Open'].iloc[i-1], data['Close'].iloc[i-1]
                open3, close3 = data['Open'].iloc[i], data['Close'].iloc[i]
                
                # Morning star criteria
                if (close1 < open1 and  # First candle is bearish
                    abs(close2 - open2) < abs(close1 - open1) * 0.3 and  # Second is small
                    close3 > open3 and  # Third candle is bullish
                    close2 < close1 and  # Gap down
                    open3 > close2 and  # Gap up
                    close3 > (open1 + close1) / 2):  # Third closes above midpoint of first
                    
                    # Calculate pattern strength
                    first_body = abs(close1 - open1)
                    third_body = abs(close3 - open3)
                    
                    if first_body > 0:
                        strength_ratio = third_body / first_body
                        confidence = min(1.0, strength_ratio)
                        
                        patterns.append({
                            'start_index': i-2,
                            'end_index': i,
                            'confidence': confidence,
                            'volume_confirmation': self._check_volume_confirmation(data, i, has_volume)
                        })
                        
        except Exception as e:
            logger.error(f"Error detecting morning star: {e}")
        
        return patterns
    
    def _detect_evening_star(self, data: pd.DataFrame, has_volume: bool) -> List[Dict[str, Any]]:
        """Detect Evening Star patterns"""
        patterns = []
        
        try:
            for i in range(2, len(data)):
                # Three candles
                open1, close1 = data['Open'].iloc[i-2], data['Close'].iloc[i-2]
                open2, close2 = data['Open'].iloc[i-1], data['Close'].iloc[i-1]
                open3, close3 = data['Open'].iloc[i], data['Close'].iloc[i]
                
                # Evening star criteria
                if (close1 > open1 and  # First candle is bullish
                    abs(close2 - open2) < abs(close1 - open1) * 0.3 and  # Second is small
                    close3 < open3 and  # Third candle is bearish
                    close2 > close1 and  # Gap up
                    open3 < close2 and  # Gap down
                    close3 < (open1 + close1) / 2):  # Third closes below midpoint of first
                    
                    # Calculate pattern strength
                    first_body = abs(close1 - open1)
                    third_body = abs(close3 - open3)
                    
                    if first_body > 0:
                        strength_ratio = third_body / first_body
                        confidence = min(1.0, strength_ratio)
                        
                        patterns.append({
                            'start_index': i-2,
                            'end_index': i,
                            'confidence': confidence,
                            'volume_confirmation': self._check_volume_confirmation(data, i, has_volume)
                        })
                        
        except Exception as e:
            logger.error(f"Error detecting evening star: {e}")
        
        return patterns
    
    def _detect_three_white_soldiers(self, data: pd.DataFrame, has_volume: bool) -> List[Dict[str, Any]]:
        """Detect Three White Soldiers patterns"""
        patterns = []
        
        try:
            for i in range(2, len(data)):
                # Three consecutive candles
                candles = []
                for j in range(3):
                    idx = i - 2 + j
                    candles.append({
                        'open': data['Open'].iloc[idx],
                        'close': data['Close'].iloc[idx],
                        'high': data['High'].iloc[idx],
                        'low': data['Low'].iloc[idx]
                    })
                
                # Three white soldiers criteria
                all_bullish = all(candle['close'] > candle['open'] for candle in candles)
                ascending_closes = (candles[1]['close'] > candles[0]['close'] and 
                                  candles[2]['close'] > candles[1]['close'])
                ascending_opens = (candles[1]['open'] > candles[0]['open'] and 
                                 candles[2]['open'] > candles[1]['open'])
                
                if all_bullish and ascending_closes and ascending_opens:
                    # Check for reasonable body sizes
                    bodies = [abs(candle['close'] - candle['open']) for candle in candles]
                    avg_body = np.mean(bodies)
                    
                    if all(body > avg_body * 0.5 for body in bodies):  # No unusually small bodies
                        confidence = min(1.0, avg_body / (candles[0]['high'] - candles[0]['low']))
                        
                        patterns.append({
                            'start_index': i-2,
                            'end_index': i,
                            'confidence': confidence,
                            'volume_confirmation': self._check_volume_confirmation(data, i, has_volume)
                        })
                        
        except Exception as e:
            logger.error(f"Error detecting three white soldiers: {e}")
        
        return patterns
    
    def _detect_three_black_crows(self, data: pd.DataFrame, has_volume: bool) -> List[Dict[str, Any]]:
        """Detect Three Black Crows patterns"""
        patterns = []
        
        try:
            for i in range(2, len(data)):
                # Three consecutive candles
                candles = []
                for j in range(3):
                    idx = i - 2 + j
                    candles.append({
                        'open': data['Open'].iloc[idx],
                        'close': data['Close'].iloc[idx],
                        'high': data['High'].iloc[idx],
                        'low': data['Low'].iloc[idx]
                    })
                
                # Three black crows criteria
                all_bearish = all(candle['close'] < candle['open'] for candle in candles)
                descending_closes = (candles[1]['close'] < candles[0]['close'] and 
                                   candles[2]['close'] < candles[1]['close'])
                descending_opens = (candles[1]['open'] < candles[0]['open'] and 
                                  candles[2]['open'] > candles[1]['open'])
                
                if all_bearish and descending_closes and descending_opens:
                    # Check for reasonable body sizes
                    bodies = [abs(candle['close'] - candle['open']) for candle in candles]
                    avg_body = np.mean(bodies)
                    
                    if all(body > avg_body * 0.5 for body in bodies):  # No unusually small bodies
                        confidence = min(1.0, avg_body / (candles[0]['high'] - candles[0]['low']))
                        
                        patterns.append({
                            'start_index': i-2,
                            'end_index': i,
                            'confidence': confidence,
                            'volume_confirmation': self._check_volume_confirmation(data, i, has_volume)
                        })
                        
        except Exception as e:
            logger.error(f"Error detecting three black crows: {e}")
        
        return patterns
    
    def _detect_bullish_harami(self, data: pd.DataFrame, has_volume: bool) -> List[Dict[str, Any]]:
        """Detect Bullish Harami patterns"""
        patterns = []
        
        try:
            for i in range(1, len(data)):
                # First candle (bearish)
                open1 = data['Open'].iloc[i-1]
                close1 = data['Close'].iloc[i-1]
                high1 = data['High'].iloc[i-1]
                low1 = data['Low'].iloc[i-1]
                
                # Second candle (bullish)
                open2 = data['Open'].iloc[i]
                close2 = data['Close'].iloc[i]
                high2 = data['High'].iloc[i]
                low2 = data['Low'].iloc[i]
                
                # Bullish harami criteria
                if (close1 < open1 and  # First candle is bearish
                    close2 > open2 and  # Second candle is bullish
                    open2 > close1 and  # Second opens above first close
                    close2 < open1 and  # Second closes below first open
                    high2 < high1 and   # Second high is lower
                    low2 > low1):       # Second low is higher
                    
                    # Calculate containment ratio
                    first_body = abs(close1 - open1)
                    second_body = abs(close2 - open2)
                    
                    if first_body > 0:
                        containment_ratio = second_body / first_body
                        confidence = min(1.0, 1.0 - containment_ratio)  # Smaller second candle = higher confidence
                        
                        patterns.append({
                            'start_index': i-1,
                            'end_index': i,
                            'confidence': confidence,
                            'volume_confirmation': self._check_volume_confirmation(data, i, has_volume)
                        })
                        
        except Exception as e:
            logger.error(f"Error detecting bullish harami: {e}")
        
        return patterns
    
    def _detect_bearish_harami(self, data: pd.DataFrame, has_volume: bool) -> List[Dict[str, Any]]:
        """Detect Bearish Harami patterns"""
        patterns = []
        
        try:
            for i in range(1, len(data)):
                # First candle (bullish)
                open1 = data['Open'].iloc[i-1]
                close1 = data['Close'].iloc[i-1]
                high1 = data['High'].iloc[i-1]
                low1 = data['Low'].iloc[i-1]
                
                # Second candle (bearish)
                open2 = data['Open'].iloc[i]
                close2 = data['Close'].iloc[i]
                high2 = data['High'].iloc[i]
                low2 = data['Low'].iloc[i]
                
                # Bearish harami criteria
                if (close1 > open1 and  # First candle is bullish
                    close2 < open2 and  # Second candle is bearish
                    open2 < close1 and  # Second opens below first close
                    close2 > open1 and  # Second closes above first open
                    high2 < high1 and   # Second high is lower
                    low2 > low1):       # Second low is higher
                    
                    # Calculate containment ratio
                    first_body = abs(close1 - open1)
                    second_body = abs(close2 - open2)
                    
                    if first_body > 0:
                        containment_ratio = second_body / first_body
                        confidence = min(1.0, 1.0 - containment_ratio)
                        
                        patterns.append({
                            'start_index': i-1,
                            'end_index': i,
                            'confidence': confidence,
                            'volume_confirmation': self._check_volume_confirmation(data, i, has_volume)
                        })
                        
        except Exception as e:
            logger.error(f"Error detecting bearish harami: {e}")
        
        return patterns
    
    def _detect_dark_cloud_cover(self, data: pd.DataFrame, has_volume: bool) -> List[Dict[str, Any]]:
        """Detect Dark Cloud Cover patterns"""
        patterns = []
        
        try:
            for i in range(1, len(data)):
                # First candle (bullish)
                open1 = data['Open'].iloc[i-1]
                close1 = data['Close'].iloc[i-1]
                
                # Second candle (bearish)
                open2 = data['Open'].iloc[i]
                close2 = data['Close'].iloc[i]
                
                # Dark cloud cover criteria
                if (close1 > open1 and  # First candle is bullish
                    close2 < open2 and  # Second candle is bearish
                    open2 > close1 and  # Second opens above first close (gap up)
                    close2 < (open1 + close1) / 2):  # Second closes below midpoint of first
                    
                    # Calculate penetration depth
                    first_body = close1 - open1
                    penetration = close1 - close2
                    
                    if first_body > 0:
                        penetration_ratio = penetration / first_body
                        confidence = min(1.0, penetration_ratio)
                        
                        patterns.append({
                            'start_index': i-1,
                            'end_index': i,
                            'confidence': confidence,
                            'volume_confirmation': self._check_volume_confirmation(data, i, has_volume)
                        })
                        
        except Exception as e:
            logger.error(f"Error detecting dark cloud cover: {e}")
        
        return patterns
    
    def _detect_piercing_pattern(self, data: pd.DataFrame, has_volume: bool) -> List[Dict[str, Any]]:
        """Detect Piercing Pattern"""
        patterns = []
        
        try:
            for i in range(1, len(data)):
                # First candle (bearish)
                open1 = data['Open'].iloc[i-1]
                close1 = data['Close'].iloc[i-1]
                
                # Second candle (bullish)
                open2 = data['Open'].iloc[i]
                close2 = data['Close'].iloc[i]
                
                # Piercing pattern criteria
                if (close1 < open1 and  # First candle is bearish
                    close2 > open2 and  # Second candle is bullish
                    open2 < close1 and  # Second opens below first close (gap down)
                    close2 > (open1 + close1) / 2):  # Second closes above midpoint of first
                    
                    # Calculate penetration depth
                    first_body = open1 - close1
                    penetration = close2 - close1
                    
                    if first_body > 0:
                        penetration_ratio = penetration / first_body
                        confidence = min(1.0, penetration_ratio)
                        
                        patterns.append({
                            'start_index': i-1,
                            'end_index': i,
                            'confidence': confidence,
                            'volume_confirmation': self._check_volume_confirmation(data, i, has_volume)
                        })
                        
        except Exception as e:
            logger.error(f"Error detecting piercing pattern: {e}")
        
        return patterns
    
    def _check_volume_confirmation(self, data: pd.DataFrame, index: int, has_volume: bool) -> bool:
        """Check if volume confirms the pattern"""
        try:
            if not has_volume or index < 10:
                return False
            
            current_volume = data['Volume'].iloc[index]
            avg_volume = data['Volume'].iloc[max(0, index-10):index].mean()
            
            return current_volume > avg_volume * self.volume_threshold
            
        except Exception as e:
            logger.error(f"Error checking volume confirmation: {e}")
            return False
    
    def _check_uptrend_context(self, data: pd.DataFrame, index: int) -> bool:
        """Check if we're in an uptrend context"""
        try:
            if index < self.trend_lookback:
                return False
            
            start_price = data['Close'].iloc[index - self.trend_lookback]
            current_price = data['Close'].iloc[index]
            
            return current_price > start_price * 1.05  # 5% increase over lookback period
            
        except Exception as e:
            logger.error(f"Error checking uptrend context: {e}")
            return False
    
    def _check_downtrend_context(self, data: pd.DataFrame, index: int) -> bool:
        """Check if we're in a downtrend context"""
        try:
            if index < self.trend_lookback:
                return False
            
            start_price = data['Close'].iloc[index - self.trend_lookback]
            current_price = data['Close'].iloc[index]
            
            return current_price < start_price * 0.95  # 5% decrease over lookback period
            
        except Exception as e:
            logger.error(f"Error checking downtrend context: {e}")
            return False
    
    def _analyze_trend_context(self, data: pd.DataFrame) -> str:
        """Analyze overall trend context"""
        try:
            if len(data) < self.trend_lookback:
                return "INSUFFICIENT_DATA"
            
            # Calculate trend using linear regression
            prices = data['Close'].tail(self.trend_lookback).values
            x = np.arange(len(prices))
            
            # Simple linear regression
            slope = np.polyfit(x, prices, 1)[0]
            
            # Determine trend based on slope
            price_change_pct = slope / prices[0] * len(prices)
            
            if price_change_pct > 0.05:
                return "UPTREND"
            elif price_change_pct < -0.05:
                return "DOWNTREND"
            else:
                return "SIDEWAYS"
                
        except Exception as e:
            logger.error(f"Error analyzing trend context: {e}")
            return "UNKNOWN"
    
    def _enhance_pattern(self, pattern_result: Dict[str, Any], pattern_name: str,
                        pattern_info: Dict[str, Any], data: pd.DataFrame,
                        trend_context: str) -> Optional[CandlePattern]:
        """Enhance pattern with additional analysis"""
        try:
            start_idx = pattern_result['start_index']
            end_idx = pattern_result['end_index']
            
            # Calculate pattern strength
            strength = self._calculate_pattern_strength(
                pattern_result['confidence'], pattern_info['reliability']
            )
            
            # Generate trading signal
            trading_signal = self._generate_trading_signal(pattern_info['type'], trend_context)
            
            # Calculate target price and stop loss
            target_price, stop_loss = self._calculate_price_targets(
                data, start_idx, end_idx, pattern_info['type']
            )
            
            # Calculate success probability
            probability = self._calculate_success_probability(
                pattern_name, pattern_result['confidence'], trend_context
            )
            
            return CandlePattern(
                name=pattern_name,
                pattern_type=pattern_info['type'],
                strength=strength,
                confidence=pattern_result['confidence'],
                start_index=start_idx,
                end_index=end_idx,
                description=pattern_info['description'],
                trading_signal=trading_signal,
                target_price=target_price,
                stop_loss=stop_loss,
                probability=probability,
                volume_confirmation=pattern_result['volume_confirmation'],
                trend_context=trend_context
            )
            
        except Exception as e:
            logger.error(f"Error enhancing pattern: {e}")
            return None
    
    def _calculate_pattern_strength(self, confidence: float, reliability: float) -> PatternStrength:
        """Calculate pattern strength based on confidence and reliability"""
        try:
            combined_score = (confidence + reliability) / 2
            
            if combined_score >= 0.85:
                return PatternStrength.VERY_STRONG
            elif combined_score >= 0.7:
                return PatternStrength.STRONG
            elif combined_score >= 0.55:
                return PatternStrength.MODERATE
            else:
                return PatternStrength.WEAK
                
        except Exception as e:
            logger.error(f"Error calculating pattern strength: {e}")
            return PatternStrength.MODERATE
    
    def _generate_trading_signal(self, pattern_type: PatternType, trend_context: str) -> str:
        """Generate trading signal based on pattern type and trend context"""
        try:
            if pattern_type == PatternType.BULLISH:
                if trend_context in ["UPTREND", "SIDEWAYS"]:
                    return "STRONG_BUY"
                else:
                    return "BUY"
            elif pattern_type == PatternType.BEARISH:
                if trend_context in ["DOWNTREND", "SIDEWAYS"]:
                    return "STRONG_SELL"
                else:
                    return "SELL"
            elif pattern_type == PatternType.REVERSAL:
                if trend_context == "UPTREND":
                    return "SELL"
                elif trend_context == "DOWNTREND":
                    return "BUY"
                else:
                    return "HOLD"
            else:
                return "HOLD"
                
        except Exception as e:
            logger.error(f"Error generating trading signal: {e}")
            return "HOLD"
    
    def _calculate_price_targets(self, data: pd.DataFrame, start_idx: int, end_idx: int,
                               pattern_type: PatternType) -> Tuple[Optional[float], Optional[float]]:
        """Calculate target price and stop loss"""
        try:
            current_price = data['Close'].iloc[end_idx]
            
            # Calculate average true range for target calculation
            if end_idx >= 14:
                atr_data = data.iloc[max(0, end_idx-14):end_idx+1]
                atr = talib.ATR(atr_data['High'].values, atr_data['Low'].values, 
                               atr_data['Close'].values, timeperiod=min(14, len(atr_data)))
                avg_atr = atr[-1] if len(atr) > 0 and not np.isnan(atr[-1]) else current_price * 0.02
            else:
                avg_atr = current_price * 0.02
            
            # Calculate targets based on pattern type
            if pattern_type == PatternType.BULLISH:
                target_price = current_price + (avg_atr * 2)
                stop_loss = current_price - (avg_atr * 1.5)
            elif pattern_type == PatternType.BEARISH:
                target_price = current_price - (avg_atr * 2)
                stop_loss = current_price + (avg_atr * 1.5)
            else:
                target_price = None
                stop_loss = None
            
            return target_price, stop_loss
            
        except Exception as e:
            logger.error(f"Error calculating price targets: {e}")
            return None, None
    
    def _calculate_success_probability(self, pattern_name: str, confidence: float,
                                     trend_context: str) -> float:
        """Calculate success probability based on historical performance"""
        try:
            # Base probability from pattern definition
            base_prob = self.pattern_definitions[pattern_name]['reliability']
            
            # Adjust for confidence
            confidence_adjustment = (confidence - 0.5) * 0.2
            
            # Adjust for trend context
            trend_adjustment = 0.0
            pattern_type = self.pattern_definitions[pattern_name]['type']
            
            if pattern_type == PatternType.BULLISH and trend_context == "UPTREND":
                trend_adjustment = 0.1
            elif pattern_type == PatternType.BEARISH and trend_context == "DOWNTREND":
                trend_adjustment = 0.1
            elif pattern_type == PatternType.BULLISH and trend_context == "DOWNTREND":
                trend_adjustment = -0.1
            elif pattern_type == PatternType.BEARISH and trend_context == "UPTREND":
                trend_adjustment = -0.1
            
            # Historical performance adjustment
            historical_adjustment = 0.0
            if pattern_name in self.pattern_performance:
                historical_success_rate = self.pattern_performance[pattern_name].get('success_rate', 0.5)
                historical_adjustment = (historical_success_rate - 0.5) * 0.1
            
            final_probability = base_prob + confidence_adjustment + trend_adjustment + historical_adjustment
            
            return max(0.1, min(0.9, final_probability))
            
        except Exception as e:
            logger.error(f"Error calculating success probability: {e}")
            return 0.5
    
    def _store_detection_history(self, symbol: str, patterns: List[CandlePattern]):
        """Store pattern detection history"""
        try:
            detection_record = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'patterns_detected': len(patterns),
                'patterns': [
                    {
                        'name': p.name,
                        'type': p.pattern_type.value,
                        'confidence': p.confidence,
                        'strength': p.strength.value
                    }
                    for p in patterns
                ]
            }
            
            self.detection_history.append(detection_record)
            
            # Keep only recent history
            if len(self.detection_history) > 1000:
                self.detection_history = self.detection_history[-500:]
                
        except Exception as e:
            logger.error(f"Error storing detection history: {e}")
    
    def get_pattern_signals(self, symbol: str, min_confirmation: float = 0.6) -> List[Dict[str, Any]]:
        """Get trading signals from detected patterns"""
        try:
            # This would typically fetch recent market data and detect patterns
            # For now, return mock signals based on stored patterns
            signals = []
            
            for record in self.detection_history[-10:]:  # Last 10 detections
                if record.get('symbol') == symbol:
                    for pattern in record['patterns']:
                        if pattern['confidence'] >= min_confirmation:
                            signals.append({
                                'symbol': symbol,
                                'pattern_name': pattern['name'],
                                'signal_type': pattern['type'],
                                'confidence': pattern['confidence'],
                                'timestamp': record['timestamp'],
                                'source': 'pattern_detection'
                            })
            
            return signals
            
        except Exception as e:
            logger.error(f"Error getting pattern signals: {e}")
            return []
    
    def update_pattern_performance(self, pattern_name: str, success: bool, return_pct: float):
        """Update pattern performance tracking"""
        try:
            if pattern_name not in self.pattern_performance:
                self.pattern_performance[pattern_name] = {
                    'total_occurrences': 0,
                    'successful_trades': 0,
                    'total_return': 0.0,
                    'success_rate': 0.0,
                    'avg_return': 0.0
                }
            
            perf = self.pattern_performance[pattern_name]
            perf['total_occurrences'] += 1
            
            if success:
                perf['successful_trades'] += 1
            
            perf['total_return'] += return_pct
            perf['success_rate'] = perf['successful_trades'] / perf['total_occurrences']
            perf['avg_return'] = perf['total_return'] / perf['total_occurrences']
            
        except Exception as e:
            logger.error(f"Error updating pattern performance: {e}")
    
    def get_pattern_statistics(self) -> Dict[str, Any]:
        """Get pattern detection and performance statistics"""
        try:
            stats = {
                'total_detections': len(self.detection_history),
                'patterns_tracked': len(self.pattern_performance),
                'pattern_performance': self.pattern_performance.copy()
            }
            
            # Calculate overall statistics
            if self.detection_history:
                recent_detections = self.detection_history[-100:]
                pattern_counts = {}
                
                for record in recent_detections:
                    for pattern in record['patterns']:
                        pattern_name = pattern['name']
                        pattern_counts[pattern_name] = pattern_counts.get(pattern_name, 0) + 1
                
                stats['recent_pattern_frequency'] = pattern_counts
                stats['most_common_pattern'] = max(pattern_counts, key=pattern_counts.get) if pattern_counts else None
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting pattern statistics: {e}")
            return {}

def main():
    """Test the Candle Pattern Detector"""
    # Create sample OHLCV data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    
    # Generate realistic OHLCV data
    base_price = 100
    prices = []
    volumes = []
    
    for i in range(len(dates)):
        # Random walk with some volatility
        change = np.random.normal(0.001, 0.02)
        base_price *= (1 + change)
        
        # Generate OHLC
        open_price = base_price
        close_price = base_price * (1 + np.random.normal(0, 0.015))
        high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.01)))
        low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.01)))
        volume = np.random.randint(100000, 1000000)
        
        prices.append([open_price, high_price, low_price, close_price])
        volumes.append(volume)
        base_price = close_price
    
    # Create DataFrame
    price_data = np.array(prices)
    test_data = pd.DataFrame({
        'Open': price_data[:, 0],
        'High': price_data[:, 1],
        'Low': price_data[:, 2],
        'Close': price_data[:, 3],
        'Volume': volumes
    }, index=dates)
    
    # Initialize detector
    detector = CandlePatternDetector()
    
    # Detect patterns
    patterns = detector.detect_patterns(test_data, symbol='TEST')
    
    print(f"Detected {len(patterns)} candlestick patterns:")
    print("=" * 80)
    
    for pattern in patterns[:10]:  # Show top 10 patterns
        print(f"Pattern: {pattern.name}")
        print(f"Type: {pattern.pattern_type.value}")
        print(f"Strength: {pattern.strength.name}")
        print(f"Confidence: {pattern.confidence:.2f}")
        print(f"Trading Signal: {pattern.trading_signal}")
        print(f"Success Probability: {pattern.probability:.2%}")
        print(f"Volume Confirmation: {pattern.volume_confirmation}")
        print(f"Trend Context: {pattern.trend_context}")
        print(f"Description: {pattern.description}")
        if pattern.target_price:
            print(f"Target Price: ${pattern.target_price:.2f}")
        if pattern.stop_loss:
            print(f"Stop Loss: ${pattern.stop_loss:.2f}")
        print("-" * 80)
    
    # Get statistics
    stats = detector.get_pattern_statistics()
    print(f"\nPattern Detection Statistics:")
    print(f"Total Detections: {stats.get('total_detections', 0)}")
    print(f"Patterns Tracked: {stats.get('patterns_tracked', 0)}")
    
    if 'recent_pattern_frequency' in stats:
        print(f"Most Common Recent Patterns:")
        for pattern, count in sorted(stats['recent_pattern_frequency'].items(), 
                                   key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {pattern}: {count} occurrences")

if __name__ == "__main__":
    main()
