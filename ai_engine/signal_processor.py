import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import statistics

logger = logging.getLogger(__name__)

@dataclass
class ProcessedSignal:
    original_signal: Any
    confidence: float
    priority: int
    risk_score: float
    correlation_group: Optional[str] = None
    timestamp: str = None

class SignalProcessor:
    """Advanced signal processing with filtering and correlation analysis"""
    
    def __init__(self, min_confidence: float = 0.8, max_signals_per_minute: int = 10, correlation_threshold: float = 0.7):
        self.min_confidence = min_confidence
        self.max_signals_per_minute = max_signals_per_minute
        self.correlation_threshold = correlation_threshold
        
        # Signal tracking
        self.recent_signals = []
        self.signal_history = []
        self.correlation_matrix = {}
        
        logger.info("Signal Processor initialized")
    
    async def process_signals(self, raw_signals: List[Any]) -> List[ProcessedSignal]:
        """Process and filter raw trading signals"""
        try:
            # Filter by confidence
            high_confidence_signals = [s for s in raw_signals if s.confidence >= self.min_confidence]
            
            # Remove duplicates
            unique_signals = await self._remove_duplicates(high_confidence_signals)
            
            # Apply correlation filtering
            correlation_filtered = await self._filter_by_correlation(unique_signals)
            
            # Apply rate limiting
            rate_limited = await self._apply_rate_limiting(correlation_filtered)
            
            # Calculate risk scores
            processed_signals = []
            for signal in rate_limited:
                risk_score = await self._calculate_risk_score(signal)
                priority = await self._calculate_priority(signal)
                
                processed_signal = ProcessedSignal(
                    original_signal=signal,
                    confidence=signal.confidence,
                    priority=priority,
                    risk_score=risk_score,
                    timestamp=datetime.now().isoformat()
                )
                processed_signals.append(processed_signal)
            
            # Sort by priority
            processed_signals.sort(key=lambda x: x.priority, reverse=True)
            
            # Update tracking
            self.recent_signals.extend(processed_signals)
            self._cleanup_old_signals()
            
            logger.info(f"Processed {len(raw_signals)} signals -> {len(processed_signals)} final signals")
            return processed_signals
            
        except Exception as e:
            logger.error(f"Error processing signals: {e}")
            return []
    
    async def _remove_duplicates(self, signals: List[Any]) -> List[Any]:
        """Remove duplicate signals for the same symbol"""
        seen_symbols = set()
        unique_signals = []
        
        # Sort by confidence first
        sorted_signals = sorted(signals, key=lambda x: x.confidence, reverse=True)
        
        for signal in sorted_signals:
            if signal.symbol not in seen_symbols:
                unique_signals.append(signal)
                seen_symbols.add(signal.symbol)
        
        return unique_signals
    
    async def _filter_by_correlation(self, signals: List[Any]) -> List[Any]:
        """Filter signals based on symbol correlation"""
        if len(signals) <= 1:
            return signals
        
        # Group signals by correlation
        correlation_groups = {}
        processed_symbols = set()
        
        for signal in signals:
            if signal.symbol in processed_symbols:
                continue
            
            # Find correlated symbols
            correlated_symbols = [signal.symbol]
            for other_signal in signals:
                if (other_signal.symbol != signal.symbol and 
                    other_signal.symbol not in processed_symbols):
                    
                    correlation = await self._get_correlation(signal.symbol, other_signal.symbol)
                    if correlation > self.correlation_threshold:
                        correlated_symbols.append(other_signal.symbol)
            
            # Select best signal from correlated group
            group_signals = [s for s in signals if s.symbol in correlated_symbols]
            best_signal = max(group_signals, key=lambda x: x.confidence)
            
            correlation_groups[f"group_{len(correlation_groups)}"] = best_signal
            processed_symbols.update(correlated_symbols)
        
        return list(correlation_groups.values())
    
    async def _get_correlation(self, symbol1: str, symbol2: str) -> float:
        """Get correlation between two symbols"""
        # In a real implementation, this would calculate actual correlation
        # For now, return a mock correlation
        if symbol1 == symbol2:
            return 1.0
        
        # Mock correlation based on symbol similarity
        if symbol1[:2] == symbol2[:2]:  # Same sector/category
            return 0.8
        else:
            return 0.3
    
    async def _apply_rate_limiting(self, signals: List[Any]) -> List[Any]:
        """Apply rate limiting to prevent too many signals"""
        current_time = datetime.now()
        one_minute_ago = current_time - timedelta(minutes=1)
        
        # Count recent signals
        recent_count = len([s for s in self.recent_signals 
                           if datetime.fromisoformat(s.timestamp) > one_minute_ago])
        
        # Calculate how many more signals we can accept
        available_slots = max(0, self.max_signals_per_minute - recent_count)
        
        # Return top signals within limit
        return signals[:available_slots]
    
    async def _calculate_risk_score(self, signal: Any) -> float:
        """Calculate risk score for a signal"""
        try:
            risk_factors = []
            
            # Confidence-based risk (lower confidence = higher risk)
            confidence_risk = 1.0 - signal.confidence
            risk_factors.append(confidence_risk)
            
            # Volatility risk (mock calculation)
            volatility_risk = 0.3  # Mock value
            risk_factors.append(volatility_risk)
            
            # Market condition risk
            market_risk = 0.2  # Mock value
            risk_factors.append(market_risk)
            
            # Calculate weighted average
            weights = [0.5, 0.3, 0.2]
            risk_score = sum(r * w for r, w in zip(risk_factors, weights))
            
            return min(1.0, max(0.0, risk_score))
            
        except Exception as e:
            logger.error(f"Error calculating risk score: {e}")
            return 0.5  # Default medium risk
    
    async def _calculate_priority(self, signal: Any) -> int:
        """Calculate priority score for a signal"""
        try:
            priority_score = 0
            
            # Confidence contribution (0-40 points)
            priority_score += int(signal.confidence * 40)
            
            # Signal strength contribution (0-30 points)
            if signal.signal == 'buy' or signal.signal == 'sell':
                priority_score += 30
            elif signal.signal == 'hold':
                priority_score += 10
            
            # Time sensitivity (0-20 points)
            priority_score += 20  # Mock value
            
            # Market conditions (0-10 points)
            priority_score += 10  # Mock value
            
            return min(100, max(0, priority_score))
            
        except Exception as e:
            logger.error(f"Error calculating priority: {e}")
            return 50  # Default medium priority
    
    def _cleanup_old_signals(self):
        """Remove old signals from tracking"""
        cutoff_time = datetime.now() - timedelta(hours=1)
        self.recent_signals = [
            s for s in self.recent_signals 
            if datetime.fromisoformat(s.timestamp) > cutoff_time
        ]
    
    async def get_processing_stats(self) -> Dict[str, Any]:
        """Get signal processing statistics"""
        try:
            current_time = datetime.now()
            one_hour_ago = current_time - timedelta(hours=1)
            
            recent_signals = [
                s for s in self.recent_signals 
                if datetime.fromisoformat(s.timestamp) > one_hour_ago
            ]
            
            stats = {
                'total_processed': len(self.signal_history),
                'recent_signals': len(recent_signals),
                'average_confidence': statistics.mean([s.confidence for s in recent_signals]) if recent_signals else 0,
                'average_risk_score': statistics.mean([s.risk_score for s in recent_signals]) if recent_signals else 0,
                'min_confidence_threshold': self.min_confidence,
                'max_signals_per_minute': self.max_signals_per_minute,
                'correlation_threshold': self.correlation_threshold
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting processing stats: {e}")
            return {}
