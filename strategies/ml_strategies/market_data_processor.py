"""
Market Data Processor
===================

Handles real-time market data processing and cleaning for ML strategies.
This is the first step in the ML modeling workflow from Modeling.md:

Real-Time Market Data → Data Cleaning & Feature Engineering

Processes:
- Raw data streams capturing trades, order books, cancellations, quotes
- Data validation and cleaning
- Normalization and preprocessing
- Real-time data quality monitoring
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import threading
from enum import Enum

from .base_ml_strategy import MarketMicrostructureData

logger = logging.getLogger(__name__)

class DataQualityIssue(Enum):
    """Types of data quality issues"""
    MISSING_DATA = "missing_data"
    STALE_DATA = "stale_data"
    INVALID_PRICE = "invalid_price"
    INVALID_VOLUME = "invalid_volume"
    CROSSED_SPREAD = "crossed_spread"
    EXTREME_MOVEMENT = "extreme_movement"
    TIMESTAMP_OUT_OF_ORDER = "timestamp_out_of_order"

@dataclass
class DataQualityMetrics:
    """Metrics for data quality monitoring"""
    total_messages_processed: int = 0
    total_messages_rejected: int = 0
    issue_counts: Dict[DataQualityIssue, int] = field(default_factory=dict)
    last_message_timestamp: Optional[int] = None
    avg_message_rate_per_second: float = 0.0
    data_latency_ns: float = 0.0

@dataclass
class ProcessingConfig:
    """Configuration for market data processing"""
    # Data validation
    max_price_change_pct: float = 0.1  # 10% max price change
    max_spread_pct: float = 0.05  # 5% max spread
    min_volume: float = 0.0
    max_volume: float = 1e9
    stale_data_threshold_ms: int = 1000  # 1 second
    
    # Buffer management
    max_buffer_size: int = 10000
    cleanup_interval_seconds: int = 60
    
    # Quality thresholds
    min_data_quality_score: float = 0.8
    max_consecutive_rejections: int = 100

class MarketDataProcessor:
    """
    Processes and cleans real-time market data for ML strategies
    
    Responsibilities:
    1. Validate incoming market data for quality issues
    2. Clean and normalize data streams
    3. Handle missing or corrupted data
    4. Maintain data quality metrics
    5. Buffer clean data for feature engineering
    """
    
    def __init__(self, symbols: List[str], config: ProcessingConfig):
        self.symbols = set(symbols)
        self.config = config
        
        # Data buffers per symbol
        self.data_buffers: Dict[str, deque] = {
            symbol: deque(maxlen=config.max_buffer_size) 
            for symbol in symbols
        }
        
        # Quality tracking
        self.quality_metrics = DataQualityMetrics()
        self.last_valid_data: Dict[str, MarketMicrostructureData] = {}
        self.consecutive_rejections: Dict[str, int] = {symbol: 0 for symbol in symbols}
        
        # Threading
        self.lock = threading.RLock()
        
        # Reference data for validation
        self.reference_prices: Dict[str, float] = {}
        self.price_history: Dict[str, deque] = {
            symbol: deque(maxlen=100) for symbol in symbols
        }
        
        logger.info(f"Initialized market data processor for {len(symbols)} symbols")
    
    def process_market_data(self, raw_data: Dict[str, Any]) -> Optional[MarketMicrostructureData]:
        """
        Process and validate incoming market data
        
        Args:
            raw_data: Raw market data dictionary
            
        Returns:
            MarketMicrostructureData: Cleaned and validated market data, or None if rejected
        """
        with self.lock:
            try:
                # Parse raw data into structured format
                market_data = self._parse_raw_data(raw_data)
                if market_data is None:
                    return None
                
                # Validate data quality
                quality_issues = self._validate_data_quality(market_data)
                
                if quality_issues:
                    self._handle_quality_issues(market_data.symbol, quality_issues)
                    return None
                
                # Clean and normalize data
                cleaned_data = self._clean_and_normalize(market_data)
                
                # Update buffers and metrics
                self._update_buffers(cleaned_data)
                self._update_quality_metrics(cleaned_data)
                
                # Reset consecutive rejections counter
                self.consecutive_rejections[cleaned_data.symbol] = 0
                
                return cleaned_data
                
            except Exception as e:
                logger.error(f"Error processing market data: {e}")
                return None
    
    def _parse_raw_data(self, raw_data: Dict[str, Any]) -> Optional[MarketMicrostructureData]:
        """Parse raw data dictionary into MarketMicrostructureData"""
        try:
            symbol = raw_data.get('symbol')
            if symbol not in self.symbols:
                return None
            
            # Extract basic fields
            timestamp = raw_data.get('timestamp', int(datetime.now().timestamp() * 1e9))
            best_bid = float(raw_data.get('best_bid', 0))
            best_ask = float(raw_data.get('best_ask', 0))
            bid_volume = float(raw_data.get('bid_volume', 0))
            ask_volume = float(raw_data.get('ask_volume', 0))
            
            # Extract order book depth
            bid_depth = []
            ask_depth = []
            
            if 'bid_depth' in raw_data:
                for level in raw_data['bid_depth']:
                    bid_depth.append((float(level['price']), float(level['volume'])))
            
            if 'ask_depth' in raw_data:
                for level in raw_data['ask_depth']:
                    ask_depth.append((float(level['price']), float(level['volume'])))
            
            # Extract trade data
            last_price = float(raw_data.get('last_price', (best_bid + best_ask) / 2))
            last_volume = float(raw_data.get('last_volume', 0))
            trade_direction = int(raw_data.get('trade_direction', 0))
            
            # Calculate derived fields
            mid_price = (best_bid + best_ask) / 2 if best_bid > 0 and best_ask > 0 else last_price
            spread = best_ask - best_bid if best_bid > 0 and best_ask > 0 else 0
            
            # Calculate order book imbalance
            total_bid_vol = bid_volume + sum(vol for _, vol in bid_depth)
            total_ask_vol = ask_volume + sum(vol for _, vol in ask_depth)
            order_book_imbalance = (total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol) if (total_bid_vol + total_ask_vol) > 0 else 0
            
            # Calculate VWAP (simplified)
            vwap = self._calculate_vwap(symbol, last_price, last_volume)
            
            return MarketMicrostructureData(
                timestamp=timestamp,
                symbol=symbol,
                best_bid=best_bid,
                best_ask=best_ask,
                bid_volume=bid_volume,
                ask_volume=ask_volume,
                bid_depth=bid_depth,
                ask_depth=ask_depth,
                last_price=last_price,
                last_volume=last_volume,
                trade_direction=trade_direction,
                mid_price=mid_price,
                spread=spread,
                order_book_imbalance=order_book_imbalance,
                volume_weighted_average_price=vwap
            )
            
        except Exception as e:
            logger.error(f"Error parsing raw data: {e}")
            return None
    
    def _validate_data_quality(self, market_data: MarketMicrostructureData) -> List[DataQualityIssue]:
        """Validate market data quality and return list of issues"""
        issues = []
        
        # Check for missing or invalid prices
        if market_data.best_bid <= 0 or market_data.best_ask <= 0:
            issues.append(DataQualityIssue.INVALID_PRICE)
        
        # Check for crossed spread
        if market_data.best_bid >= market_data.best_ask:
            issues.append(DataQualityIssue.CROSSED_SPREAD)
        
        # Check spread percentage
        if market_data.spread > 0:
            spread_pct = market_data.spread / market_data.mid_price
            if spread_pct > self.config.max_spread_pct:
                issues.append(DataQualityIssue.EXTREME_MOVEMENT)
        
        # Check volume ranges
        if (market_data.bid_volume < self.config.min_volume or 
            market_data.ask_volume < self.config.min_volume or
            market_data.bid_volume > self.config.max_volume or
            market_data.ask_volume > self.config.max_volume):
            issues.append(DataQualityIssue.INVALID_VOLUME)
        
        # Check for extreme price movements
        if market_data.symbol in self.reference_prices:
            ref_price = self.reference_prices[market_data.symbol]
            price_change_pct = abs(market_data.mid_price - ref_price) / ref_price
            if price_change_pct > self.config.max_price_change_pct:
                issues.append(DataQualityIssue.EXTREME_MOVEMENT)
        
        # Check for stale data
        if self.quality_metrics.last_message_timestamp:
            time_diff_ms = (market_data.timestamp - self.quality_metrics.last_message_timestamp) / 1e6
            if time_diff_ms > self.config.stale_data_threshold_ms:
                issues.append(DataQualityIssue.STALE_DATA)
        
        # Check timestamp ordering
        if (market_data.symbol in self.last_valid_data and 
            market_data.timestamp < self.last_valid_data[market_data.symbol].timestamp):
            issues.append(DataQualityIssue.TIMESTAMP_OUT_OF_ORDER)
        
        return issues
    
    def _clean_and_normalize(self, market_data: MarketMicrostructureData) -> MarketMicrostructureData:
        """Clean and normalize market data"""
        # Apply data cleaning transformations
        
        # Normalize volumes (remove outliers)
        market_data.bid_volume = self._normalize_volume(market_data.bid_volume)
        market_data.ask_volume = self._normalize_volume(market_data.ask_volume)
        
        # Smooth extreme price movements
        if market_data.symbol in self.price_history:
            market_data.mid_price = self._smooth_price(market_data.symbol, market_data.mid_price)
        
        # Recalculate derived fields after cleaning
        market_data.spread = market_data.best_ask - market_data.best_bid
        
        return market_data
    
    def _normalize_volume(self, volume: float) -> float:
        """Normalize volume to remove extreme outliers"""
        # Simple outlier detection and capping
        if volume > self.config.max_volume:
            return self.config.max_volume
        elif volume < self.config.min_volume:
            return self.config.min_volume
        return volume
    
    def _smooth_price(self, symbol: str, price: float) -> float:
        """Apply price smoothing to reduce noise"""
        if symbol not in self.price_history or len(self.price_history[symbol]) < 3:
            return price
        
        recent_prices = list(self.price_history[symbol])[-3:]
        median_price = np.median(recent_prices)
        
        # If price is too far from recent median, use weighted average
        price_diff_pct = abs(price - median_price) / median_price
        if price_diff_pct > self.config.max_price_change_pct / 2:
            # Weight recent price less if it's an outlier
            smoothed_price = 0.7 * median_price + 0.3 * price
            return smoothed_price
        
        return price
    
    def _calculate_vwap(self, symbol: str, price: float, volume: float) -> float:
        """Calculate volume-weighted average price"""
        if symbol not in self.data_buffers or len(self.data_buffers[symbol]) == 0:
            return price
        
        # Use recent data to calculate VWAP
        recent_data = list(self.data_buffers[symbol])[-20:]  # Last 20 data points
        
        total_volume = sum(data.last_volume for data in recent_data if data.last_volume > 0)
        total_value = sum(data.last_price * data.last_volume for data in recent_data if data.last_volume > 0)
        
        if total_volume > 0:
            return total_value / total_volume
        else:
            return price
    
    def _handle_quality_issues(self, symbol: str, issues: List[DataQualityIssue]) -> None:
        """Handle data quality issues"""
        self.consecutive_rejections[symbol] += 1
        self.quality_metrics.total_messages_rejected += 1
        
        # Update issue counts
        for issue in issues:
            if issue not in self.quality_metrics.issue_counts:
                self.quality_metrics.issue_counts[issue] = 0
            self.quality_metrics.issue_counts[issue] += 1
        
        # Log quality issues
        if self.consecutive_rejections[symbol] % 10 == 0:
            logger.warning(f"Data quality issues for {symbol}: {issues}. "
                         f"Consecutive rejections: {self.consecutive_rejections[symbol]}")
        
        # Check if we're rejecting too many messages
        if self.consecutive_rejections[symbol] > self.config.max_consecutive_rejections:
            logger.error(f"Too many consecutive rejections for {symbol}. "
                        f"Consider checking data source or adjusting quality thresholds.")
    
    def _update_buffers(self, market_data: MarketMicrostructureData) -> None:
        """Update data buffers with new market data"""
        symbol = market_data.symbol
        
        # Add to main data buffer
        self.data_buffers[symbol].append(market_data)
        
        # Update reference data
        self.reference_prices[symbol] = market_data.mid_price
        self.price_history[symbol].append(market_data.mid_price)
        self.last_valid_data[symbol] = market_data
    
    def _update_quality_metrics(self, market_data: MarketMicrostructureData) -> None:
        """Update data quality metrics"""
        self.quality_metrics.total_messages_processed += 1
        self.quality_metrics.last_message_timestamp = market_data.timestamp
        
        # Calculate message rate
        if hasattr(self, '_first_message_time'):
            elapsed_seconds = (market_data.timestamp - self._first_message_time) / 1e9
            if elapsed_seconds > 0:
                self.quality_metrics.avg_message_rate_per_second = (
                    self.quality_metrics.total_messages_processed / elapsed_seconds
                )
        else:
            self._first_message_time = market_data.timestamp
        
        # Calculate data latency (assuming current time as processing time)
        processing_time_ns = int(datetime.now().timestamp() * 1e9)
        self.quality_metrics.data_latency_ns = processing_time_ns - market_data.timestamp
    
    def get_data_quality_score(self, symbol: Optional[str] = None) -> float:
        """Calculate data quality score (0-1, where 1 is perfect quality)"""
        if symbol and symbol in self.consecutive_rejections:
            rejection_rate = self.consecutive_rejections[symbol] / max(1, len(self.data_buffers[symbol]))
        else:
            # Overall rejection rate
            rejection_rate = (self.quality_metrics.total_messages_rejected / 
                            max(1, self.quality_metrics.total_messages_processed))
        
        # Quality score is inverse of rejection rate
        return max(0.0, 1.0 - rejection_rate)
    
    def get_recent_data(self, symbol: str, count: int = 100) -> List[MarketMicrostructureData]:
        """Get recent market data for a symbol"""
        with self.lock:
            if symbol not in self.data_buffers:
                return []
            
            buffer = self.data_buffers[symbol]
            return list(buffer)[-count:] if len(buffer) >= count else list(buffer)
    
    def get_quality_metrics(self) -> Dict[str, Any]:
        """Get current data quality metrics"""
        with self.lock:
            return {
                'total_messages_processed': self.quality_metrics.total_messages_processed,
                'total_messages_rejected': self.quality_metrics.total_messages_rejected,
                'rejection_rate': (self.quality_metrics.total_messages_rejected / 
                                 max(1, self.quality_metrics.total_messages_processed)),
                'avg_message_rate_per_second': self.quality_metrics.avg_message_rate_per_second,
                'data_latency_ns': self.quality_metrics.data_latency_ns,
                'data_latency_ms': self.quality_metrics.data_latency_ns / 1e6,
                'issue_counts': dict(self.quality_metrics.issue_counts),
                'overall_quality_score': self.get_data_quality_score(),
                'symbol_quality_scores': {
                    symbol: self.get_data_quality_score(symbol) for symbol in self.symbols
                },
                'consecutive_rejections': dict(self.consecutive_rejections)
            }
    
    def reset_quality_metrics(self) -> None:
        """Reset all quality metrics"""
        with self.lock:
            self.quality_metrics = DataQualityMetrics()
            self.consecutive_rejections = {symbol: 0 for symbol in self.symbols}
            logger.info("Data quality metrics reset")
    
    def cleanup_old_data(self) -> None:
        """Clean up old data to free memory"""
        with self.lock:
            for symbol in self.symbols:
                # Keep only recent data in price history
                if len(self.price_history[symbol]) > 100:
                    # Keep last 50 entries
                    recent_prices = list(self.price_history[symbol])[-50:]
                    self.price_history[symbol].clear()
                    self.price_history[symbol].extend(recent_prices)
            
            logger.debug("Cleaned up old market data")
