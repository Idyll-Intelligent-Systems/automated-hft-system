"""
Feature Engineering Module
========================

Implements Step 2 of the ML modeling workflow from Modeling.md:
Data Cleaning & Feature Engineering

Extract meaningful signals including:
- Order book imbalance, trade momentum, volatility signals
- Market microstructure indicators (book imbalance, spread depth, etc.)
- Statistical indicators (realized volatility, VWAP deviation)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import deque
import logging
from datetime import datetime
import talib
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler

from .base_ml_strategy import MarketMicrostructureData, MLFeatures

logger = logging.getLogger(__name__)

@dataclass
class FeatureConfig:
    """Configuration for feature engineering"""
    # Window sizes for different calculations
    short_window: int = 20
    medium_window: int = 50
    long_window: int = 100
    
    # Volatility estimation
    volatility_window: int = 50
    volatility_scaling_factor: float = np.sqrt(252 * 24 * 60 * 60)  # Annualized
    
    # Momentum calculations
    momentum_windows: List[int] = None
    volume_momentum_window: int = 30
    
    # Order book analysis
    depth_levels: int = 10
    imbalance_window: int = 25
    
    # Statistical features
    autocorr_lags: List[int] = None
    rolling_correlation_window: int = 40
    
    # Normalization
    normalize_features: bool = True
    use_robust_scaling: bool = True
    
    def __post_init__(self):
        if self.momentum_windows is None:
            self.momentum_windows = [5, 10, 20, 50]
        if self.autocorr_lags is None:
            self.autocorr_lags = [1, 5, 10, 20]

class FeatureEngineer:
    """
    Feature engineering for HFT ML strategies
    
    Extracts meaningful market microstructure and statistical features
    from real-time market data following the ML modeling workflow.
    """
    
    def __init__(self, symbols: List[str], config: FeatureConfig):
        self.symbols = symbols
        self.config = config
        
        # Data buffers for feature calculation
        self.price_buffers: Dict[str, deque] = {
            symbol: deque(maxlen=config.long_window * 2) for symbol in symbols
        }
        self.volume_buffers: Dict[str, deque] = {
            symbol: deque(maxlen=config.long_window * 2) for symbol in symbols
        }
        self.spread_buffers: Dict[str, deque] = {
            symbol: deque(maxlen=config.long_window) for symbol in symbols
        }
        self.imbalance_buffers: Dict[str, deque] = {
            symbol: deque(maxlen=config.imbalance_window) for symbol in symbols
        }
        self.trade_direction_buffers: Dict[str, deque] = {
            symbol: deque(maxlen=config.long_window) for symbol in symbols
        }
        
        # Scalers for normalization
        if config.normalize_features:
            self.scalers: Dict[str, Any] = {}
            for symbol in symbols:
                if config.use_robust_scaling:
                    self.scalers[symbol] = RobustScaler()
                else:
                    self.scalers[symbol] = StandardScaler()
        
        # Cache for expensive calculations
        self.feature_cache: Dict[str, Dict] = {symbol: {} for symbol in symbols}
        self.last_calculation_timestamp: Dict[str, int] = {symbol: 0 for symbol in symbols}
        
        logger.info(f"Initialized feature engineer for {len(symbols)} symbols")
    
    def extract_features(self, market_data: MarketMicrostructureData) -> MLFeatures:
        """
        Main feature extraction method implementing the feature engineering workflow
        
        Args:
            market_data: Current market microstructure data
            
        Returns:
            MLFeatures: Comprehensive set of engineered features
        """
        symbol = market_data.symbol
        
        # Update data buffers
        self._update_buffers(market_data)
        
        # Extract different categories of features
        microstructure_features = self._extract_microstructure_features(market_data)
        statistical_features = self._extract_statistical_features(symbol)
        technical_features = self._extract_technical_features(symbol)
        time_based_features = self._extract_time_based_features(market_data)
        order_flow_features = self._extract_order_flow_features(market_data)
        
        # Combine all features
        all_features = {
            **microstructure_features,
            **statistical_features,
            **technical_features,
            **time_based_features,
            **order_flow_features
        }
        
        # Create MLFeatures object
        features = MLFeatures(
            # Core microstructure features
            order_book_imbalance=microstructure_features.get('order_book_imbalance', 0.0),
            spread_depth_ratio=microstructure_features.get('spread_depth_ratio', 0.0),
            price_momentum=technical_features.get('price_momentum_short', 0.0),
            volume_momentum=technical_features.get('volume_momentum', 0.0),
            volatility_signal=statistical_features.get('volatility_signal', 0.0),
            
            # Statistical features
            realized_volatility=statistical_features.get('realized_volatility', 0.0),
            vwap_deviation=statistical_features.get('vwap_deviation', 0.0),
            autocorrelation=statistical_features.get('autocorr_1', 0.0),
            
            # Time-based features
            time_of_day=time_based_features.get('time_of_day', 0.0),
            day_of_week=time_based_features.get('day_of_week', 0),
            market_session=time_based_features.get('market_session', 'unknown'),
            
            # Custom features (all other computed features)
            custom_features={k: v for k, v in all_features.items() 
                           if k not in ['order_book_imbalance', 'spread_depth_ratio', 
                                      'price_momentum_short', 'volume_momentum', 'volatility_signal',
                                      'realized_volatility', 'vwap_deviation', 'autocorr_1',
                                      'time_of_day', 'day_of_week', 'market_session']}
        )
        
        # Apply normalization if configured
        if self.config.normalize_features:
            features = self._normalize_features(features, symbol)
        
        return features
    
    def _update_buffers(self, market_data: MarketMicrostructureData) -> None:
        """Update all data buffers with new market data"""
        symbol = market_data.symbol
        
        self.price_buffers[symbol].append(market_data.mid_price)
        self.volume_buffers[symbol].append(market_data.last_volume)
        self.spread_buffers[symbol].append(market_data.spread)
        self.imbalance_buffers[symbol].append(market_data.order_book_imbalance)
        self.trade_direction_buffers[symbol].append(market_data.trade_direction)
    
    def _extract_microstructure_features(self, market_data: MarketMicrostructureData) -> Dict[str, float]:
        """Extract market microstructure features"""
        features = {}
        symbol = market_data.symbol
        
        # Order book imbalance (already calculated in market data)
        features['order_book_imbalance'] = market_data.order_book_imbalance
        
        # Spread-related features
        if market_data.mid_price > 0:
            features['spread_bps'] = (market_data.spread / market_data.mid_price) * 10000
            features['relative_spread'] = market_data.spread / market_data.mid_price
        else:
            features['spread_bps'] = 0.0
            features['relative_spread'] = 0.0
        
        # Order book depth analysis
        if market_data.bid_depth and market_data.ask_depth:
            features.update(self._analyze_order_book_depth(market_data))
        
        # Volume-weighted features
        total_volume = market_data.bid_volume + market_data.ask_volume
        if total_volume > 0:
            features['bid_volume_ratio'] = market_data.bid_volume / total_volume
            features['ask_volume_ratio'] = market_data.ask_volume / total_volume
            features['volume_imbalance'] = (market_data.bid_volume - market_data.ask_volume) / total_volume
        else:
            features['bid_volume_ratio'] = 0.5
            features['ask_volume_ratio'] = 0.5
            features['volume_imbalance'] = 0.0
        
        # Spread depth ratio
        if len(self.spread_buffers[symbol]) > 1:
            recent_spreads = list(self.spread_buffers[symbol])
            avg_spread = np.mean(recent_spreads)
            features['spread_depth_ratio'] = market_data.spread / max(avg_spread, 1e-8)
        else:
            features['spread_depth_ratio'] = 1.0
        
        return features
    
    def _analyze_order_book_depth(self, market_data: MarketMicrostructureData) -> Dict[str, float]:
        """Analyze order book depth features"""
        features = {}
        
        # Calculate cumulative volumes at different levels
        bid_volumes = [vol for _, vol in market_data.bid_depth[:self.config.depth_levels]]
        ask_volumes = [vol for _, vol in market_data.ask_depth[:self.config.depth_levels]]
        
        if bid_volumes and ask_volumes:
            # Depth imbalance at different levels
            for i in range(min(len(bid_volumes), len(ask_volumes), 5)):
                bid_cum_vol = sum(bid_volumes[:i+1])
                ask_cum_vol = sum(ask_volumes[:i+1])
                total_vol = bid_cum_vol + ask_cum_vol
                
                if total_vol > 0:
                    features[f'depth_imbalance_level_{i+1}'] = (bid_cum_vol - ask_cum_vol) / total_vol
                else:
                    features[f'depth_imbalance_level_{i+1}'] = 0.0
            
            # Order book slope (price impact estimation)
            bid_prices = [price for price, _ in market_data.bid_depth[:5]]
            ask_prices = [price for price, _ in market_data.ask_depth[:5]]
            
            if len(bid_prices) >= 2 and len(ask_prices) >= 2:
                bid_slope = np.polyfit(range(len(bid_prices)), bid_prices, 1)[0]
                ask_slope = np.polyfit(range(len(ask_prices)), ask_prices, 1)[0]
                features['bid_slope'] = bid_slope
                features['ask_slope'] = ask_slope
                features['depth_asymmetry'] = ask_slope - bid_slope
        
        return features
    
    def _extract_statistical_features(self, symbol: str) -> Dict[str, float]:
        """Extract statistical features like volatility, autocorrelation"""
        features = {}
        
        if len(self.price_buffers[symbol]) < self.config.short_window:
            return self._get_default_statistical_features()
        
        prices = np.array(list(self.price_buffers[symbol]))
        returns = np.diff(np.log(prices + 1e-8))  # Log returns
        
        # Realized volatility
        if len(returns) >= self.config.volatility_window:
            recent_returns = returns[-self.config.volatility_window:]
            features['realized_volatility'] = np.std(recent_returns) * self.config.volatility_scaling_factor
            features['volatility_signal'] = features['realized_volatility']
        else:
            features['realized_volatility'] = 0.0
            features['volatility_signal'] = 0.0
        
        # Return statistics
        if len(returns) >= self.config.short_window:
            recent_returns = returns[-self.config.short_window:]
            features['return_mean'] = np.mean(recent_returns)
            features['return_std'] = np.std(recent_returns)
            features['return_skewness'] = stats.skew(recent_returns)
            features['return_kurtosis'] = stats.kurtosis(recent_returns)
        
        # Autocorrelation features
        for lag in self.config.autocorr_lags:
            if len(returns) > lag + self.config.short_window:
                recent_returns = returns[-self.config.short_window:]
                if len(recent_returns) > lag:
                    autocorr = np.corrcoef(recent_returns[:-lag], recent_returns[lag:])[0, 1]
                    features[f'autocorr_{lag}'] = autocorr if not np.isnan(autocorr) else 0.0
                else:
                    features[f'autocorr_{lag}'] = 0.0
        
        # VWAP deviation
        if len(self.price_buffers[symbol]) >= self.config.short_window:
            recent_prices = list(self.price_buffers[symbol])[-self.config.short_window:]
            recent_volumes = list(self.volume_buffers[symbol])[-self.config.short_window:]
            
            vwap = self._calculate_vwap(recent_prices, recent_volumes)
            current_price = prices[-1]
            features['vwap_deviation'] = (current_price - vwap) / max(vwap, 1e-8)
        else:
            features['vwap_deviation'] = 0.0
        
        # Volatility clustering (GARCH-like)
        if len(returns) >= self.config.medium_window:
            recent_returns = returns[-self.config.medium_window:]
            volatility_series = pd.Series(recent_returns).rolling(10).std()
            if len(volatility_series.dropna()) > 1:
                vol_autocorr = volatility_series.dropna().autocorr(lag=1)
                features['volatility_clustering'] = vol_autocorr if not np.isnan(vol_autocorr) else 0.0
            else:
                features['volatility_clustering'] = 0.0
        
        return features
    
    def _extract_technical_features(self, symbol: str) -> Dict[str, float]:
        """Extract technical analysis features"""
        features = {}
        
        if len(self.price_buffers[symbol]) < self.config.short_window:
            return self._get_default_technical_features()
        
        prices = np.array(list(self.price_buffers[symbol]))
        volumes = np.array(list(self.volume_buffers[symbol]))
        
        # Price momentum at different timeframes
        for window in self.config.momentum_windows:
            if len(prices) > window:
                momentum = (prices[-1] / prices[-window] - 1) if prices[-window] > 0 else 0
                features[f'price_momentum_{window}'] = momentum
        
        # Volume momentum
        if len(volumes) >= self.config.volume_momentum_window:
            recent_volume = np.mean(volumes[-self.config.volume_momentum_window//2:])
            older_volume = np.mean(volumes[-self.config.volume_momentum_window:-self.config.volume_momentum_window//2])
            
            if older_volume > 0:
                features['volume_momentum'] = (recent_volume / older_volume - 1)
            else:
                features['volume_momentum'] = 0.0
        else:
            features['volume_momentum'] = 0.0
        
        # Moving average features
        if len(prices) >= self.config.long_window:
            sma_short = np.mean(prices[-self.config.short_window:])
            sma_medium = np.mean(prices[-self.config.medium_window:])
            sma_long = np.mean(prices[-self.config.long_window:])
            
            current_price = prices[-1]
            features['price_vs_sma_short'] = (current_price / sma_short - 1) if sma_short > 0 else 0
            features['price_vs_sma_medium'] = (current_price / sma_medium - 1) if sma_medium > 0 else 0
            features['price_vs_sma_long'] = (current_price / sma_long - 1) if sma_long > 0 else 0
            
            # Moving average crossovers
            features['sma_cross_short_medium'] = (sma_short / sma_medium - 1) if sma_medium > 0 else 0
            features['sma_cross_medium_long'] = (sma_medium / sma_long - 1) if sma_long > 0 else 0
        
        # Technical indicators using TA-Lib
        if len(prices) >= 14:  # Minimum for RSI
            try:
                rsi = talib.RSI(prices.astype(float), timeperiod=14)
                features['rsi'] = rsi[-1] if not np.isnan(rsi[-1]) else 50.0
                features['rsi_normalized'] = (rsi[-1] - 50) / 50 if not np.isnan(rsi[-1]) else 0.0
            except:
                features['rsi'] = 50.0
                features['rsi_normalized'] = 0.0
        
        if len(prices) >= 20:  # Minimum for Bollinger Bands
            try:
                bb_upper, bb_middle, bb_lower = talib.BBANDS(prices.astype(float), timeperiod=20)
                current_price = prices[-1]
                bb_width = bb_upper[-1] - bb_lower[-1]
                
                if bb_width > 0:
                    features['bb_position'] = (current_price - bb_middle[-1]) / bb_width
                else:
                    features['bb_position'] = 0.0
                
                features['bb_squeeze'] = bb_width / bb_middle[-1] if bb_middle[-1] > 0 else 0.0
            except:
                features['bb_position'] = 0.0
                features['bb_squeeze'] = 0.0
        
        return features
    
    def _extract_time_based_features(self, market_data: MarketMicrostructureData) -> Dict[str, float]:
        """Extract time-based features"""
        features = {}
        
        # Convert timestamp to datetime
        dt = datetime.fromtimestamp(market_data.timestamp / 1e9)
        
        # Time of day features
        features['time_of_day'] = (dt.hour * 3600 + dt.minute * 60 + dt.second) / 86400
        features['hour_of_day'] = dt.hour / 24
        features['minute_of_hour'] = dt.minute / 60
        features['day_of_week'] = dt.weekday()
        features['day_of_month'] = dt.day / 31
        
        # Market session identification
        hour = dt.hour
        if 9 <= hour < 16:
            features['market_session'] = 'regular'
            features['is_regular_session'] = 1.0
            features['is_pre_market'] = 0.0
            features['is_after_market'] = 0.0
        elif 4 <= hour < 9:
            features['market_session'] = 'pre_market'
            features['is_regular_session'] = 0.0
            features['is_pre_market'] = 1.0
            features['is_after_market'] = 0.0
        elif 16 <= hour < 20:
            features['market_session'] = 'after_market'
            features['is_regular_session'] = 0.0
            features['is_pre_market'] = 0.0
            features['is_after_market'] = 1.0
        else:
            features['market_session'] = 'closed'
            features['is_regular_session'] = 0.0
            features['is_pre_market'] = 0.0
            features['is_after_market'] = 0.0
        
        # Cyclical encoding for periodic features
        features['hour_sin'] = np.sin(2 * np.pi * dt.hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * dt.hour / 24)
        features['dow_sin'] = np.sin(2 * np.pi * dt.weekday() / 7)
        features['dow_cos'] = np.cos(2 * np.pi * dt.weekday() / 7)
        
        return features
    
    def _extract_order_flow_features(self, market_data: MarketMicrostructureData) -> Dict[str, float]:
        """Extract order flow and trade direction features"""
        features = {}
        symbol = market_data.symbol
        
        # Trade direction momentum
        if len(self.trade_direction_buffers[symbol]) >= self.config.short_window:
            recent_directions = list(self.trade_direction_buffers[symbol])[-self.config.short_window:]
            
            # Trade direction bias
            buy_trades = sum(1 for d in recent_directions if d > 0)
            sell_trades = sum(1 for d in recent_directions if d < 0)
            total_trades = len([d for d in recent_directions if d != 0])
            
            if total_trades > 0:
                features['trade_direction_bias'] = (buy_trades - sell_trades) / total_trades
                features['trade_direction_momentum'] = np.mean(recent_directions)
            else:
                features['trade_direction_bias'] = 0.0
                features['trade_direction_momentum'] = 0.0
        
        # Order book imbalance momentum
        if len(self.imbalance_buffers[symbol]) >= self.config.short_window:
            recent_imbalances = list(self.imbalance_buffers[symbol])[-self.config.short_window:]
            features['imbalance_momentum'] = np.mean(recent_imbalances)
            features['imbalance_volatility'] = np.std(recent_imbalances)
            
            # Imbalance persistence
            if len(recent_imbalances) > 1:
                imbalance_changes = np.diff(recent_imbalances)
                features['imbalance_persistence'] = np.mean(np.sign(imbalance_changes))
            else:
                features['imbalance_persistence'] = 0.0
        
        return features
    
    def _calculate_vwap(self, prices: List[float], volumes: List[float]) -> float:
        """Calculate volume-weighted average price"""
        if not prices or not volumes or len(prices) != len(volumes):
            return prices[-1] if prices else 0.0
        
        # Filter out zero volumes
        price_vol_pairs = [(p, v) for p, v in zip(prices, volumes) if v > 0]
        
        if not price_vol_pairs:
            return prices[-1]
        
        total_value = sum(p * v for p, v in price_vol_pairs)
        total_volume = sum(v for _, v in price_vol_pairs)
        
        return total_value / total_volume if total_volume > 0 else prices[-1]
    
    def _normalize_features(self, features: MLFeatures, symbol: str) -> MLFeatures:
        """Apply normalization to features"""
        if symbol not in self.scalers:
            return features
        
        # Convert features to array format for scaling
        feature_dict = {
            'order_book_imbalance': features.order_book_imbalance,
            'spread_depth_ratio': features.spread_depth_ratio,
            'price_momentum': features.price_momentum,
            'volume_momentum': features.volume_momentum,
            'volatility_signal': features.volatility_signal,
            'realized_volatility': features.realized_volatility,
            'vwap_deviation': features.vwap_deviation,
            'autocorrelation': features.autocorrelation,
        }
        feature_dict.update(features.custom_features)
        
        # Create feature array
        feature_names = sorted(feature_dict.keys())
        feature_array = np.array([feature_dict[name] for name in feature_names]).reshape(1, -1)
        
        # Apply scaling
        try:
            # For online learning, we update the scaler incrementally
            if hasattr(self.scalers[symbol], 'partial_fit'):
                self.scalers[symbol].partial_fit(feature_array)
            else:
                # Update with exponential weighting for RobustScaler
                pass
            
            scaled_features = self.scalers[symbol].transform(feature_array)[0]
            
            # Update feature values
            scaled_dict = dict(zip(feature_names, scaled_features))
            
            return MLFeatures(
                order_book_imbalance=scaled_dict.get('order_book_imbalance', features.order_book_imbalance),
                spread_depth_ratio=scaled_dict.get('spread_depth_ratio', features.spread_depth_ratio),
                price_momentum=scaled_dict.get('price_momentum', features.price_momentum),
                volume_momentum=scaled_dict.get('volume_momentum', features.volume_momentum),
                volatility_signal=scaled_dict.get('volatility_signal', features.volatility_signal),
                realized_volatility=scaled_dict.get('realized_volatility', features.realized_volatility),
                vwap_deviation=scaled_dict.get('vwap_deviation', features.vwap_deviation),
                autocorrelation=scaled_dict.get('autocorrelation', features.autocorrelation),
                time_of_day=features.time_of_day,  # Don't normalize time features
                day_of_week=features.day_of_week,
                market_session=features.market_session,
                custom_features={k: v for k, v in scaled_dict.items() 
                               if k not in ['order_book_imbalance', 'spread_depth_ratio', 'price_momentum',
                                          'volume_momentum', 'volatility_signal', 'realized_volatility',
                                          'vwap_deviation', 'autocorrelation']}
            )
            
        except Exception as e:
            logger.warning(f"Feature normalization failed for {symbol}: {e}")
            return features
    
    def _get_default_statistical_features(self) -> Dict[str, float]:
        """Return default values for statistical features when insufficient data"""
        return {
            'realized_volatility': 0.0,
            'volatility_signal': 0.0,
            'return_mean': 0.0,
            'return_std': 0.0,
            'return_skewness': 0.0,
            'return_kurtosis': 0.0,
            'vwap_deviation': 0.0,
            'volatility_clustering': 0.0,
            **{f'autocorr_{lag}': 0.0 for lag in self.config.autocorr_lags}
        }
    
    def _get_default_technical_features(self) -> Dict[str, float]:
        """Return default values for technical features when insufficient data"""
        default_features = {
            'volume_momentum': 0.0,
            'rsi': 50.0,
            'rsi_normalized': 0.0,
            'bb_position': 0.0,
            'bb_squeeze': 0.0,
        }
        
        # Add momentum features
        for window in self.config.momentum_windows:
            default_features[f'price_momentum_{window}'] = 0.0
        
        return default_features
    
    def get_feature_importance(self, symbol: str) -> Dict[str, float]:
        """Get feature importance scores (placeholder for now)"""
        # This would be updated based on model training results
        return {}
    
    def reset_buffers(self, symbol: Optional[str] = None) -> None:
        """Reset feature calculation buffers"""
        if symbol:
            symbols_to_reset = [symbol]
        else:
            symbols_to_reset = self.symbols
        
        for sym in symbols_to_reset:
            if sym in self.price_buffers:
                self.price_buffers[sym].clear()
                self.volume_buffers[sym].clear()
                self.spread_buffers[sym].clear()
                self.imbalance_buffers[sym].clear()
                self.trade_direction_buffers[sym].clear()
                self.feature_cache[sym].clear()
                self.last_calculation_timestamp[sym] = 0
        
        logger.info(f"Reset feature buffers for {symbols_to_reset}")
