"""
Base ML Strategy Framework
========================

Abstract base class implementing the core ML modeling logic workflow for HFT systems
following the first principles approach from Modeling.md:

1. Real-Time Market Data Processing
2. Data Cleaning & Feature Engineering  
3. Statistical Patterns & Relationships Identification
4. Model Formulation (Statistical / ML-Based)
5. Model Calibration (Backtest & Optimize)
6. Real-time Prediction/Classification
7. Optimal Order Placement Logic
8. Continuous Learning & Model Adaptation
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Types of ML models for HFT"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    TIME_SERIES = "time_series"
    STATISTICAL = "statistical"

class PredictionType(Enum):
    """Types of predictions"""
    MARKET_DIRECTION = "market_direction"
    SPREAD_PREDICTION = "spread_prediction"
    ORDER_FLOW = "order_flow"
    LIQUIDITY_CONSUMPTION = "liquidity_consumption"
    PRICE_REVERSAL = "price_reversal"

@dataclass
class MarketMicrostructureData:
    """Market microstructure data for ML processing"""
    timestamp: int
    symbol: str
    # Order book data
    best_bid: float
    best_ask: float
    bid_volume: float
    ask_volume: float
    bid_depth: List[Tuple[float, float]]  # (price, volume) pairs
    ask_depth: List[Tuple[float, float]]  # (price, volume) pairs
    # Trade data
    last_price: float
    last_volume: float
    trade_direction: int  # 1 for buy, -1 for sell, 0 for unknown
    # Calculated features
    mid_price: float
    spread: float
    order_book_imbalance: float
    volume_weighted_average_price: float

@dataclass
class MLFeatures:
    """Engineered features for ML models"""
    # Microstructure features
    order_book_imbalance: float
    spread_depth_ratio: float
    price_momentum: float
    volume_momentum: float
    volatility_signal: float
    # Statistical features  
    realized_volatility: float
    vwap_deviation: float
    autocorrelation: float
    # Time-based features
    time_of_day: float
    day_of_week: int
    market_session: str
    # Custom features
    custom_features: Dict[str, float] = field(default_factory=dict)

@dataclass
class MLPrediction:
    """ML model prediction output"""
    timestamp: int
    symbol: str
    prediction_type: PredictionType
    prediction_value: Union[float, int, str]
    confidence: float
    feature_importance: Dict[str, float]
    model_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RiskConstraints:
    """Risk management constraints for ML strategies"""
    max_position: float
    max_daily_loss: float
    max_leverage: float
    var_limit: float
    max_drawdown: float
    concentration_limit: float
    stop_loss_pct: float

@dataclass
class ModelCalibrationResult:
    """Results from model calibration and backtesting"""
    model_id: str
    calibration_period: Tuple[datetime, datetime]
    # Performance metrics
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    total_return: float
    win_rate: float
    profit_factor: float
    # Model metrics
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    mse: Optional[float] = None
    mae: Optional[float] = None
    # Additional metadata
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    feature_importance: Dict[str, float] = field(default_factory=dict)

class BaseMLStrategy(ABC):
    """
    Abstract base class for ML-based HFT strategies
    
    Implements the complete ML modeling logic workflow:
    1. Data processing and feature engineering
    2. Statistical pattern identification
    3. Model formulation and training
    4. Model calibration and backtesting
    5. Real-time prediction generation
    6. Optimal order placement
    7. Continuous learning and adaptation
    """
    
    def __init__(self, 
                 strategy_id: str,
                 symbols: List[str],
                 model_type: ModelType,
                 prediction_type: PredictionType,
                 risk_constraints: RiskConstraints,
                 config: Dict[str, Any]):
        self.strategy_id = strategy_id
        self.symbols = symbols
        self.model_type = model_type
        self.prediction_type = prediction_type
        self.risk_constraints = risk_constraints
        self.config = config
        
        # Internal state
        self.is_initialized = False
        self.model = None
        self.feature_buffer = {}
        self.prediction_history = []
        self.performance_metrics = {}
        self.last_calibration = None
        
        logger.info(f"Initialized ML strategy {strategy_id} with type {model_type.value}")
    
    # Step 1: Real-Time Market Data Processing
    @abstractmethod
    def process_market_data(self, market_data: MarketMicrostructureData) -> bool:
        """
        Process incoming real-time market data
        
        Args:
            market_data: Raw market microstructure data
            
        Returns:
            bool: True if data was processed successfully
        """
        pass
    
    # Step 2: Data Cleaning & Feature Engineering
    @abstractmethod
    def engineer_features(self, market_data: MarketMicrostructureData) -> MLFeatures:
        """
        Extract meaningful signals from market data
        
        Implements feature engineering for:
        - Order book imbalance, trade momentum, volatility signals
        - Market microstructure indicators (book imbalance, spread depth)
        - Statistical indicators (realized volatility, VWAP deviation)
        
        Args:
            market_data: Processed market data
            
        Returns:
            MLFeatures: Engineered features for ML models
        """
        pass
    
    # Step 3: Statistical Patterns & Relationships Identification
    @abstractmethod
    def identify_statistical_patterns(self, features: MLFeatures) -> Dict[str, float]:
        """
        Identify statistical patterns using:
        - Cointegration, autocorrelation, volatility clustering
        - Microstructure noise detection
        
        Args:
            features: Engineered features
            
        Returns:
            Dict[str, float]: Statistical pattern indicators
        """
        pass
    
    # Step 4: Model Formulation (Statistical / ML-Based)
    @abstractmethod
    def formulate_model(self, training_data: pd.DataFrame) -> Any:
        """
        Develop predictive models (classification, regression, or RL)
        to anticipate micro-price moves, liquidity consumption, or short-term reversions
        
        Models can include:
        - Statistical learning: Linear regression, logistic regression
        - ML models: Gradient Boosting, Random Forests, Neural Networks
        - Deep RL: PPO, DQN, A3C
        
        Args:
            training_data: Historical data for model training
            
        Returns:
            Any: Trained ML model
        """
        pass
    
    # Step 5: Model Calibration (Backtest & Optimize)
    @abstractmethod
    def calibrate_model(self, 
                       model: Any, 
                       validation_data: pd.DataFrame) -> ModelCalibrationResult:
        """
        Fit the model precisely using past tick data, simulated execution,
        order placement costs, slippage estimation
        
        Includes:
        - Simulation of tick-by-tick historical market data
        - Realistic execution-cost models (slippage, latency, market impact)
        
        Args:
            model: Trained model to calibrate
            validation_data: Validation dataset
            
        Returns:
            ModelCalibrationResult: Calibration results and performance metrics
        """
        pass
    
    # Step 6: Real-time Prediction/Classification
    @abstractmethod
    def generate_prediction(self, features: MLFeatures) -> MLPrediction:
        """
        Generate immediate predictions (microsecond level) using optimized models
        
        Prediction types:
        - Market Direction
        - Spread Prediction  
        - Order-Flow
        
        Args:
            features: Current market features
            
        Returns:
            MLPrediction: Model prediction with confidence
        """
        pass
    
    # Step 7: Optimal Order Placement Logic
    @abstractmethod
    def calculate_optimal_orders(self, 
                               prediction: MLPrediction,
                               current_position: float,
                               market_data: MarketMicrostructureData) -> List[Dict[str, Any]]:
        """
        Decide precise entry and exit levels based on model's forecast,
        risk management criteria, and position inventory constraints
        
        Implements:
        - Avellaneda-Stoikov stochastic control
        - Almgren-Chriss cost-minimization frameworks
        - RL-based execution
        
        Args:
            prediction: Model prediction
            current_position: Current inventory position
            market_data: Current market state
            
        Returns:
            List[Dict]: List of optimal orders to place
        """
        pass
    
    # Step 8: Continuous Learning & Model Adaptation
    @abstractmethod
    def adapt_model(self, recent_performance: Dict[str, float]) -> bool:
        """
        Models dynamically adapt to regime shifts in market microstructure,
        volatility conditions, or liquidity profiles
        
        Includes:
        - Real-time parameter tuning (Bayesian optimization)
        - Model retraining on rolling window (intraday, daily)
        - Monitoring performance metrics (Sharpe, drawdown, latency)
        
        Args:
            recent_performance: Recent performance metrics
            
        Returns:
            bool: True if model was adapted successfully
        """
        pass
    
    # Utility methods
    def validate_risk_constraints(self, 
                                 proposed_position: float, 
                                 current_pnl: float) -> bool:
        """Validate proposed position against risk constraints"""
        # Check position limits
        if abs(proposed_position) > self.risk_constraints.max_position:
            logger.warning(f"Position {proposed_position} exceeds max position {self.risk_constraints.max_position}")
            return False
        
        # Check daily loss limit
        if current_pnl < -self.risk_constraints.max_daily_loss:
            logger.warning(f"Daily loss {current_pnl} exceeds limit {self.risk_constraints.max_daily_loss}")
            return False
        
        return True
    
    def update_performance_metrics(self, 
                                  prediction: MLPrediction,
                                  actual_outcome: float) -> None:
        """Update model performance tracking"""
        # Store prediction vs actual for later analysis
        self.prediction_history.append({
            'timestamp': prediction.timestamp,
            'predicted': prediction.prediction_value,
            'actual': actual_outcome,
            'confidence': prediction.confidence
        })
        
        # Keep only recent history (e.g., last 10,000 predictions)
        if len(self.prediction_history) > 10000:
            self.prediction_history = self.prediction_history[-10000:]
    
    def get_model_diagnostics(self) -> Dict[str, Any]:
        """Get current model diagnostic information"""
        if not self.prediction_history:
            return {}
        
        predictions = pd.DataFrame(self.prediction_history)
        
        # Calculate accuracy metrics
        if self.prediction_type == PredictionType.MARKET_DIRECTION:
            # For classification
            correct_predictions = (np.sign(predictions['predicted']) == np.sign(predictions['actual'])).sum()
            accuracy = correct_predictions / len(predictions)
            return {'accuracy': accuracy, 'total_predictions': len(predictions)}
        else:
            # For regression
            mse = np.mean((predictions['predicted'] - predictions['actual']) ** 2)
            mae = np.mean(np.abs(predictions['predicted'] - predictions['actual']))
            return {'mse': mse, 'mae': mae, 'total_predictions': len(predictions)}
    
    def should_retrain_model(self) -> bool:
        """Determine if model needs retraining based on performance degradation"""
        if not self.prediction_history or len(self.prediction_history) < 100:
            return False
        
        # Simple performance degradation check
        recent_predictions = pd.DataFrame(self.prediction_history[-100:])
        older_predictions = pd.DataFrame(self.prediction_history[-200:-100]) if len(self.prediction_history) >= 200 else None
        
        if older_predictions is None:
            return False
        
        # Compare recent vs older performance
        if self.prediction_type == PredictionType.MARKET_DIRECTION:
            recent_accuracy = (np.sign(recent_predictions['predicted']) == np.sign(recent_predictions['actual'])).mean()
            older_accuracy = (np.sign(older_predictions['predicted']) == np.sign(older_predictions['actual'])).mean()
            
            # Retrain if accuracy dropped by more than 5%
            return recent_accuracy < older_accuracy - 0.05
        else:
            recent_mse = np.mean((recent_predictions['predicted'] - recent_predictions['actual']) ** 2)
            older_mse = np.mean((older_predictions['predicted'] - older_predictions['actual']) ** 2)
            
            # Retrain if MSE increased by more than 20%
            return recent_mse > older_mse * 1.2
