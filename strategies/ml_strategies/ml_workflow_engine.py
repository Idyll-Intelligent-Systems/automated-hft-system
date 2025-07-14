"""
ML Workflow Engine
================

Orchestrates the complete ML modeling logic workflow for HFT systems as defined in Modeling.md.
This engine manages the flow from real-time data ingestion through continuous learning.

Workflow Steps:
1. Real-Time Market Data → 2. Data Cleaning & Feature Engineering → 
3. Statistical Patterns & Relationships Identification → 4. Model Formulation → 
5. Model Calibration → 6. Real-time Prediction → 7. Optimal Order Placement → 
8. Continuous Learning & Model Adaptation
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading
import time

from .base_ml_strategy import (
    BaseMLStrategy, MarketMicrostructureData, MLFeatures, MLPrediction,
    ModelType, PredictionType, RiskConstraints
)

logger = logging.getLogger(__name__)

@dataclass
class WorkflowConfig:
    """Configuration for ML workflow engine"""
    # Data processing
    max_data_buffer_size: int = 10000
    feature_calculation_window: int = 100
    min_data_points_for_training: int = 1000
    
    # Model management
    model_retrain_interval_minutes: int = 60
    model_calibration_window_hours: int = 24
    prediction_confidence_threshold: float = 0.6
    
    # Performance monitoring
    performance_evaluation_interval_minutes: int = 15
    max_prediction_history: int = 50000
    
    # Risk management
    max_concurrent_predictions: int = 10
    order_placement_cooldown_ms: int = 100
    
    # Continuous learning
    adaptive_learning_enabled: bool = True
    model_performance_window_size: int = 1000
    performance_degradation_threshold: float = 0.05

@dataclass
class WorkflowMetrics:
    """Metrics for workflow performance monitoring"""
    total_data_processed: int = 0
    total_predictions_generated: int = 0
    total_orders_placed: int = 0
    model_retrains: int = 0
    avg_prediction_latency_ns: float = 0.0
    avg_feature_engineering_latency_ns: float = 0.0
    current_prediction_accuracy: float = 0.0
    current_model_confidence: float = 0.0

class MLWorkflowEngine:
    """
    Main engine orchestrating the ML modeling workflow for HFT strategies
    
    This engine implements the complete workflow from Modeling.md:
    - Manages real-time data flow
    - Coordinates feature engineering and pattern identification
    - Handles model training, calibration, and adaptation
    - Orchestrates prediction generation and order placement
    - Implements continuous learning loops
    """
    
    def __init__(self, 
                 strategy: BaseMLStrategy,
                 config: WorkflowConfig,
                 data_callback: Optional[Callable] = None,
                 order_callback: Optional[Callable] = None):
        self.strategy = strategy
        self.config = config
        self.data_callback = data_callback
        self.order_callback = order_callback
        
        # Data management
        self.data_buffer = deque(maxlen=config.max_data_buffer_size)
        self.feature_buffer = deque(maxlen=config.max_data_buffer_size)
        self.prediction_buffer = deque(maxlen=config.max_prediction_history)
        
        # Model state
        self.current_model = None
        self.model_last_trained = None
        self.model_last_calibrated = None
        self.model_performance_history = deque(maxlen=config.model_performance_window_size)
        
        # Threading and async management
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.running = False
        self.workflow_lock = threading.RLock()
        
        # Metrics and monitoring
        self.metrics = WorkflowMetrics()
        self.start_time = datetime.now()
        
        # Internal state
        self.last_retrain_check = datetime.now()
        self.last_performance_evaluation = datetime.now()
        self.pending_predictions = {}
        
        logger.info(f"Initialized ML workflow engine for strategy {strategy.strategy_id}")
    
    async def start_workflow(self) -> None:
        """Start the ML workflow engine"""
        if self.running:
            logger.warning("Workflow already running")
            return
        
        self.running = True
        logger.info("Starting ML workflow engine")
        
        # Start background tasks
        background_tasks = [
            self._model_management_loop(),
            self._performance_monitoring_loop(),
            self._continuous_learning_loop()
        ]
        
        try:
            await asyncio.gather(*background_tasks)
        except Exception as e:
            logger.error(f"Error in workflow engine: {e}")
            await self.stop_workflow()
    
    async def stop_workflow(self) -> None:
        """Stop the ML workflow engine"""
        logger.info("Stopping ML workflow engine")
        self.running = False
        self.executor.shutdown(wait=True)
    
    # Step 1: Real-Time Market Data Processing
    async def process_market_data(self, market_data: MarketMicrostructureData) -> bool:
        """
        Process incoming real-time market data through the workflow
        
        This is the entry point for the ML workflow - data flows through:
        1. Data validation and cleaning
        2. Feature engineering
        3. Statistical pattern identification
        4. Prediction generation (if model is ready)
        5. Order placement logic (if prediction meets criteria)
        """
        start_time = time.perf_counter_ns()
        
        try:
            # Step 1: Process raw market data
            if not self.strategy.process_market_data(market_data):
                return False
            
            # Buffer the data
            self.data_buffer.append(market_data)
            self.metrics.total_data_processed += 1
            
            # Step 2: Feature engineering
            features = await self._engineer_features_async(market_data)
            if features is None:
                return False
            
            self.feature_buffer.append(features)
            
            # Step 3: Statistical pattern identification
            patterns = await self._identify_patterns_async(features)
            
            # Step 4: Generate prediction if model is available
            if self.current_model is not None:
                prediction = await self._generate_prediction_async(features)
                if prediction is not None:
                    self.prediction_buffer.append(prediction)
                    self.metrics.total_predictions_generated += 1
                    
                    # Step 5: Order placement logic
                    if prediction.confidence >= self.config.prediction_confidence_threshold:
                        await self._execute_order_placement_logic(prediction, market_data)
            
            # Update latency metrics
            total_latency = time.perf_counter_ns() - start_time
            self._update_latency_metrics(total_latency)
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing market data: {e}")
            return False
    
    async def _engineer_features_async(self, market_data: MarketMicrostructureData) -> Optional[MLFeatures]:
        """Async wrapper for feature engineering"""
        feature_start = time.perf_counter_ns()
        
        try:
            # Run feature engineering in thread pool to avoid blocking
            features = await asyncio.get_event_loop().run_in_executor(
                self.executor, 
                self.strategy.engineer_features, 
                market_data
            )
            
            feature_latency = time.perf_counter_ns() - feature_start
            self.metrics.avg_feature_engineering_latency_ns = (
                self.metrics.avg_feature_engineering_latency_ns * 0.95 + feature_latency * 0.05
            )
            
            return features
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {e}")
            return None
    
    async def _identify_patterns_async(self, features: MLFeatures) -> Optional[Dict[str, float]]:
        """Async wrapper for statistical pattern identification"""
        try:
            return await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.strategy.identify_statistical_patterns,
                features
            )
        except Exception as e:
            logger.error(f"Error in pattern identification: {e}")
            return None
    
    async def _generate_prediction_async(self, features: MLFeatures) -> Optional[MLPrediction]:
        """Async wrapper for prediction generation"""
        prediction_start = time.perf_counter_ns()
        
        try:
            prediction = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.strategy.generate_prediction,
                features
            )
            
            prediction_latency = time.perf_counter_ns() - prediction_start
            self.metrics.avg_prediction_latency_ns = (
                self.metrics.avg_prediction_latency_ns * 0.95 + prediction_latency * 0.05
            )
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error in prediction generation: {e}")
            return None
    
    async def _execute_order_placement_logic(self, 
                                           prediction: MLPrediction,
                                           market_data: MarketMicrostructureData) -> None:
        """Execute optimal order placement logic"""
        try:
            # Get current position (would come from position manager)
            current_position = 0.0  # Placeholder
            
            # Calculate optimal orders
            orders = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.strategy.calculate_optimal_orders,
                prediction,
                current_position,
                market_data
            )
            
            # Execute orders through callback
            if orders and self.order_callback:
                await self.order_callback(orders)
                self.metrics.total_orders_placed += len(orders)
                
        except Exception as e:
            logger.error(f"Error in order placement: {e}")
    
    # Background workflow management tasks
    async def _model_management_loop(self) -> None:
        """Background task for model training and calibration"""
        while self.running:
            try:
                current_time = datetime.now()
                
                # Check if we need to retrain the model
                if (self.model_last_trained is None or 
                    current_time - self.model_last_trained > timedelta(minutes=self.config.model_retrain_interval_minutes)):
                    
                    if len(self.data_buffer) >= self.config.min_data_points_for_training:
                        await self._retrain_model()
                
                # Check if we need to recalibrate
                if (self.model_last_calibrated is None or
                    current_time - self.model_last_calibrated > timedelta(hours=self.config.model_calibration_window_hours)):
                    
                    if self.current_model is not None:
                        await self._recalibrate_model()
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in model management loop: {e}")
                await asyncio.sleep(60)
    
    async def _performance_monitoring_loop(self) -> None:
        """Background task for performance monitoring"""
        while self.running:
            try:
                current_time = datetime.now()
                
                if (current_time - self.last_performance_evaluation > 
                    timedelta(minutes=self.config.performance_evaluation_interval_minutes)):
                    
                    await self._evaluate_performance()
                    self.last_performance_evaluation = current_time
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in performance monitoring loop: {e}")
                await asyncio.sleep(30)
    
    async def _continuous_learning_loop(self) -> None:
        """Background task for continuous learning and adaptation"""
        while self.running:
            try:
                if self.config.adaptive_learning_enabled and self.current_model is not None:
                    # Check if model performance is degrading
                    if self.strategy.should_retrain_model():
                        logger.info("Performance degradation detected, scheduling model retraining")
                        await self._retrain_model()
                    
                    # Adaptive parameter tuning
                    recent_performance = self._calculate_recent_performance()
                    if recent_performance:
                        adaptation_success = await asyncio.get_event_loop().run_in_executor(
                            self.executor,
                            self.strategy.adapt_model,
                            recent_performance
                        )
                        
                        if adaptation_success:
                            logger.info("Model adaptation completed successfully")
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in continuous learning loop: {e}")
                await asyncio.sleep(300)
    
    async def _retrain_model(self) -> None:
        """Retrain the ML model with recent data"""
        logger.info("Starting model retraining")
        
        try:
            with self.workflow_lock:
                # Prepare training data from buffer
                training_data = self._prepare_training_data()
                
                if training_data is not None and len(training_data) >= self.config.min_data_points_for_training:
                    # Train new model
                    new_model = await asyncio.get_event_loop().run_in_executor(
                        self.executor,
                        self.strategy.formulate_model,
                        training_data
                    )
                    
                    if new_model is not None:
                        self.current_model = new_model
                        self.model_last_trained = datetime.now()
                        self.metrics.model_retrains += 1
                        
                        logger.info("Model retraining completed successfully")
                    else:
                        logger.error("Model retraining failed")
                else:
                    logger.warning("Insufficient data for model retraining")
                    
        except Exception as e:
            logger.error(f"Error during model retraining: {e}")
    
    async def _recalibrate_model(self) -> None:
        """Recalibrate the current model"""
        logger.info("Starting model recalibration")
        
        try:
            # Prepare validation data
            validation_data = self._prepare_validation_data()
            
            if validation_data is not None and len(validation_data) > 100:
                calibration_result = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    self.strategy.calibrate_model,
                    self.current_model,
                    validation_data
                )
                
                if calibration_result is not None:
                    self.model_last_calibrated = datetime.now()
                    self.model_performance_history.append(calibration_result)
                    
                    # Update metrics
                    self.metrics.current_prediction_accuracy = getattr(calibration_result, 'accuracy', 0.0) or 0.0
                    
                    logger.info(f"Model recalibration completed. Sharpe ratio: {calibration_result.sharpe_ratio:.2f}")
                else:
                    logger.error("Model recalibration failed")
            else:
                logger.warning("Insufficient data for model recalibration")
                
        except Exception as e:
            logger.error(f"Error during model recalibration: {e}")
    
    async def _evaluate_performance(self) -> None:
        """Evaluate current model performance"""
        try:
            diagnostics = self.strategy.get_model_diagnostics()
            
            if diagnostics:
                if 'accuracy' in diagnostics:
                    self.metrics.current_prediction_accuracy = diagnostics['accuracy']
                
                # Log performance metrics
                logger.info(f"Model diagnostics: {diagnostics}")
                
        except Exception as e:
            logger.error(f"Error evaluating performance: {e}")
    
    def _prepare_training_data(self) -> Optional[pd.DataFrame]:
        """Prepare training data from buffers"""
        try:
            if len(self.feature_buffer) < self.config.min_data_points_for_training:
                return None
            
            # Convert feature buffer to DataFrame
            features_list = []
            for features in self.feature_buffer:
                feature_dict = {
                    'order_book_imbalance': features.order_book_imbalance,
                    'spread_depth_ratio': features.spread_depth_ratio,
                    'price_momentum': features.price_momentum,
                    'volume_momentum': features.volume_momentum,
                    'volatility_signal': features.volatility_signal,
                    'realized_volatility': features.realized_volatility,
                    'vwap_deviation': features.vwap_deviation,
                    'autocorrelation': features.autocorrelation,
                    'time_of_day': features.time_of_day,
                    'day_of_week': features.day_of_week,
                }
                feature_dict.update(features.custom_features)
                features_list.append(feature_dict)
            
            return pd.DataFrame(features_list)
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return None
    
    def _prepare_validation_data(self) -> Optional[pd.DataFrame]:
        """Prepare validation data from recent history"""
        # Similar to training data but from more recent history
        return self._prepare_training_data()
    
    def _calculate_recent_performance(self) -> Optional[Dict[str, float]]:
        """Calculate recent performance metrics"""
        try:
            if len(self.prediction_buffer) < 100:
                return None
            
            recent_predictions = list(self.prediction_buffer)[-100:]
            
            # Calculate basic performance metrics
            confidences = [p.confidence for p in recent_predictions]
            avg_confidence = np.mean(confidences)
            
            return {
                'avg_confidence': avg_confidence,
                'prediction_count': len(recent_predictions),
                'avg_latency_ns': self.metrics.avg_prediction_latency_ns
            }
            
        except Exception as e:
            logger.error(f"Error calculating recent performance: {e}")
            return None
    
    def _update_latency_metrics(self, total_latency_ns: float) -> None:
        """Update latency tracking metrics"""
        # Exponential moving average for latency
        self.metrics.avg_prediction_latency_ns = (
            self.metrics.avg_prediction_latency_ns * 0.95 + total_latency_ns * 0.05
        )
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """Get current workflow status and metrics"""
        uptime = datetime.now() - self.start_time
        
        return {
            'running': self.running,
            'uptime_seconds': uptime.total_seconds(),
            'strategy_id': self.strategy.strategy_id,
            'model_trained': self.current_model is not None,
            'last_retrain': self.model_last_trained.isoformat() if self.model_last_trained else None,
            'last_calibration': self.model_last_calibrated.isoformat() if self.model_last_calibrated else None,
            'data_buffer_size': len(self.data_buffer),
            'feature_buffer_size': len(self.feature_buffer),
            'prediction_buffer_size': len(self.prediction_buffer),
            'metrics': {
                'total_data_processed': self.metrics.total_data_processed,
                'total_predictions_generated': self.metrics.total_predictions_generated,
                'total_orders_placed': self.metrics.total_orders_placed,
                'model_retrains': self.metrics.model_retrains,
                'avg_prediction_latency_ns': self.metrics.avg_prediction_latency_ns,
                'avg_feature_engineering_latency_ns': self.metrics.avg_feature_engineering_latency_ns,
                'current_prediction_accuracy': self.metrics.current_prediction_accuracy,
                'current_model_confidence': self.metrics.current_model_confidence
            }
        }
