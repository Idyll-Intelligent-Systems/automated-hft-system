"""
ML Strategy Integration Module
=============================

Main integration point for ML strategies in the HFT system.
Provides orchestration, lifecycle management, and integration
with the broader trading system.

This module implements the complete ML workflow from Modeling.md
and provides seamless integration with:
- Live market data feeds
- Order management system
- Risk management
- Performance monitoring
- Model persistence and versioning
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Type
from pathlib import Path
import pickle
import json
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict, deque

from .base_ml_strategy import (
    BaseMLStrategy, MarketMicrostructureData, MLFeatures, MLPrediction,
    ModelType, PredictionType, RiskConstraints
)
from .ml_workflow_engine import MLWorkflowEngine
from .examples.market_direction_strategy import MarketDirectionStrategy

logger = logging.getLogger(__name__)

class MLStrategyManager:
    """
    Central manager for all ML strategies in the HFT system
    
    Responsibilities:
    - Strategy lifecycle management (creation, deployment, monitoring, retirement)
    - Real-time data orchestration
    - Model versioning and persistence
    - Performance aggregation and reporting
    - Risk monitoring across all strategies
    - Resource allocation and optimization
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.strategies: Dict[str, BaseMLStrategy] = {}
        self.workflow_engines: Dict[str, MLWorkflowEngine] = {}
        
        # Data management
        self.market_data_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.prediction_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=5000))
        
        # Performance tracking
        self.strategy_performance: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.system_metrics = {
            'total_predictions': 0,
            'average_latency_ms': 0.0,
            'memory_usage_mb': 0.0,
            'active_strategies': 0
        }
        
        # Threading and async management
        self.executor = ThreadPoolExecutor(max_workers=config.get('max_workers', 4))
        self.is_running = False
        
        # Model persistence
        self.model_storage_path = Path(config.get('model_storage_path', 'models/ml_strategies'))
        self.model_storage_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("MLStrategyManager initialized")
    
    async def start(self) -> None:
        """Start the ML strategy manager and all active strategies"""
        try:
            self.is_running = True
            
            # Load persisted strategies
            await self._load_persisted_strategies()
            
            # Start monitoring tasks
            monitoring_tasks = [
                self._monitor_system_health(),
                self._monitor_strategy_performance(),
                self._manage_model_lifecycle(),
                self._aggregate_predictions()
            ]
            
            await asyncio.gather(*monitoring_tasks, return_exceptions=True)
            
            logger.info("MLStrategyManager started successfully")
            
        except Exception as e:
            logger.error(f"Error starting MLStrategyManager: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop all strategies and clean up resources"""
        try:
            self.is_running = False
            
            # Stop all strategies
            for strategy_id, strategy in self.strategies.items():
                logger.info(f"Stopping strategy: {strategy_id}")
                # Assuming strategies have a stop method
                if hasattr(strategy, 'stop'):
                    await strategy.stop()
            
            # Save strategies state
            await self._persist_strategies()
            
            # Shutdown executor
            self.executor.shutdown(wait=True)
            
            logger.info("MLStrategyManager stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping MLStrategyManager: {e}")
    
    def register_strategy(self, 
                         strategy_class: Type[BaseMLStrategy],
                         strategy_id: str,
                         symbols: List[str],
                         config: Dict[str, Any]) -> str:
        """Register a new ML strategy"""
        try:
            # Create strategy instance
            strategy = strategy_class(symbols=symbols, config=config)
            
            # Create workflow engine for the strategy
            workflow_engine = MLWorkflowEngine(strategy, config)
            
            # Register strategy
            self.strategies[strategy_id] = strategy
            self.workflow_engines[strategy_id] = workflow_engine
            
            # Initialize performance tracking
            self.strategy_performance[strategy_id] = {
                'total_predictions': 0,
                'correct_predictions': 0,
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'last_updated': datetime.now().isoformat()
            }
            
            logger.info(f"Registered strategy: {strategy_id} for symbols: {symbols}")
            return strategy_id
            
        except Exception as e:
            logger.error(f"Error registering strategy {strategy_id}: {e}")
            raise
    
    def unregister_strategy(self, strategy_id: str) -> bool:
        """Unregister and remove a strategy"""
        try:
            if strategy_id in self.strategies:
                # Save strategy state before removal
                self._save_strategy_state(strategy_id)
                
                # Remove from active strategies
                del self.strategies[strategy_id]
                del self.workflow_engines[strategy_id]
                del self.strategy_performance[strategy_id]
                
                logger.info(f"Unregistered strategy: {strategy_id}")
                return True
            else:
                logger.warning(f"Strategy not found: {strategy_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error unregistering strategy {strategy_id}: {e}")
            return False
    
    async def process_market_data(self, market_data: MarketMicrostructureData) -> Dict[str, Any]:
        """Process incoming market data through all relevant strategies"""
        try:
            symbol = market_data.symbol
            processing_results = {}
            
            # Store market data
            self.market_data_buffer[symbol].append(market_data)
            
            # Process through each strategy that trades this symbol
            for strategy_id, strategy in self.strategies.items():
                if symbol in strategy.symbols:
                    try:
                        # Process data through workflow engine
                        result = await self.workflow_engines[strategy_id].process_market_data_async(market_data)
                        processing_results[strategy_id] = result
                        
                        # Update system metrics
                        self.system_metrics['total_predictions'] += 1
                        
                    except Exception as e:
                        logger.error(f"Error processing data in strategy {strategy_id}: {e}")
                        processing_results[strategy_id] = {'error': str(e)}
            
            return processing_results
            
        except Exception as e:
            logger.error(f"Error processing market data: {e}")
            return {'error': str(e)}
    
    async def get_strategy_predictions(self, strategy_id: str, symbol: str) -> Optional[MLPrediction]:
        """Get latest prediction from a specific strategy"""
        try:
            if strategy_id not in self.strategies:
                logger.warning(f"Strategy not found: {strategy_id}")
                return None
            
            workflow_engine = self.workflow_engines[strategy_id]
            return await workflow_engine.get_latest_prediction(symbol)
            
        except Exception as e:
            logger.error(f"Error getting prediction from strategy {strategy_id}: {e}")
            return None
    
    async def get_aggregated_predictions(self, symbol: str) -> Dict[str, Any]:
        """Get aggregated predictions from all strategies for a symbol"""
        try:
            predictions = {}
            
            for strategy_id, workflow_engine in self.workflow_engines.items():
                if symbol in self.strategies[strategy_id].symbols:
                    prediction = await workflow_engine.get_latest_prediction(symbol)
                    if prediction:
                        predictions[strategy_id] = {
                            'prediction_value': prediction.prediction_value,
                            'confidence': prediction.confidence,
                            'timestamp': prediction.timestamp,
                            'prediction_type': prediction.prediction_type.value
                        }
            
            # Calculate consensus if multiple predictions
            if len(predictions) > 1:
                consensus = self._calculate_prediction_consensus(predictions)
                predictions['consensus'] = consensus
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error getting aggregated predictions for {symbol}: {e}")
            return {}
    
    def _calculate_prediction_consensus(self, predictions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate consensus prediction from multiple strategies"""
        try:
            values = []
            confidences = []
            weights = []
            
            for strategy_id, pred_data in predictions.items():
                if strategy_id == 'consensus':
                    continue
                
                # Weight predictions by confidence and strategy performance
                strategy_perf = self.strategy_performance.get(strategy_id, {})
                accuracy = strategy_perf.get('correct_predictions', 0) / max(strategy_perf.get('total_predictions', 1), 1)
                
                weight = pred_data['confidence'] * accuracy
                
                values.append(pred_data['prediction_value'])
                confidences.append(pred_data['confidence'])
                weights.append(weight)
            
            if not values:
                return {'prediction_value': 0, 'confidence': 0.0, 'method': 'no_predictions'}
            
            # Weighted average
            weights = np.array(weights)
            weights = weights / np.sum(weights) if np.sum(weights) > 0 else np.ones(len(weights)) / len(weights)
            
            consensus_value = np.average(values, weights=weights)
            consensus_confidence = np.average(confidences, weights=weights)
            
            return {
                'prediction_value': consensus_value,
                'confidence': consensus_confidence,
                'num_strategies': len(values),
                'method': 'weighted_average'
            }
            
        except Exception as e:
            logger.error(f"Error calculating consensus: {e}")
            return {'prediction_value': 0, 'confidence': 0.0, 'method': 'error'}
    
    def get_strategy_orders(self, strategy_id: str) -> List[Dict[str, Any]]:
        """Get pending orders from a specific strategy"""
        try:
            if strategy_id not in self.workflow_engines:
                return []
            
            return self.workflow_engines[strategy_id].get_pending_orders()
            
        except Exception as e:
            logger.error(f"Error getting orders from strategy {strategy_id}: {e}")
            return []
    
    def get_all_strategy_orders(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get pending orders from all strategies"""
        try:
            all_orders = {}
            
            for strategy_id in self.strategies:
                orders = self.get_strategy_orders(strategy_id)
                if orders:
                    all_orders[strategy_id] = orders
            
            return all_orders
            
        except Exception as e:
            logger.error(f"Error getting all strategy orders: {e}")
            return {}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            status = {
                'system_metrics': self.system_metrics.copy(),
                'active_strategies': len(self.strategies),
                'strategy_status': {},
                'memory_usage_mb': self._get_memory_usage(),
                'uptime_seconds': (datetime.now() - self._start_time).total_seconds() if hasattr(self, '_start_time') else 0,
                'is_running': self.is_running
            }
            
            # Get individual strategy status
            for strategy_id, strategy in self.strategies.items():
                if hasattr(strategy, 'get_strategy_status'):
                    status['strategy_status'][strategy_id] = strategy.get_strategy_status()
                else:
                    status['strategy_status'][strategy_id] = {
                        'strategy_id': strategy_id,
                        'symbols': strategy.symbols,
                        'model_type': strategy.model_type.value,
                        'prediction_type': strategy.prediction_type.value
                    }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {'error': str(e)}
    
    def get_performance_report(self, 
                             strategy_id: Optional[str] = None,
                             time_range: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        try:
            if strategy_id and strategy_id in self.strategy_performance:
                # Single strategy report
                return self._generate_strategy_performance_report(strategy_id, time_range)
            else:
                # System-wide report
                return self._generate_system_performance_report(time_range)
                
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return {'error': str(e)}
    
    def _generate_strategy_performance_report(self, strategy_id: str, time_range: Optional[str]) -> Dict[str, Any]:
        """Generate performance report for a specific strategy"""
        
        perf_data = self.strategy_performance[strategy_id]
        
        # Get recent predictions for analysis
        recent_predictions = [
            pred for pred in self.prediction_history[strategy_id]
            if self._is_in_time_range(pred.get('timestamp', 0), time_range)
        ]
        
        # Calculate metrics
        total_predictions = len(recent_predictions)
        accuracy = perf_data.get('correct_predictions', 0) / max(perf_data.get('total_predictions', 1), 1)
        
        return {
            'strategy_id': strategy_id,
            'time_range': time_range or 'all_time',
            'total_predictions': total_predictions,
            'accuracy': accuracy,
            'total_return': perf_data.get('total_return', 0.0),
            'sharpe_ratio': perf_data.get('sharpe_ratio', 0.0),
            'max_drawdown': perf_data.get('max_drawdown', 0.0),
            'last_updated': perf_data.get('last_updated'),
            'symbols': self.strategies[strategy_id].symbols if strategy_id in self.strategies else []
        }
    
    def _generate_system_performance_report(self, time_range: Optional[str]) -> Dict[str, Any]:
        """Generate system-wide performance report"""
        
        # Aggregate metrics across all strategies
        total_predictions = sum(data.get('total_predictions', 0) for data in self.strategy_performance.values())
        total_correct = sum(data.get('correct_predictions', 0) for data in self.strategy_performance.values())
        total_return = sum(data.get('total_return', 0.0) for data in self.strategy_performance.values())
        
        # Calculate weighted averages
        strategies_count = len(self.strategy_performance)
        avg_accuracy = total_correct / max(total_predictions, 1)
        avg_sharpe = np.mean([data.get('sharpe_ratio', 0.0) for data in self.strategy_performance.values()])
        worst_drawdown = min([data.get('max_drawdown', 0.0) for data in self.strategy_performance.values()], default=0.0)
        
        return {
            'system_summary': {
                'total_strategies': strategies_count,
                'total_predictions': total_predictions,
                'system_accuracy': avg_accuracy,
                'total_return': total_return,
                'average_sharpe_ratio': avg_sharpe,
                'worst_max_drawdown': worst_drawdown
            },
            'strategy_breakdown': {
                strategy_id: self._generate_strategy_performance_report(strategy_id, time_range)
                for strategy_id in self.strategy_performance
            },
            'time_range': time_range or 'all_time',
            'report_generated': datetime.now().isoformat()
        }
    
    def _is_in_time_range(self, timestamp: int, time_range: Optional[str]) -> bool:
        """Check if timestamp is within specified time range"""
        if not time_range:
            return True
        
        try:
            now = datetime.now()
            if time_range == '1h':
                cutoff = now - timedelta(hours=1)
            elif time_range == '1d':
                cutoff = now - timedelta(days=1)
            elif time_range == '1w':
                cutoff = now - timedelta(weeks=1)
            elif time_range == '1m':
                cutoff = now - timedelta(days=30)
            else:
                return True
            
            # Convert timestamp to datetime (assuming nanoseconds)
            dt = datetime.fromtimestamp(timestamp / 1e9)
            return dt >= cutoff
            
        except Exception:
            return True
    
    async def _monitor_system_health(self) -> None:
        """Monitor overall system health"""
        while self.is_running:
            try:
                # Update system metrics
                self.system_metrics['active_strategies'] = len(self.strategies)
                self.system_metrics['memory_usage_mb'] = self._get_memory_usage()
                
                # Check for unhealthy strategies
                for strategy_id, strategy in self.strategies.items():
                    # Example health checks
                    perf = self.strategy_performance[strategy_id]
                    accuracy = perf.get('correct_predictions', 0) / max(perf.get('total_predictions', 1), 1)
                    
                    if accuracy < 0.3 and perf.get('total_predictions', 0) > 100:
                        logger.warning(f"Strategy {strategy_id} has low accuracy: {accuracy:.3f}")
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in system health monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _monitor_strategy_performance(self) -> None:
        """Monitor individual strategy performance"""
        while self.is_running:
            try:
                for strategy_id, workflow_engine in self.workflow_engines.items():
                    # Get recent performance metrics
                    performance = await workflow_engine.get_performance_metrics()
                    
                    # Update strategy performance
                    if performance:
                        self.strategy_performance[strategy_id].update(performance)
                        self.strategy_performance[strategy_id]['last_updated'] = datetime.now().isoformat()
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in strategy performance monitoring: {e}")
                await asyncio.sleep(300)
    
    async def _manage_model_lifecycle(self) -> None:
        """Manage model lifecycle (retraining, versioning, etc.)"""
        while self.is_running:
            try:
                for strategy_id, workflow_engine in self.workflow_engines.items():
                    # Check if models need retraining
                    should_retrain = await workflow_engine.should_retrain_models()
                    
                    if should_retrain:
                        logger.info(f"Triggering model retraining for strategy {strategy_id}")
                        await workflow_engine.retrain_models()
                        
                        # Save updated models
                        self._save_strategy_models(strategy_id)
                
                await asyncio.sleep(3600)  # Check every hour
                
            except Exception as e:
                logger.error(f"Error in model lifecycle management: {e}")
                await asyncio.sleep(3600)
    
    async def _aggregate_predictions(self) -> None:
        """Aggregate and analyze predictions across strategies"""
        while self.is_running:
            try:
                # Collect recent predictions
                for strategy_id, workflow_engine in self.workflow_engines.items():
                    recent_predictions = await workflow_engine.get_recent_predictions()
                    
                    for prediction in recent_predictions:
                        self.prediction_history[strategy_id].append({
                            'timestamp': prediction.timestamp,
                            'symbol': prediction.symbol,
                            'prediction_value': prediction.prediction_value,
                            'confidence': prediction.confidence
                        })
                
                await asyncio.sleep(60)  # Aggregate every minute
                
            except Exception as e:
                logger.error(f"Error in prediction aggregation: {e}")
                await asyncio.sleep(60)
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    def _save_strategy_state(self, strategy_id: str) -> None:
        """Save strategy state to disk"""
        try:
            strategy = self.strategies[strategy_id]
            state_file = self.model_storage_path / f"{strategy_id}_state.pkl"
            
            with open(state_file, 'wb') as f:
                pickle.dump({
                    'strategy_config': strategy.config,
                    'performance_data': self.strategy_performance[strategy_id],
                    'prediction_history': list(self.prediction_history[strategy_id])
                }, f)
            
            logger.info(f"Saved state for strategy {strategy_id}")
            
        except Exception as e:
            logger.error(f"Error saving strategy state for {strategy_id}: {e}")
    
    def _save_strategy_models(self, strategy_id: str) -> None:
        """Save strategy models to disk"""
        try:
            strategy = self.strategies[strategy_id]
            
            if hasattr(strategy, 'models'):
                models_file = self.model_storage_path / f"{strategy_id}_models.pkl"
                
                with open(models_file, 'wb') as f:
                    pickle.dump(strategy.models, f)
                
                logger.info(f"Saved models for strategy {strategy_id}")
            
        except Exception as e:
            logger.error(f"Error saving models for {strategy_id}: {e}")
    
    async def _load_persisted_strategies(self) -> None:
        """Load previously saved strategies"""
        try:
            state_files = list(self.model_storage_path.glob("*_state.pkl"))
            
            for state_file in state_files:
                strategy_id = state_file.stem.replace('_state', '')
                
                try:
                    with open(state_file, 'rb') as f:
                        state_data = pickle.load(f)
                    
                    # Restore strategy performance
                    self.strategy_performance[strategy_id] = state_data.get('performance_data', {})
                    
                    # Restore prediction history
                    predictions = state_data.get('prediction_history', [])
                    for pred in predictions:
                        self.prediction_history[strategy_id].append(pred)
                    
                    logger.info(f"Loaded persisted state for strategy {strategy_id}")
                    
                except Exception as e:
                    logger.error(f"Error loading state for {strategy_id}: {e}")
            
        except Exception as e:
            logger.error(f"Error loading persisted strategies: {e}")
    
    async def _persist_strategies(self) -> None:
        """Persist all current strategies"""
        try:
            for strategy_id in self.strategies:
                self._save_strategy_state(strategy_id)
                self._save_strategy_models(strategy_id)
            
            logger.info("Persisted all strategies")
            
        except Exception as e:
            logger.error(f"Error persisting strategies: {e}")

# Example usage and setup functions
def create_market_direction_strategy_manager(symbols: List[str], config: Dict[str, Any]) -> MLStrategyManager:
    """Create ML strategy manager with market direction strategy"""
    
    manager = MLStrategyManager(config)
    
    # Register market direction strategy
    strategy_config = {
        'max_position': 10000,
        'max_daily_loss': 50000,
        'retrain_hours': 6,
        'max_training_samples': 10000,
        'risk_aversion': 0.1
    }
    
    manager.register_strategy(
        strategy_class=MarketDirectionStrategy,
        strategy_id="market_direction_v1",
        symbols=symbols,
        config=strategy_config
    )
    
    return manager

async def run_ml_strategies_example():
    """Example of running ML strategies"""
    
    # Configuration
    config = {
        'max_workers': 4,
        'model_storage_path': 'models/ml_strategies'
    }
    
    symbols = ['AAPL', 'GOOGL', 'TSLA']
    
    # Create manager
    manager = create_market_direction_strategy_manager(symbols, config)
    
    try:
        # Start manager
        await manager.start()
        
        # Simulate market data processing
        for i in range(100):
            for symbol in symbols:
                # Create mock market data
                market_data = MarketMicrostructureData(
                    timestamp=int(datetime.now().timestamp() * 1e9),
                    symbol=symbol,
                    best_bid=100.0 + np.random.normal(0, 0.1),
                    best_ask=100.1 + np.random.normal(0, 0.1),
                    bid_size=1000,
                    ask_size=1000,
                    last_price=100.05 + np.random.normal(0, 0.1),
                    last_size=100,
                    volume=10000,
                    vwap=100.03,
                    high=100.5,
                    low=99.5,
                    open_price=100.0
                )
                
                # Process through strategies
                results = await manager.process_market_data(market_data)
                
                if i % 20 == 0:  # Print status every 20 iterations
                    print(f"Processed {i} samples. Results: {len(results)} strategies")
            
            await asyncio.sleep(0.1)  # Simulate real-time delays
        
        # Get final status
        status = manager.get_system_status()
        print(f"Final system status: {status}")
        
        # Get performance report
        report = manager.get_performance_report()
        print(f"Performance report: {report}")
        
    finally:
        await manager.stop()

if __name__ == "__main__":
    # Run example
    asyncio.run(run_ml_strategies_example())
