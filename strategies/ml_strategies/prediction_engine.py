"""
Prediction Engine
===============

Implements Step 6 of the ML modeling workflow from Modeling.md:
Real-time Prediction/Classification (Market Direction, Spread Prediction, Order-Flow)

Generates immediate predictions (microsecond level) using optimized models for:
- Market Direction prediction
- Spread Prediction
- Order-Flow prediction
- Liquidity consumption prediction
- Price reversal detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime
import joblib
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
import time

# ML model imports
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
import xgboost as xgb
import lightgbm as lgb

from .base_ml_strategy import MLFeatures, MLPrediction, PredictionType, ModelType

logger = logging.getLogger(__name__)

class PredictionConfidence(Enum):
    """Prediction confidence levels"""
    VERY_LOW = "very_low"      # < 0.5
    LOW = "low"                # 0.5 - 0.6
    MEDIUM = "medium"          # 0.6 - 0.7
    HIGH = "high"              # 0.7 - 0.8
    VERY_HIGH = "very_high"    # > 0.8

@dataclass
class ModelMetadata:
    """Metadata for ML models"""
    model_type: str
    training_timestamp: datetime
    feature_names: List[str]
    model_parameters: Dict[str, Any]
    performance_metrics: Dict[str, float]
    prediction_latency_ns: float = 0.0

@dataclass
class PredictionConfig:
    """Configuration for prediction engine"""
    # Model selection
    default_classification_model: str = "xgboost"
    default_regression_model: str = "xgboost"
    
    # Ensemble settings
    use_ensemble: bool = True
    ensemble_weights: Dict[str, float] = None
    
    # Performance settings
    max_prediction_latency_ns: int = 50_000  # 50 microseconds
    batch_prediction_size: int = 1
    
    # Confidence thresholds
    min_confidence_threshold: float = 0.5
    high_confidence_threshold: float = 0.7
    
    # Feature selection
    max_features: int = 50
    feature_selection_method: str = "importance"
    
    def __post_init__(self):
        if self.ensemble_weights is None:
            self.ensemble_weights = {
                "xgboost": 0.4,
                "lightgbm": 0.3,
                "random_forest": 0.2,
                "neural_network": 0.1
            }

class PredictionEngine:
    """
    High-performance prediction engine for HFT ML strategies
    
    Generates real-time predictions using optimized ML models with
    sub-microsecond latency targeting.
    """
    
    def __init__(self, symbols: List[str], config: PredictionConfig):
        self.symbols = symbols
        self.config = config
        
        # Model storage
        self.models: Dict[str, Dict[PredictionType, Any]] = {
            symbol: {} for symbol in symbols
        }
        self.model_metadata: Dict[str, Dict[PredictionType, ModelMetadata]] = {
            symbol: {} for symbol in symbols
        }
        
        # Feature selection
        self.selected_features: Dict[str, Dict[PredictionType, List[str]]] = {
            symbol: {} for symbol in symbols
        }
        
        # Performance tracking
        self.prediction_history: Dict[str, List[MLPrediction]] = {
            symbol: [] for symbol in symbols
        }
        self.latency_stats: Dict[str, List[float]] = {
            symbol: [] for symbol in symbols
        }
        
        # Threading for async predictions
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.prediction_lock = threading.RLock()
        
        # Model registry
        self.model_registry = {
            # Classification models
            "xgboost_classifier": xgb.XGBClassifier,
            "lightgbm_classifier": lgb.LGBMClassifier,
            "random_forest_classifier": RandomForestClassifier,
            "gradient_boosting_classifier": GradientBoostingClassifier,
            "logistic_regression": LogisticRegression,
            "neural_network_classifier": MLPClassifier,
            
            # Regression models
            "xgboost_regressor": xgb.XGBRegressor,
            "lightgbm_regressor": lgb.LGBMRegressor,
            "random_forest_regressor": RandomForestRegressor,
            "linear_regression": LinearRegression,
            "neural_network_regressor": MLPRegressor
        }
        
        logger.info(f"Initialized prediction engine for {len(symbols)} symbols")
    
    def register_model(self, 
                      symbol: str,
                      prediction_type: PredictionType,
                      model: Any,
                      metadata: ModelMetadata) -> bool:
        """Register a trained model for predictions"""
        try:
            with self.prediction_lock:
                self.models[symbol][prediction_type] = model
                self.model_metadata[symbol][prediction_type] = metadata
                
                logger.info(f"Registered {metadata.model_type} model for {symbol} - {prediction_type.value}")
                return True
                
        except Exception as e:
            logger.error(f"Error registering model: {e}")
            return False
    
    async def predict(self, 
                     symbol: str,
                     prediction_type: PredictionType,
                     features: MLFeatures) -> Optional[MLPrediction]:
        """
        Generate prediction for given features
        
        Args:
            symbol: Trading symbol
            prediction_type: Type of prediction to generate
            features: Engineered features
            
        Returns:
            MLPrediction: Prediction with confidence score
        """
        start_time = time.perf_counter_ns()
        
        try:
            # Check if model is available
            if (symbol not in self.models or 
                prediction_type not in self.models[symbol]):
                logger.warning(f"No model available for {symbol} - {prediction_type.value}")
                return None
            
            # Prepare feature array
            feature_array = self._prepare_feature_array(symbol, prediction_type, features)
            if feature_array is None:
                return None
            
            # Generate prediction using ensemble or single model
            if self.config.use_ensemble:
                prediction_value, confidence = await self._ensemble_predict(
                    symbol, prediction_type, feature_array
                )
            else:
                prediction_value, confidence = await self._single_model_predict(
                    symbol, prediction_type, feature_array
                )
            
            # Calculate latency
            latency_ns = time.perf_counter_ns() - start_time
            self._update_latency_stats(symbol, latency_ns)
            
            # Check latency constraint
            if latency_ns > self.config.max_prediction_latency_ns:
                logger.warning(f"Prediction latency {latency_ns}ns exceeds limit {self.config.max_prediction_latency_ns}ns")
            
            # Create prediction object
            prediction = MLPrediction(
                timestamp=int(datetime.now().timestamp() * 1e9),
                symbol=symbol,
                prediction_type=prediction_type,
                prediction_value=prediction_value,
                confidence=confidence,
                feature_importance=self._get_feature_importance(symbol, prediction_type),
                model_metadata={
                    'latency_ns': latency_ns,
                    'model_type': self.model_metadata[symbol][prediction_type].model_type,
                    'confidence_level': self._get_confidence_level(confidence).value
                }
            )
            
            # Store prediction history
            self._store_prediction(symbol, prediction)
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error generating prediction: {e}")
            return None
    
    async def _single_model_predict(self,
                                  symbol: str,
                                  prediction_type: PredictionType,
                                  feature_array: np.ndarray) -> Tuple[Union[float, int], float]:
        """Generate prediction using single model"""
        model = self.models[symbol][prediction_type]
        
        # Run prediction in thread pool to avoid blocking
        prediction_result = await asyncio.get_event_loop().run_in_executor(
            self.executor,
            self._execute_model_prediction,
            model,
            feature_array,
            prediction_type
        )
        
        return prediction_result
    
    async def _ensemble_predict(self,
                              symbol: str,
                              prediction_type: PredictionType,
                              feature_array: np.ndarray) -> Tuple[Union[float, int], float]:
        """Generate prediction using ensemble of models"""
        # For now, use single model - ensemble would require multiple trained models
        return await self._single_model_predict(symbol, prediction_type, feature_array)
    
    def _execute_model_prediction(self,
                                model: Any,
                                feature_array: np.ndarray,
                                prediction_type: PredictionType) -> Tuple[Union[float, int], float]:
        """Execute model prediction synchronously"""
        try:
            if prediction_type in [PredictionType.MARKET_DIRECTION]:
                # Classification
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(feature_array)[0]
                    prediction_value = np.argmax(probabilities) - 1  # Convert to -1, 0, 1
                    confidence = np.max(probabilities)
                else:
                    prediction_value = model.predict(feature_array)[0]
                    confidence = 0.6  # Default confidence for models without probability
                
            else:
                # Regression
                prediction_value = model.predict(feature_array)[0]
                
                # Estimate confidence for regression (using prediction uncertainty)
                if hasattr(model, 'predict_proba'):
                    # For models that can provide uncertainty estimates
                    confidence = 0.7  # Placeholder
                else:
                    # Simple confidence based on model type
                    confidence = 0.6
            
            return prediction_value, confidence
            
        except Exception as e:
            logger.error(f"Error executing model prediction: {e}")
            return 0.0, 0.0
    
    def _prepare_feature_array(self,
                             symbol: str,
                             prediction_type: PredictionType,
                             features: MLFeatures) -> Optional[np.ndarray]:
        """Prepare feature array for model input"""
        try:
            # Get selected features for this symbol and prediction type
            if (symbol in self.selected_features and 
                prediction_type in self.selected_features[symbol]):
                feature_names = self.selected_features[symbol][prediction_type]
            else:
                # Use all available features
                feature_names = self._get_all_feature_names(features)
                if len(feature_names) > self.config.max_features:
                    feature_names = feature_names[:self.config.max_features]
            
            # Extract feature values
            feature_dict = self._features_to_dict(features)
            feature_values = [feature_dict.get(name, 0.0) for name in feature_names]
            
            return np.array(feature_values).reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Error preparing feature array: {e}")
            return None
    
    def _features_to_dict(self, features: MLFeatures) -> Dict[str, float]:
        """Convert MLFeatures to dictionary"""
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
            'day_of_week': float(features.day_of_week),
            # Market session as numeric
            'is_regular_session': 1.0 if features.market_session == 'regular' else 0.0,
            'is_pre_market': 1.0 if features.market_session == 'pre_market' else 0.0,
            'is_after_market': 1.0 if features.market_session == 'after_market' else 0.0,
        }
        
        # Add custom features
        feature_dict.update(features.custom_features)
        
        return feature_dict
    
    def _get_all_feature_names(self, features: MLFeatures) -> List[str]:
        """Get all available feature names"""
        feature_dict = self._features_to_dict(features)
        return sorted(feature_dict.keys())
    
    def _get_feature_importance(self, 
                              symbol: str,
                              prediction_type: PredictionType) -> Dict[str, float]:
        """Get feature importance from trained model"""
        try:
            if (symbol not in self.models or 
                prediction_type not in self.models[symbol]):
                return {}
            
            model = self.models[symbol][prediction_type]
            
            # Get feature importance based on model type
            if hasattr(model, 'feature_importances_'):
                # Tree-based models (XGBoost, LightGBM, RandomForest)
                importances = model.feature_importances_
                if symbol in self.selected_features and prediction_type in self.selected_features[symbol]:
                    feature_names = self.selected_features[symbol][prediction_type]
                    return dict(zip(feature_names, importances))
            
            elif hasattr(model, 'coef_'):
                # Linear models
                coefficients = np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
                if symbol in self.selected_features and prediction_type in self.selected_features[symbol]:
                    feature_names = self.selected_features[symbol][prediction_type]
                    return dict(zip(feature_names, coefficients))
            
            return {}
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return {}
    
    def _get_confidence_level(self, confidence: float) -> PredictionConfidence:
        """Convert numeric confidence to confidence level"""
        if confidence < 0.5:
            return PredictionConfidence.VERY_LOW
        elif confidence < 0.6:
            return PredictionConfidence.LOW
        elif confidence < 0.7:
            return PredictionConfidence.MEDIUM
        elif confidence < 0.8:
            return PredictionConfidence.HIGH
        else:
            return PredictionConfidence.VERY_HIGH
    
    def _store_prediction(self, symbol: str, prediction: MLPrediction) -> None:
        """Store prediction in history"""
        with self.prediction_lock:
            self.prediction_history[symbol].append(prediction)
            
            # Keep only recent predictions to avoid memory issues
            if len(self.prediction_history[symbol]) > 10000:
                self.prediction_history[symbol] = self.prediction_history[symbol][-5000:]
    
    def _update_latency_stats(self, symbol: str, latency_ns: float) -> None:
        """Update latency statistics"""
        with self.prediction_lock:
            self.latency_stats[symbol].append(latency_ns)
            
            # Keep only recent latency measurements
            if len(self.latency_stats[symbol]) > 1000:
                self.latency_stats[symbol] = self.latency_stats[symbol][-500:]
    
    # Model training methods
    def train_model(self,
                   symbol: str,
                   prediction_type: PredictionType,
                   training_data: pd.DataFrame,
                   target_column: str,
                   model_type: str = None) -> bool:
        """Train a new model for the given symbol and prediction type"""
        try:
            if model_type is None:
                if prediction_type == PredictionType.MARKET_DIRECTION:
                    model_type = self.config.default_classification_model + "_classifier"
                else:
                    model_type = self.config.default_regression_model + "_regressor"
            
            # Prepare training data
            feature_columns = [col for col in training_data.columns if col != target_column]
            X = training_data[feature_columns]
            y = training_data[target_column]
            
            # Select features
            selected_features = self._select_features(X, y, prediction_type)
            X_selected = X[selected_features]
            
            # Initialize model
            model_class = self.model_registry.get(model_type)
            if model_class is None:
                logger.error(f"Unknown model type: {model_type}")
                return False
            
            # Configure model parameters based on type and prediction type
            model_params = self._get_model_parameters(model_type, prediction_type)
            model = model_class(**model_params)
            
            # Train model
            start_time = time.perf_counter()
            model.fit(X_selected, y)
            training_time = time.perf_counter() - start_time
            
            # Calculate performance metrics
            performance_metrics = self._calculate_model_performance(model, X_selected, y, prediction_type)
            
            # Create metadata
            metadata = ModelMetadata(
                model_type=model_type,
                training_timestamp=datetime.now(),
                feature_names=selected_features,
                model_parameters=model_params,
                performance_metrics=performance_metrics
            )
            
            # Register model
            self.selected_features[symbol][prediction_type] = selected_features
            success = self.register_model(symbol, prediction_type, model, metadata)
            
            if success:
                logger.info(f"Successfully trained {model_type} for {symbol} - {prediction_type.value} "
                          f"in {training_time:.2f}s. Performance: {performance_metrics}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return False
    
    def _select_features(self, 
                        X: pd.DataFrame,
                        y: pd.Series,
                        prediction_type: PredictionType) -> List[str]:
        """Select most important features for the model"""
        try:
            if len(X.columns) <= self.config.max_features:
                return list(X.columns)
            
            if self.config.feature_selection_method == "importance":
                # Use a simple model to get feature importance
                if prediction_type == PredictionType.MARKET_DIRECTION:
                    selector_model = RandomForestClassifier(n_estimators=50, random_state=42)
                else:
                    selector_model = RandomForestRegressor(n_estimators=50, random_state=42)
                
                selector_model.fit(X, y)
                feature_importance = selector_model.feature_importances_
                
                # Select top features
                feature_indices = np.argsort(feature_importance)[-self.config.max_features:]
                return [X.columns[i] for i in feature_indices]
            
            else:
                # Fallback to first N features
                return list(X.columns[:self.config.max_features])
                
        except Exception as e:
            logger.error(f"Error selecting features: {e}")
            return list(X.columns[:self.config.max_features])
    
    def _get_model_parameters(self, model_type: str, prediction_type: PredictionType) -> Dict[str, Any]:
        """Get optimized parameters for each model type"""
        base_params = {}
        
        if "xgboost" in model_type:
            base_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42,
                'n_jobs': -1
            }
            if prediction_type == PredictionType.MARKET_DIRECTION:
                base_params['objective'] = 'multi:softprob'
                base_params['num_class'] = 3  # -1, 0, 1
        
        elif "lightgbm" in model_type:
            base_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1
            }
            if prediction_type == PredictionType.MARKET_DIRECTION:
                base_params['objective'] = 'multiclass'
                base_params['num_class'] = 3
        
        elif "random_forest" in model_type:
            base_params = {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42,
                'n_jobs': -1
            }
        
        elif "neural_network" in model_type:
            base_params = {
                'hidden_layer_sizes': (100, 50),
                'max_iter': 200,
                'random_state': 42
            }
        
        elif "logistic_regression" in model_type:
            base_params = {
                'random_state': 42,
                'max_iter': 1000
            }
            if prediction_type == PredictionType.MARKET_DIRECTION:
                base_params['multi_class'] = 'ovr'
        
        return base_params
    
    def _calculate_model_performance(self,
                                   model: Any,
                                   X: pd.DataFrame,
                                   y: pd.Series,
                                   prediction_type: PredictionType) -> Dict[str, float]:
        """Calculate model performance metrics"""
        try:
            if prediction_type == PredictionType.MARKET_DIRECTION:
                # Classification metrics
                y_pred = model.predict(X)
                accuracy = np.mean(y_pred == y)
                
                # Precision, recall for each class
                from sklearn.metrics import precision_recall_fscore_support
                precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='weighted')
                
                return {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                }
            else:
                # Regression metrics
                y_pred = model.predict(X)
                mse = np.mean((y_pred - y) ** 2)
                mae = np.mean(np.abs(y_pred - y))
                
                # R-squared
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                return {
                    'mse': mse,
                    'mae': mae,
                    'r2_score': r2
                }
                
        except Exception as e:
            logger.error(f"Error calculating model performance: {e}")
            return {}
    
    # Utility methods
    def get_prediction_statistics(self, symbol: str) -> Dict[str, Any]:
        """Get prediction statistics for a symbol"""
        with self.prediction_lock:
            if symbol not in self.prediction_history:
                return {}
            
            predictions = self.prediction_history[symbol]
            latencies = self.latency_stats.get(symbol, [])
            
            if not predictions:
                return {}
            
            # Calculate statistics
            confidences = [p.confidence for p in predictions]
            
            stats = {
                'total_predictions': len(predictions),
                'avg_confidence': np.mean(confidences),
                'min_confidence': np.min(confidences),
                'max_confidence': np.max(confidences),
                'high_confidence_predictions': sum(1 for c in confidences if c >= self.config.high_confidence_threshold),
            }
            
            if latencies:
                stats.update({
                    'avg_latency_ns': np.mean(latencies),
                    'max_latency_ns': np.max(latencies),
                    'p95_latency_ns': np.percentile(latencies, 95),
                    'p99_latency_ns': np.percentile(latencies, 99)
                })
            
            return stats
    
    def get_model_info(self, symbol: str, prediction_type: PredictionType) -> Optional[Dict[str, Any]]:
        """Get information about a trained model"""
        if (symbol not in self.model_metadata or 
            prediction_type not in self.model_metadata[symbol]):
            return None
        
        metadata = self.model_metadata[symbol][prediction_type]
        
        return {
            'model_type': metadata.model_type,
            'training_timestamp': metadata.training_timestamp.isoformat(),
            'feature_count': len(metadata.feature_names),
            'feature_names': metadata.feature_names,
            'model_parameters': metadata.model_parameters,
            'performance_metrics': metadata.performance_metrics
        }
    
    def save_models(self, filepath: str) -> bool:
        """Save all trained models to disk"""
        try:
            model_data = {
                'models': self.models,
                'metadata': self.model_metadata,
                'selected_features': self.selected_features
            }
            
            joblib.dump(model_data, filepath)
            logger.info(f"Saved models to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            return False
    
    def load_models(self, filepath: str) -> bool:
        """Load trained models from disk"""
        try:
            model_data = joblib.load(filepath)
            
            self.models = model_data['models']
            self.model_metadata = model_data['metadata']
            self.selected_features = model_data['selected_features']
            
            logger.info(f"Loaded models from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
