"""
Market Direction Prediction Strategy
==================================

Concrete implementation of ML strategy for market direction prediction
following the complete workflow from Modeling.md.

This strategy predicts short-term market direction (-1, 0, 1) using:
- Enhanced microstructure features
- Statistical pattern identification
- ML model training and calibration
- Real-time prediction generation
- Optimal order placement
- Continuous learning and adaptation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime, timedelta
from collections import deque

# ML imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

from ..base_ml_strategy import (
    BaseMLStrategy, MarketMicrostructureData, MLFeatures, MLPrediction,
    ModelType, PredictionType, RiskConstraints, ModelCalibrationResult
)
from ..feature_engineering import FeatureEngineer, FeatureConfig
from ..optimal_execution import OptimalExecutionManager, ExecutionParams

logger = logging.getLogger(__name__)

class MarketDirectionStrategy(BaseMLStrategy):
    """
    ML strategy for predicting market direction using the complete workflow
    
    Implements all 8 steps of the ML modeling logic from Modeling.md:
    1. Real-Time Market Data Processing
    2. Data Cleaning & Feature Engineering
    3. Statistical Patterns & Relationships Identification
    4. Model Formulation (Statistical / ML-Based)
    5. Model Calibration (Backtest & Optimize)
    6. Real-time Prediction/Classification
    7. Optimal Order Placement Logic
    8. Continuous Learning & Model Adaptation
    """
    
    def __init__(self, symbols: List[str], config: Dict[str, Any]):
        # Initialize base strategy
        risk_constraints = RiskConstraints(
            max_position=config.get('max_position', 10000),
            max_daily_loss=config.get('max_daily_loss', 50000),
            max_leverage=config.get('max_leverage', 2.0),
            var_limit=config.get('var_limit', 25000),
            max_drawdown=config.get('max_drawdown', 0.15),
            concentration_limit=config.get('concentration_limit', 0.3),
            stop_loss_pct=config.get('stop_loss_pct', 0.05)
        )
        
        super().__init__(
            strategy_id="market_direction_ml_v1",
            symbols=symbols,
            model_type=ModelType.CLASSIFICATION,
            prediction_type=PredictionType.MARKET_DIRECTION,
            risk_constraints=risk_constraints,
            config=config
        )
        
        # Initialize components
        self.feature_engineer = FeatureEngineer(symbols, FeatureConfig())
        self.execution_manager = OptimalExecutionManager()
        
        # Data storage for training
        self.training_data: Dict[str, List[Dict]] = {symbol: [] for symbol in symbols}
        self.max_training_samples = config.get('max_training_samples', 10000)
        
        # Model storage
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        
        # Performance tracking
        self.prediction_accuracy: Dict[str, deque] = {
            symbol: deque(maxlen=1000) for symbol in symbols
        }
        self.recent_returns: Dict[str, deque] = {
            symbol: deque(maxlen=100) for symbol in symbols
        }
        
        # Strategy state
        self.last_predictions: Dict[str, MLPrediction] = {}
        self.last_retraining: Dict[str, datetime] = {}
        self.retraining_interval = timedelta(hours=config.get('retrain_hours', 6))
        
        logger.info(f"Initialized MarketDirectionStrategy for {len(symbols)} symbols")
    
    # Step 1: Real-Time Market Data Processing
    def process_market_data(self, market_data: MarketMicrostructureData) -> bool:
        """Process incoming real-time market data"""
        try:
            symbol = market_data.symbol
            
            # Basic validation
            if (market_data.best_bid <= 0 or market_data.best_ask <= 0 or
                market_data.best_bid >= market_data.best_ask):
                logger.warning(f"Invalid market data for {symbol}")
                return False
            
            # Store for training data if we have sufficient history
            if len(self.recent_returns[symbol]) > 0:
                # Calculate label (future return direction)
                current_price = market_data.mid_price
                
                # Look back to see if we have data from ~30 seconds ago
                if len(self.training_data[symbol]) > 0:
                    recent_data = self.training_data[symbol][-1]
                    time_diff = market_data.timestamp - recent_data['timestamp']
                    
                    # If enough time has passed, calculate return and add label
                    if time_diff > 30_000_000_000:  # 30 seconds in nanoseconds
                        previous_price = recent_data['mid_price']
                        return_pct = (current_price - previous_price) / previous_price
                        
                        # Classify return direction
                        if return_pct > 0.0005:  # 0.05% threshold
                            label = 1  # Up
                        elif return_pct < -0.0005:
                            label = -1  # Down
                        else:
                            label = 0  # Flat
                        
                        # Update the previous data point with label
                        recent_data['future_direction'] = label
                        self.recent_returns[symbol].append(return_pct)
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing market data: {e}")
            return False
    
    # Step 2: Data Cleaning & Feature Engineering
    def engineer_features(self, market_data: MarketMicrostructureData) -> MLFeatures:
        """Extract meaningful signals from market data"""
        return self.feature_engineer.extract_features(market_data)
    
    # Step 3: Statistical Patterns & Relationships Identification
    def identify_statistical_patterns(self, features: MLFeatures) -> Dict[str, float]:
        """Identify statistical patterns using advanced methods"""
        patterns = {}
        
        # Volatility clustering detection
        if len(self.recent_returns[features.custom_features.get('symbol', self.symbols[0])]) > 20:
            symbol = features.custom_features.get('symbol', self.symbols[0])
            returns = np.array(list(self.recent_returns[symbol]))
            
            # Calculate volatility clustering
            if len(returns) > 1:
                volatilities = pd.Series(returns).rolling(window=5).std()
                if len(volatilities.dropna()) > 1:
                    vol_autocorr = volatilities.dropna().autocorr(lag=1)
                    patterns['volatility_clustering'] = vol_autocorr if not np.isnan(vol_autocorr) else 0.0
                else:
                    patterns['volatility_clustering'] = 0.0
            
            # Mean reversion tendency
            if len(returns) >= 10:
                # Check if returns tend to reverse
                reversals = 0
                for i in range(1, len(returns)):
                    if returns[i] * returns[i-1] < 0:  # Opposite signs
                        reversals += 1
                patterns['mean_reversion_strength'] = reversals / (len(returns) - 1)
            else:
                patterns['mean_reversion_strength'] = 0.5
        
        # Momentum patterns
        momentum_score = features.price_momentum
        patterns['momentum_strength'] = abs(momentum_score)
        patterns['momentum_direction'] = np.sign(momentum_score)
        
        return patterns
    
    # Step 4: Model Formulation (Statistical / ML-Based)
    def formulate_model(self, training_data: pd.DataFrame) -> Any:
        """Develop predictive model for market direction classification"""
        try:
            # Prepare features and labels
            feature_columns = [col for col in training_data.columns 
                             if col not in ['future_direction', 'timestamp', 'symbol']]
            
            X = training_data[feature_columns]
            y = training_data['future_direction']
            
            # Handle missing values
            X = X.fillna(0)
            
            # Remove samples with no label
            valid_mask = ~y.isna()
            X = X[valid_mask]
            y = y[valid_mask]
            
            if len(X) < 100:
                logger.warning("Insufficient training data")
                return None
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train ensemble of models
            models = {}
            
            # XGBoost
            xgb_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='mlogloss'
            )
            xgb_model.fit(X_train_scaled, y_train)
            models['xgboost'] = xgb_model
            
            # Random Forest
            rf_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            rf_model.fit(X_train_scaled, y_train)
            models['random_forest'] = rf_model
            
            # Evaluate models
            best_model = None
            best_score = 0
            
            for model_name, model in models.items():
                y_pred = model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)
                logger.info(f"{model_name} accuracy: {accuracy:.3f}")
                
                if accuracy > best_score:
                    best_score = accuracy
                    best_model = model
            
            return {
                'model': best_model,
                'scaler': scaler,
                'feature_names': feature_columns,
                'accuracy': best_score
            }
            
        except Exception as e:
            logger.error(f"Error formulating model: {e}")
            return None
    
    # Step 5: Model Calibration (Backtest & Optimize)
    def calibrate_model(self, model: Any, validation_data: pd.DataFrame) -> ModelCalibrationResult:
        """Calibrate model using validation data and calculate performance metrics"""
        try:
            if model is None:
                return self._default_calibration_result()
            
            # Extract model components
            ml_model = model['model']
            scaler = model['scaler']
            feature_names = model['feature_names']
            
            # Prepare validation data
            X_val = validation_data[feature_names].fillna(0)
            y_val = validation_data['future_direction'].dropna()
            
            # Align X and y
            valid_indices = y_val.index
            X_val = X_val.loc[valid_indices]
            
            if len(X_val) < 50:
                return self._default_calibration_result()
            
            # Scale features
            X_val_scaled = scaler.transform(X_val)
            
            # Make predictions
            y_pred = ml_model.predict(X_val_scaled)
            y_pred_proba = ml_model.predict_proba(X_val_scaled) if hasattr(ml_model, 'predict_proba') else None
            
            # Calculate performance metrics
            accuracy = accuracy_score(y_val, y_pred)
            
            # Calculate precision, recall, f1 for each class
            from sklearn.metrics import precision_recall_fscore_support
            precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='weighted')
            
            # Simulate trading performance
            returns = self._simulate_trading_performance(y_val, y_pred, y_pred_proba)
            
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            max_drawdown = self._calculate_max_drawdown(returns)
            total_return = np.sum(returns)
            
            # Calculate win rate
            winning_trades = np.sum(returns > 0)
            total_trades = len(returns)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # Calculate profit factor
            gross_profit = np.sum(returns[returns > 0])
            gross_loss = abs(np.sum(returns[returns < 0]))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
            
            return ModelCalibrationResult(
                model_id=f"market_direction_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                calibration_period=(datetime.now() - timedelta(days=1), datetime.now()),
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sharpe_ratio * 1.2,  # Approximate
                max_drawdown=max_drawdown,
                total_return=total_return,
                win_rate=win_rate,
                profit_factor=profit_factor,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                feature_importance=self._get_feature_importance(ml_model, feature_names)
            )
            
        except Exception as e:
            logger.error(f"Error calibrating model: {e}")
            return self._default_calibration_result()
    
    def _simulate_trading_performance(self, 
                                    y_true: pd.Series, 
                                    y_pred: np.ndarray,
                                    y_pred_proba: Optional[np.ndarray] = None) -> np.ndarray:
        """Simulate trading performance based on predictions"""
        
        returns = []
        
        for i in range(len(y_true)):
            true_direction = y_true.iloc[i]
            pred_direction = y_pred[i]
            
            # Calculate confidence if probabilities available
            confidence = 1.0
            if y_pred_proba is not None:
                confidence = np.max(y_pred_proba[i])
            
            # Only trade if confidence is high enough
            if confidence < 0.6:
                returns.append(0.0)
                continue
            
            # Simulate return based on correct prediction
            base_return = 0.0005  # 5 basis points base return
            
            if pred_direction == true_direction and pred_direction != 0:
                # Correct directional prediction
                returns.append(base_return * confidence)
            elif pred_direction != true_direction and pred_direction != 0:
                # Wrong directional prediction
                returns.append(-base_return * confidence)
            else:
                # No trade (predicted flat)
                returns.append(0.0)
        
        return np.array(returns)
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0.0
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        if len(returns) == 0:
            return 0.0
        
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        
        return np.min(drawdown)
    
    def _get_feature_importance(self, model: Any, feature_names: List[str]) -> Dict[str, float]:
        """Get feature importance from model"""
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                return dict(zip(feature_names, importances))
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
                return dict(zip(feature_names, importances))
            else:
                return {}
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return {}
    
    def _default_calibration_result(self) -> ModelCalibrationResult:
        """Return default calibration result when calibration fails"""
        return ModelCalibrationResult(
            model_id="default_model",
            calibration_period=(datetime.now() - timedelta(days=1), datetime.now()),
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            total_return=0.0,
            win_rate=0.5,
            profit_factor=1.0,
            accuracy=0.33,  # Random guess for 3-class problem
            precision=0.33,
            recall=0.33,
            f1_score=0.33
        )
    
    # Step 6: Real-time Prediction/Classification
    def generate_prediction(self, features: MLFeatures) -> MLPrediction:
        """Generate real-time market direction prediction"""
        
        # Assume we're working with the first symbol for this example
        symbol = self.symbols[0] if self.symbols else "UNKNOWN"
        
        try:
            # Check if we have a trained model
            if symbol not in self.models or self.models[symbol] is None:
                return self._default_prediction(symbol)
            
            model_data = self.models[symbol]
            ml_model = model_data['model']
            scaler = model_data['scaler']
            feature_names = model_data['feature_names']
            
            # Prepare feature vector
            feature_dict = self._features_to_dict(features)
            feature_vector = [feature_dict.get(name, 0.0) for name in feature_names]
            feature_array = np.array(feature_vector).reshape(1, -1)
            
            # Scale features
            feature_scaled = scaler.transform(feature_array)
            
            # Make prediction
            prediction_class = ml_model.predict(feature_scaled)[0]
            
            # Get prediction probabilities for confidence
            if hasattr(ml_model, 'predict_proba'):
                probabilities = ml_model.predict_proba(feature_scaled)[0]
                confidence = np.max(probabilities)
            else:
                confidence = 0.6  # Default confidence
            
            # Get feature importance
            feature_importance = self._get_feature_importance(ml_model, feature_names)
            
            prediction = MLPrediction(
                timestamp=int(datetime.now().timestamp() * 1e9),
                symbol=symbol,
                prediction_type=self.prediction_type,
                prediction_value=int(prediction_class),
                confidence=confidence,
                feature_importance=feature_importance,
                model_metadata={
                    'model_type': 'market_direction_classifier',
                    'feature_count': len(feature_names)
                }
            )
            
            # Store prediction for later evaluation
            self.last_predictions[symbol] = prediction
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error generating prediction: {e}")
            return self._default_prediction(symbol)
    
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
        }
        
        # Add custom features
        feature_dict.update(features.custom_features)
        
        return feature_dict
    
    def _default_prediction(self, symbol: str) -> MLPrediction:
        """Return default prediction when model is not available"""
        return MLPrediction(
            timestamp=int(datetime.now().timestamp() * 1e9),
            symbol=symbol,
            prediction_type=self.prediction_type,
            prediction_value=0,  # Neutral
            confidence=0.33,     # Low confidence
            feature_importance={},
            model_metadata={'model_type': 'default'}
        )
    
    # Step 7: Optimal Order Placement Logic
    def calculate_optimal_orders(self, 
                               prediction: MLPrediction,
                               current_position: float,
                               market_data: MarketMicrostructureData) -> List[Dict[str, Any]]:
        """Calculate optimal orders using prediction and risk management"""
        
        # Convert prediction to target position
        target_position = self._calculate_target_position(prediction, current_position)
        
        # Use execution manager to get optimal orders
        execution_params = ExecutionParams(
            risk_aversion=self.config.get('risk_aversion', 0.1),
            max_participation_rate=self.config.get('max_participation_rate', 0.05),
            min_order_size=self.config.get('min_order_size', 100),
            max_order_size=self.config.get('max_order_size', 1000)
        )
        
        order_recommendations = self.execution_manager.calculate_optimal_orders(
            prediction=prediction,
            current_position=current_position,
            target_position=target_position,
            market_data=market_data,
            execution_params=execution_params,
            algorithm='auto'
        )
        
        # Convert recommendations to order dictionaries
        orders = []
        for rec in order_recommendations:
            if self.validate_risk_constraints(
                current_position + (rec.quantity if rec.side == 'BUY' else -rec.quantity),
                0.0  # Placeholder for current PnL
            ):
                orders.append({
                    'symbol': rec.symbol,
                    'side': rec.side,
                    'quantity': rec.quantity,
                    'price': rec.price,
                    'order_type': rec.order_type,
                    'time_in_force': rec.time_in_force,
                    'metadata': {
                        'urgency': rec.urgency,
                        'expected_cost': rec.expected_cost,
                        'reasoning': rec.reasoning,
                        'strategy_id': self.strategy_id
                    }
                })
        
        return orders
    
    def _calculate_target_position(self, prediction: MLPrediction, current_position: float) -> float:
        """Calculate target position based on prediction"""
        
        # Only trade if confidence is high enough
        if prediction.confidence < 0.6:
            return current_position
        
        # Calculate position size based on prediction strength and confidence
        max_position = self.risk_constraints.max_position
        position_scaling = prediction.confidence * abs(prediction.prediction_value)
        
        if prediction.prediction_value > 0:
            # Bullish prediction - target long position
            target_position = current_position + (max_position * position_scaling * 0.1)
        elif prediction.prediction_value < 0:
            # Bearish prediction - target short position
            target_position = current_position - (max_position * position_scaling * 0.1)
        else:
            # Neutral prediction - reduce position
            target_position = current_position * 0.8
        
        # Ensure position limits
        target_position = max(-max_position, min(max_position, target_position))
        
        return target_position
    
    # Step 8: Continuous Learning & Model Adaptation
    def adapt_model(self, recent_performance: Dict[str, float]) -> bool:
        """Adapt model based on recent performance"""
        try:
            # Check if we need to retrain
            for symbol in self.symbols:
                if self._should_retrain_model(symbol, recent_performance):
                    success = self._retrain_model(symbol)
                    if success:
                        logger.info(f"Successfully retrained model for {symbol}")
                    else:
                        logger.warning(f"Failed to retrain model for {symbol}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error adapting model: {e}")
            return False
    
    def _should_retrain_model(self, symbol: str, recent_performance: Dict[str, float]) -> bool:
        """Determine if model should be retrained"""
        
        # Check time-based retraining
        if symbol in self.last_retraining:
            time_since_retrain = datetime.now() - self.last_retraining[symbol]
            if time_since_retrain < self.retraining_interval:
                return False
        
        # Check performance-based retraining
        if recent_performance.get('accuracy', 0.5) < 0.45:  # Below random
            return True
        
        if recent_performance.get('sharpe_ratio', 0) < -0.5:  # Poor risk-adjusted returns
            return True
        
        # Check if we have enough new training data
        if len(self.training_data[symbol]) > self.max_training_samples * 0.8:
            return True
        
        return False
    
    def _retrain_model(self, symbol: str) -> bool:
        """Retrain model for given symbol"""
        try:
            # Prepare training data
            if len(self.training_data[symbol]) < 500:
                logger.warning(f"Insufficient training data for {symbol}")
                return False
            
            # Convert to DataFrame
            df = pd.DataFrame(self.training_data[symbol])
            
            # Train new model
            new_model = self.formulate_model(df)
            
            if new_model is not None:
                # Calibrate model
                calibration_result = self.calibrate_model(new_model, df)
                
                # Update model if performance is acceptable
                if calibration_result.accuracy > 0.4:  # Better than random
                    self.models[symbol] = new_model
                    self.last_retraining[symbol] = datetime.now()
                    
                    logger.info(f"Model retrained for {symbol}. Accuracy: {calibration_result.accuracy:.3f}")
                    return True
                else:
                    logger.warning(f"New model performance too low for {symbol}: {calibration_result.accuracy:.3f}")
                    return False
            else:
                logger.error(f"Failed to train new model for {symbol}")
                return False
                
        except Exception as e:
            logger.error(f"Error retraining model for {symbol}: {e}")
            return False
    
    def store_training_sample(self, market_data: MarketMicrostructureData, features: MLFeatures) -> None:
        """Store market data and features for future training"""
        symbol = market_data.symbol
        
        # Create training sample
        sample = {
            'timestamp': market_data.timestamp,
            'symbol': symbol,
            'mid_price': market_data.mid_price,
            **self._features_to_dict(features),
            'future_direction': None  # Will be filled later
        }
        
        # Add to training data
        self.training_data[symbol].append(sample)
        
        # Limit training data size
        if len(self.training_data[symbol]) > self.max_training_samples:
            self.training_data[symbol] = self.training_data[symbol][-self.max_training_samples//2:]
    
    def get_strategy_status(self) -> Dict[str, Any]:
        """Get current strategy status and metrics"""
        status = {
            'strategy_id': self.strategy_id,
            'symbols': self.symbols,
            'model_type': self.model_type.value,
            'prediction_type': self.prediction_type.value,
            'models_trained': len(self.models),
            'training_samples': {symbol: len(data) for symbol, data in self.training_data.items()},
            'last_retraining': {symbol: time.isoformat() for symbol, time in self.last_retraining.items()},
            'recent_accuracy': {
                symbol: np.mean(list(acc_deque)) if len(acc_deque) > 0 else 0.0
                for symbol, acc_deque in self.prediction_accuracy.items()
            }
        }
        
        return status
