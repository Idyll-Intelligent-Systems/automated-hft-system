"""
ML Strategies Package
===================

Machine Learning strategies for HFT trading system implementing the modeling logic 
workflow from first principles as outlined in Modeling.md:

Real-Time Market Data → Data Cleaning & Feature Engineering → Statistical Patterns & 
Relationships Identification → Model Formulation → Model Calibration → Real-time 
Prediction → Optimal Order Placement → Continuous Learning & Model Adaptation

This package contains:
- Feature engineering modules for market microstructure indicators
- Statistical pattern identification algorithms
- ML models (classification, regression, reinforcement learning)
- Model calibration and backtesting frameworks
- Real-time prediction engines
- Optimal order placement logic
- Continuous learning and adaptation mechanisms
"""

from .base_ml_strategy import BaseMLStrategy
from .ml_workflow_engine import MLWorkflowEngine
from .market_data_processor import MarketDataProcessor
from .prediction_engine import PredictionEngine

__all__ = [
    'BaseMLStrategy',
    'MLWorkflowEngine', 
    'MarketDataProcessor',
    'PredictionEngine'
]
