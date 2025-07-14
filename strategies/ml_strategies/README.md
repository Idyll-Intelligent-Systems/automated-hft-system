# ML Strategies for HFT System

This directory contains the complete implementation of the ML modeling workflow from `Modeling.md` translated into production-ready code for the HFT system.

## 🎯 Overview

The ML strategies framework implements the 8-step ML modeling logic:

1. **Real-Time Market Data Processing** - Clean, validate, and normalize market microstructure data
2. **Data Cleaning & Feature Engineering** - Extract meaningful signals from raw market data
3. **Statistical Patterns & Relationships Identification** - Detect market regime changes and patterns
4. **Model Formulation (Statistical / ML-Based)** - Develop predictive models using ensemble methods
5. **Model Calibration (Backtest & Optimize)** - Validate model performance with rigorous testing
6. **Real-time Prediction/Classification** - Generate actionable predictions with confidence scores
7. **Optimal Order Placement Logic** - Execute trades using optimal execution algorithms
8. **Continuous Learning & Model Adaptation** - Continuously improve models with new data

## 📁 Directory Structure

```
ml_strategies/
├── __init__.py                    # Package interface and imports
├── base_ml_strategy.py           # Abstract base class for all ML strategies
├── ml_workflow_engine.py         # Orchestrates the complete ML workflow
├── ml_strategy_integration.py    # System integration and strategy management
├── market_data_processor.py      # Real-time data cleaning and validation
├── prediction_engine.py          # Real-time prediction generation
├── config.yml                    # Configuration for all ML strategies
├── README.md                     # This file
├── feature_engineering/          # Feature extraction modules
│   └── __init__.py               # Microstructure, statistical, technical features
├── models/                       # Advanced ML models
│   └── advanced_models.py        # Hawkes process, RL, Bayesian models
├── optimal_execution/            # Order placement algorithms
│   └── __init__.py               # Avellaneda-Stoikov, Almgren-Chriss, RL execution
└── examples/                     # Example strategy implementations
    └── market_direction_strategy.py  # Complete market direction prediction strategy
```

## 🚀 Quick Start

### 1. Basic Usage

```python
import asyncio
from ml_strategies.ml_strategy_integration import create_market_direction_strategy_manager
from ml_strategies.base_ml_strategy import MarketMicrostructureData

# Create strategy manager
config = {
    'max_workers': 4,
    'model_storage_path': 'models/ml_strategies'
}
symbols = ['AAPL', 'GOOGL', 'TSLA']
manager = create_market_direction_strategy_manager(symbols, config)

# Start the system
async def run_strategies():
    await manager.start()
    
    # Process market data
    market_data = MarketMicrostructureData(
        timestamp=1234567890000000000,
        symbol='AAPL',
        best_bid=150.00,
        best_ask=150.05,
        bid_size=1000,
        ask_size=1200,
        last_price=150.02,
        last_size=100,
        volume=50000,
        vwap=150.01,
        high=151.00,
        low=149.50,
        open_price=150.50
    )
    
    # Process through strategies
    results = await manager.process_market_data(market_data)
    print(f"Processing results: {results}")
    
    # Get predictions
    predictions = await manager.get_aggregated_predictions('AAPL')
    print(f"Predictions: {predictions}")
    
    # Get orders
    orders = manager.get_all_strategy_orders()
    print(f"Orders: {orders}")

# Run the example
asyncio.run(run_strategies())
```

### 2. Custom Strategy Implementation

```python
from ml_strategies.base_ml_strategy import BaseMLStrategy, ModelType, PredictionType

class MyCustomStrategy(BaseMLStrategy):
    def __init__(self, symbols, config):
        super().__init__(
            strategy_id="my_custom_strategy",
            symbols=symbols,
            model_type=ModelType.CLASSIFICATION,
            prediction_type=PredictionType.MARKET_DIRECTION,
            risk_constraints=risk_constraints,
            config=config
        )
    
    def process_market_data(self, market_data):
        # Implement step 1: Real-time data processing
        pass
    
    def engineer_features(self, market_data):
        # Implement step 2: Feature engineering
        pass
    
    # ... implement all 8 workflow steps
```

## 🔧 Configuration

The system is configured through `config.yml`. Key sections:

### System Configuration
```yaml
system:
  max_workers: 4
  model_storage_path: "models/ml_strategies"
  log_level: "INFO"
```

### Strategy Configuration
```yaml
strategies:
  market_direction_v1:
    symbols: ["AAPL", "GOOGL", "TSLA"]
    max_position: 15000
    retrain_hours: 4
    min_confidence_threshold: 0.65
```

### Risk Management
```yaml
risk_management:
  position_limits:
    max_total_exposure: 100000
    max_single_symbol_exposure: 20000
  circuit_breakers:
    daily_loss_limit: 25000
    drawdown_limit: 0.10
```

## 📊 Features

### Feature Engineering
The system extracts multiple feature categories:

- **Microstructure Features**: Order book imbalance, spread-depth ratio, bid-ask dynamics
- **Statistical Features**: Volatility clustering, autocorrelation, mean reversion signals
- **Technical Features**: Price momentum, volume momentum, VWAP deviation
- **Time Features**: Time of day, day of week, session indicators
- **Order Flow Features**: Trade flow analysis, market impact estimation

### Model Types
Supports various ML approaches:

- **Ensemble Methods**: XGBoost, Random Forest, Neural Networks
- **Advanced Models**: Hawkes processes, Queue theory models, Reinforcement Learning
- **Bayesian Methods**: Bayesian updating, Bayesian Neural Networks
- **Time Series**: ARMA-GARCH, Cointegration models

### Execution Algorithms
Implements optimal execution strategies:

- **Avellaneda-Stoikov**: Market making with inventory control
- **Almgren-Chriss**: Optimal execution with market impact
- **RL-based Execution**: Reinforcement learning for dynamic execution

## 📈 Performance Monitoring

### Real-time Metrics
- Prediction accuracy and confidence
- Sharpe ratio and Sortino ratio
- Maximum drawdown
- Win rate and profit factor
- Feature importance tracking

### System Health
- Memory usage monitoring
- Processing latency tracking
- Model performance degradation detection
- Automatic model retraining triggers

## 🔄 Workflow Steps in Detail

### Step 1: Real-Time Market Data Processing
```python
def process_market_data(self, market_data: MarketMicrostructureData) -> bool:
    # Validate data quality
    # Check for anomalies
    # Store for training
    # Return processing status
```

### Step 2: Feature Engineering
```python
def engineer_features(self, market_data: MarketMicrostructureData) -> MLFeatures:
    # Extract microstructure features
    # Calculate statistical indicators
    # Compute technical features
    # Add time-based features
    # Return feature vector
```

### Step 3: Statistical Pattern Identification
```python
def identify_statistical_patterns(self, features: MLFeatures) -> Dict[str, float]:
    # Detect volatility clustering
    # Identify mean reversion patterns
    # Analyze momentum signals
    # Return pattern scores
```

### Step 4: Model Formulation
```python
def formulate_model(self, training_data: pd.DataFrame) -> Any:
    # Prepare training data
    # Split train/validation
    # Train ensemble models
    # Select best model
    # Return trained model
```

### Step 5: Model Calibration
```python
def calibrate_model(self, model: Any, validation_data: pd.DataFrame) -> ModelCalibrationResult:
    # Validate model performance
    # Calculate risk metrics
    # Simulate trading performance
    # Return calibration results
```

### Step 6: Real-time Prediction
```python
def generate_prediction(self, features: MLFeatures) -> MLPrediction:
    # Apply trained model
    # Calculate confidence
    # Generate prediction
    # Return prediction with metadata
```

### Step 7: Optimal Order Placement
```python
def calculate_optimal_orders(self, 
                           prediction: MLPrediction,
                           current_position: float,
                           market_data: MarketMicrostructureData) -> List[Dict[str, Any]]:
    # Convert prediction to target position
    # Apply risk constraints
    # Use optimal execution algorithm
    # Return order recommendations
```

### Step 8: Continuous Learning
```python
def adapt_model(self, recent_performance: Dict[str, float]) -> bool:
    # Monitor model performance
    # Detect degradation
    # Trigger retraining if needed
    # Update model parameters
    # Return success status
```

## 🛡️ Risk Management

### Position Limits
- Maximum position per symbol
- Total portfolio exposure limits
- Sector concentration limits
- Leverage constraints

### Performance Monitoring
- Real-time P&L tracking
- Value at Risk (VaR) calculation
- Maximum drawdown monitoring
- Sharpe ratio tracking

### Circuit Breakers
- Daily loss limits
- Drawdown thresholds
- Position limit violations
- Model performance degradation

## 🧪 Testing and Validation

### Backtesting
```python
# Example backtesting setup
from ml_strategies.examples.market_direction_strategy import MarketDirectionStrategy

strategy = MarketDirectionStrategy(['AAPL'], config)

# Run backtest
results = strategy.backtest(
    start_date='2023-01-01',
    end_date='2023-12-31',
    initial_capital=1000000
)

print(f"Backtest results: {results}")
```

### Model Validation
- Walk-forward analysis
- Cross-validation
- Out-of-sample testing
- Stress testing scenarios

## 📝 Integration Points

### Data Feeds
- Real-time market data ingestion
- Historical data for training
- Alternative data sources
- Data quality monitoring

### Order Management
- Order routing and execution
- Fill notifications
- Position tracking
- Trade reporting

### Risk Systems
- Pre-trade risk checks
- Post-trade monitoring
- Compliance validation
- Regulatory reporting

## 🔧 Deployment

### Production Setup
1. Configure database connections
2. Set up data feeds
3. Deploy strategy containers
4. Configure monitoring
5. Enable circuit breakers

### Scaling
- Horizontal scaling with multiple workers
- Model serving optimization
- Caching strategies
- Load balancing

## 📚 Advanced Topics

### Model Research Areas
Based on `Modeling.md`, the framework supports research in:

- **Market Microstructure**: Order flow analysis, liquidity modeling
- **Machine Learning**: Deep learning, ensemble methods, feature selection
- **Optimal Execution**: Transaction cost analysis, market impact modeling
- **Risk Management**: Portfolio optimization, tail risk management
- **Alternative Data**: News sentiment, social media, satellite data

### Extension Points
- Custom feature extractors
- New model implementations
- Alternative execution algorithms
- Enhanced risk models

## 🤝 Contributing

1. Follow the 8-step workflow pattern
2. Implement comprehensive testing
3. Add proper documentation
4. Include performance metrics
5. Validate risk management

## 📞 Support

For questions about the ML strategies framework:

1. Check the configuration in `config.yml`
2. Review example implementations in `examples/`
3. Examine the base classes for required methods
4. Test with simulated data before live deployment

## ⚠️ Important Notes

- **Risk Management**: Always validate strategies in paper trading first
- **Data Quality**: Ensure high-quality, clean market data
- **Model Validation**: Thoroughly backtest before deployment
- **Performance Monitoring**: Continuously monitor strategy performance
- **Compliance**: Ensure regulatory compliance for live trading

This implementation provides a production-ready framework for ML-based HFT strategies, following the complete workflow outlined in `Modeling.md` while maintaining the flexibility for research and enhancement.
