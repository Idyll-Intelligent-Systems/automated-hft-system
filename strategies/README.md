# Trading Strategies

This directory contains trading strategy implementations and configurations.

## Structure

```
strategies/
├── market_making/         # Market making strategies
├── arbitrage/            # Arbitrage strategies
├── momentum/             # Momentum strategies
├── mean_reversion/       # Mean reversion strategies
├── ml_strategies/        # Machine learning strategies
├── templates/            # Strategy templates
└── configs/              # Strategy configurations
```

## Strategy Types

### 1. Market Making
- **Objective**: Provide liquidity and capture bid-ask spread
- **Methods**: Optimal spread calculation, inventory management
- **Models**: Avellaneda-Stoikov, Guilbaud-Pham

### 2. Statistical Arbitrage
- **Objective**: Exploit temporary price divergences
- **Methods**: Pairs trading, cointegration, mean reversion
- **Models**: Kalman filters, state-space models

### 3. Momentum
- **Objective**: Capture price trends and momentum
- **Methods**: Technical indicators, breakout detection
- **Models**: Time series momentum, cross-sectional momentum

### 4. Mean Reversion
- **Objective**: Trade on price reversions to fair value
- **Methods**: Bollinger bands, RSI, statistical measures
- **Models**: Ornstein-Uhlenbeck process

### 5. Machine Learning
- **Objective**: Use ML models for prediction and execution
- **Methods**: Supervised learning, reinforcement learning
- **Models**: XGBoost, Neural networks, Q-learning

## Strategy Framework

### Base Strategy Interface
```cpp
class Strategy {
public:
    virtual ~Strategy() = default;
    virtual void on_market_data(const MarketData& data) = 0;
    virtual void on_order_update(const OrderUpdate& update) = 0;
    virtual void on_trade(const Trade& trade) = 0;
    virtual void on_timer() = 0;
};
```

### Risk Controls
- Position limits
- Maximum drawdown
- Daily loss limits
- Concentration limits
- Leverage constraints

### Performance Metrics
- Sharpe ratio
- Sortino ratio
- Maximum drawdown
- Win rate
- Profit factor
- Calmar ratio

## Configuration Management

### Strategy Parameters
- Risk parameters
- Model parameters
- Execution parameters
- Market parameters

### Environment-specific Configs
- Development
- Testing
- Staging
- Production

## Hot-Reload Capability

Strategies support hot-reload for parameter updates without system restart:

```python
strategy_manager.update_parameters('market_maker_v1', {
    'spread_multiplier': 1.2,
    'max_position': 1000,
    'risk_limit': 0.02
})
```
