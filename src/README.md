# Source Code Structure

This directory contains the core HFT trading system source code.

## Structure

```
src/
├── cpp/                    # C++ core components
│   ├── common/            # Common utilities and data structures
│   ├── network/           # Network layer and protocol handlers
│   ├── data/              # Market data processing
│   ├── strategy/          # Strategy engine
│   ├── risk/              # Risk management
│   ├── order/             # Order management system
│   └── gateway/           # Exchange gateways
├── python/                # Python components
│   ├── analytics/         # Data analytics and ML
│   ├── backtesting/       # Backtesting framework
│   ├── monitoring/        # System monitoring
│   ├── tools/             # Development tools
│   └── api/               # REST API services
├── rust/                  # Rust components (alternative implementations)
│   ├── data_processor/    # High-performance data processing
│   └── order_router/      # Ultra-low latency order routing
└── shared/                # Shared libraries and interfaces
    ├── protocols/         # Protocol definitions (FIX, ITCH, etc.)
    ├── schemas/           # Data schemas
    └── configs/           # Configuration templates
```

## Build System

### C++
- **Build Tool**: CMake 3.20+
- **Compiler**: GCC 11+ or Clang 15+
- **Standard**: C++17/20
- **Dependencies**: Boost, TBB, FMT, spdlog

### Python
- **Version**: Python 3.11+
- **Package Manager**: Poetry
- **Key Libraries**: NumPy, Pandas, scikit-learn, asyncio

### Rust
- **Version**: Rust 1.70+
- **Build Tool**: Cargo
- **Key Crates**: tokio, serde, rayon, crossbeam

## Performance Requirements

| Component | Latency Target | Throughput |
|-----------|---------------|------------|
| Data Processing | < 2μs | > 1M msg/s |
| Strategy Engine | < 5μs | > 100K orders/s |
| Risk Engine | < 1μs | > 500K checks/s |
| Order Router | < 1μs | > 50K orders/s |
