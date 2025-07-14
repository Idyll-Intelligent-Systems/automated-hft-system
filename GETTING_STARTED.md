# Getting Started with HFT Trading System

Welcome to the Automated High-Frequency Trading (HFT) System! This guide will help you get up and running with the system for development, testing, and deployment.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Building the System](#building-the-system)
4. [Running Tests](#running-tests)
5. [Development Workflow](#development-workflow)
6. [Deployment](#deployment)
7. [Monitoring](#monitoring)
8. [Troubleshooting](#troubleshooting)

## Prerequisites

### Hardware Requirements
- **CPU**: Intel Xeon Scalable or AMD EPYC (8+ cores recommended)
- **Memory**: 32GB+ RAM (128GB+ for production)
- **Storage**: 500GB+ NVMe SSD
- **Network**: 1Gb+ Ethernet (40/100Gb for production)

### Software Requirements
- **OS**: Ubuntu 22.04 LTS (real-time kernel for production)
- **Kernel**: Linux 5.15+ (RT kernel recommended)
- **Compiler**: GCC 11+ or Clang 15+
- **Python**: 3.11+
- **Docker**: 24.0+
- **Kubernetes**: 1.28+ (for production)

### Development Tools
- Git 2.30+
- CMake 3.20+
- Poetry (Python package manager)
- VS Code or CLion (recommended IDEs)

## Environment Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-org/automated-hft-system.git
cd automated-hft-system
```

### 2. Install System Dependencies
```bash
# Run the setup script
make setup

# Or manually install dependencies
sudo apt-get update
sudo apt-get install -y build-essential cmake git python3 python3-pip
sudo apt-get install -y libboost-all-dev libtbb-dev libfmt-dev libspdlog-dev
sudo apt-get install -y docker.io docker-compose

# Install Poetry for Python package management
curl -sSL https://install.python-poetry.org | python3 -
```

### 3. Setup Development Environment
```bash
make dev-setup
```

This will:
- Install pre-commit hooks
- Setup Python virtual environment
- Install development dependencies

## Building the System

### Quick Build (All Components)
```bash
make build
```

### Component-Specific Builds
```bash
# Build C++ trading engine
make build-cpp

# Build Python analytics components
make build-python

# Build Docker images
make docker
```

### Build Configuration
```bash
# Debug build
BUILD_TYPE=Debug make build-cpp

# Release build (default)
BUILD_TYPE=Release make build-cpp

# Verbose output
VERBOSE=1 make build
```

## Running Tests

### Run All Tests
```bash
make test
```

### Specific Test Suites
```bash
# Unit tests only
make test-cpp
make test-python

# Performance benchmarks
make benchmark

# Load testing
make load-test
```

### Test Coverage
```bash
# Generate coverage report
make test-python
# View coverage at src/python/htmlcov/index.html
```

## Development Workflow

### 1. Code Style and Linting
```bash
# Format and lint code
make lint

# Security scanning
make security
```

### 2. Development Cycle
```bash
# Start development environment
make dev-setup

# Make your changes...

# Run tests in watch mode
make dev-test

# Build and test
make ci-build
```

### 3. Git Workflow
```bash
git checkout -b feature/new-strategy
# Make changes
git add .
git commit -m "Add momentum strategy"
git push origin feature/new-strategy
# Create pull request
```

### 4. Pre-commit Hooks
The system includes pre-commit hooks that automatically:
- Format C++ code with clang-format
- Format Python code with black and isort
- Run linting checks
- Validate commit messages

## Configuration

### Environment-Specific Configuration
```bash
# Copy template configuration
cp configs/environments/development.yml.template configs/environments/development.yml

# Edit configuration
nano configs/environments/development.yml
```

### Key Configuration Areas
- **Network**: Exchange connections, latency targets
- **Trading**: Risk limits, position limits, strategies
- **Data**: Market data feeds, storage settings
- **Monitoring**: Metrics, alerting, logging

## Running the System

### Development Mode
```bash
make dev-run
```

### With Custom Configuration
```bash
./src/cpp/build/trading_engine --config configs/environments/development.yml
```

### Using Docker
```bash
docker-compose -f deployment/docker-compose.dev.yml up
```

## Deployment

### Local Development
```bash
make deploy-dev
```

### Staging Environment
```bash
make deploy-staging
```

### Production Deployment
```bash
# Requires confirmation
make deploy-prod-confirmed
```

### Monitoring Stack
```bash
make deploy-monitoring
```

Access monitoring dashboards:
- **Grafana**: http://localhost:3000 (admin/admin123)
- **Prometheus**: http://localhost:9090
- **Kibana**: http://localhost:5601
- **Jaeger**: http://localhost:16686

## Monitoring

### Real-time Monitoring
```bash
# Show monitoring dashboards
make monitor

# View system logs
make logs

# Check system status
make status
```

### Key Metrics to Monitor
- **Latency**: End-to-end execution time
- **Throughput**: Messages and orders per second
- **Fill Rate**: Order execution success rate
- **P&L**: Profit and loss tracking
- **Risk Metrics**: Position exposure, VaR

### Alerting
The system includes alerts for:
- Latency degradation (> 20μs)
- High error rates (> 1%)
- P&L breaches
- System component failures

## Development Guidelines

### C++ Development
- Use C++17/20 features
- Follow RAII principles
- Prefer stack allocation over heap
- Use lock-free data structures for critical paths
- Profile performance-critical code

### Python Development
- Use type hints for all functions
- Follow PEP 8 style guidelines
- Write comprehensive tests
- Use async/await for I/O operations
- Document public APIs

### Performance Guidelines
- Target < 10μs end-to-end latency
- Minimize memory allocations in hot paths
- Use CPU affinity for critical threads
- Profile with perf and Intel VTune
- Benchmark all performance changes

## Troubleshooting

### Common Issues

#### Build Failures
```bash
# Clean and rebuild
make clean
make build
```

#### Missing Dependencies
```bash
# Reinstall dependencies
make setup
```

#### Permission Issues
```bash
# Fix Docker permissions
sudo usermod -aG docker $USER
newgrp docker
```

#### Performance Issues
```bash
# Check system configuration
cat /proc/sys/kernel/sched_rt_runtime_us
cat /proc/sys/vm/swappiness

# Monitor resource usage
htop
iotop
```

### Debugging

#### C++ Debugging
```bash
# Build with debug symbols
BUILD_TYPE=Debug make build-cpp

# Run with GDB
gdb ./src/cpp/build/trading_engine
```

#### Python Debugging
```bash
# Run with debugger
cd src/python
poetry run python -m pdb -m hft.main
```

### Log Analysis
```bash
# View trading engine logs
tail -f /var/log/hft/trading.log

# Search for errors
grep ERROR /var/log/hft/trading.log

# View structured logs in Kibana
# Navigate to http://localhost:5601
```

### Performance Analysis
```bash
# Run latency tests
make benchmark

# Profile with perf
perf record -g ./src/cpp/build/trading_engine
perf report
```

## Next Steps

1. **Explore the Documentation**: Read through docs/ for detailed architecture information
2. **Review Strategies**: Check out strategies/ for trading strategy examples
3. **Customize Configuration**: Modify configs/ for your specific requirements
4. **Set Up Monitoring**: Deploy the monitoring stack for observability
5. **Performance Testing**: Run benchmarks to establish baseline performance

## Getting Help

- 📧 **Email**: support@hft-system.com
- 💬 **Slack**: #hft-trading-system
- 📖 **Wiki**: [Project Wiki](https://wiki.hft-system.com)
- 🐛 **Issues**: [GitHub Issues](https://github.com/your-org/automated-hft-system/issues)

## Contributing

Please read our contributing guidelines before submitting changes:
1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Run the full test suite
5. Submit a pull request with benchmarks

---

**⚠️ Important**: This is a high-frequency trading system. Always test thoroughly before deploying to production environments. Trading involves substantial risk of loss.
