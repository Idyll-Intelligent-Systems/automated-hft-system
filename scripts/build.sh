#!/bin/bash

# HFT System Build Script
# Builds all components of the HFT trading system

set -e

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_TYPE="${BUILD_TYPE:-Release}"
NUM_CORES=$(nproc)
VERBOSE="${VERBOSE:-0}"

# Global targets array to store build targets parsed from command line
TARGETS=()

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check dependencies
check_dependencies() {
    log_info "Checking build dependencies..."
    
    local deps=("cmake" "g++" "python3" "poetry" "docker")
    local missing_deps=()
    
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            missing_deps+=("$dep")
        fi
    done
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        log_error "Missing dependencies: ${missing_deps[*]}"
        log_info "Please install missing dependencies and try again"
        exit 1
    fi
    
    log_success "All dependencies found"
}

# Build C++ components
build_cpp() {
    log_info "Building C++ components..."
    
    cd "$PROJECT_ROOT/src/cpp"
    
    # Create build directory
    mkdir -p build
    cd build
    
    # Configure with CMake
    cmake -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
          -DCMAKE_CXX_COMPILER=g++ \
          -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
          ..
    
    # Build
    if [ "$VERBOSE" = "1" ]; then
        make -j"$NUM_CORES" VERBOSE=1
    else
        make -j"$NUM_CORES"
    fi
    
    log_success "C++ components built successfully"
}

# Build Python components
build_python() {
    log_info "Building Python components..."
    
    cd "$PROJECT_ROOT/src/python"
    
    # Install dependencies
    poetry install
    
    # Run tests
    poetry run pytest tests/ --cov=hft
    
    # Build package
    poetry build
    
    log_success "Python components built successfully"
}

# Build Docker images
build_docker() {
    log_info "Building Docker images..."
    
    cd "$PROJECT_ROOT"
    
    # Build trading engine image
    docker build -f docker/Dockerfile.trading-engine \
                 -t hft/trading-engine:latest .
    
    # Build analytics service image
    docker build -f docker/Dockerfile.analytics \
                 -t hft/analytics:latest .
    
    # Build monitoring image
    docker build -f docker/Dockerfile.monitoring \
                 -t hft/monitoring:latest .
    
    log_success "Docker images built successfully"
}

# Run tests
run_tests() {
    log_info "Running test suite..."
    
    # C++ tests
    cd "$PROJECT_ROOT/src/cpp/build"
    if [ -f "test_runner" ]; then
        ./test_runner
    else
        log_warning "C++ test runner not found, skipping C++ tests"
    fi
    
    # Python tests
    cd "$PROJECT_ROOT/src/python"
    poetry run pytest tests/ -v
    
    log_success "All tests passed"
}

# Performance benchmarks
run_benchmarks() {
    log_info "Running performance benchmarks..."
    
    cd "$PROJECT_ROOT/src/cpp/build"
    if [ -f "benchmark_runner" ]; then
        ./benchmark_runner
    else
        log_warning "Benchmark runner not found, skipping benchmarks"
    fi
    
    log_success "Benchmarks completed"
}

# Clean build artifacts
clean() {
    log_info "Cleaning build artifacts..."
    
    # Clean C++ build
    rm -rf "$PROJECT_ROOT/src/cpp/build"
    
    # Clean Python build
    cd "$PROJECT_ROOT/src/python"
    poetry run python -c "import shutil; shutil.rmtree('dist', ignore_errors=True)"
    poetry run python -c "import shutil; shutil.rmtree('.pytest_cache', ignore_errors=True)"
    
    log_success "Build artifacts cleaned"
}

# Print usage
usage() {
    echo "Usage: $0 [OPTIONS] [TARGETS]"
    echo ""
    echo "TARGETS:"
    echo "  cpp          Build C++ components"
    echo "  python       Build Python components"
    echo "  docker       Build Docker images"
    echo "  test         Run test suite"
    echo "  benchmark    Run performance benchmarks"
    echo "  clean        Clean build artifacts"
    echo "  all          Build all components (default)"
    echo ""
    echo "OPTIONS:"
    echo "  -h, --help   Show this help message"
    echo "  -v, --verbose Enable verbose output"
    echo "  --debug      Build with debug symbols"
    echo "  --release    Build optimized release (default)"
    echo ""
    echo "ENVIRONMENT VARIABLES:"
    echo "  BUILD_TYPE   Build type (Debug|Release)"
    echo "  VERBOSE      Enable verbose output (0|1)"
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                usage
                exit 0
                ;;
            -v|--verbose)
                VERBOSE=1
                shift
                ;;
            --debug)
                BUILD_TYPE="Debug"
                shift
                ;;
            --release)
                BUILD_TYPE="Release"
                shift
                ;;
            cpp|python|docker|test|benchmark|clean|all)
                TARGETS+=("$1")
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
    
    # Default to 'all' if no targets specified
    if [ ${#TARGETS[@]} -eq 0 ]; then
        TARGETS=("all")
    fi
}

# Main function
main() {
    parse_args "$@"
    
    log_info "Starting HFT system build..."
    log_info "Build type: $BUILD_TYPE"
    log_info "Targets: ${TARGETS[*]}"
    
    check_dependencies
    
    for target in "${TARGETS[@]}"; do
        case $target in
            cpp)
                build_cpp
                ;;
            python)
                build_python
                ;;
            docker)
                build_docker
                ;;
            test)
                run_tests
                ;;
            benchmark)
                run_benchmarks
                ;;
            clean)
                clean
                ;;
            all)
                build_cpp
                build_python
                build_docker
                run_tests
                ;;
            *)
                log_error "Unknown target: $target"
                exit 1
                ;;
        esac
    done
    
    log_success "Build completed successfully!"
}

# Run main function
main "$@"
