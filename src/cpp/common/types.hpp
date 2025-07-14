#pragma once

#include <cstdint>
#include <chrono>
#include <memory>
#include <string>
#include <atomic>

namespace hft::common {

// High-precision timestamp type
using Timestamp = std::chrono::nanoseconds;

// Get current timestamp with nanosecond precision
inline Timestamp get_timestamp() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()
    );
}

// Price type with fixed-point arithmetic (6 decimal places)
class Price {
private:
    int64_t value_; // Price * 1,000,000

public:
    static constexpr int64_t SCALE = 1000000;
    
    Price() : value_(0) {}
    explicit Price(double price) : value_(static_cast<int64_t>(price * SCALE)) {}
    explicit Price(int64_t raw_value) : value_(raw_value) {}
    
    double to_double() const { return static_cast<double>(value_) / SCALE; }
    int64_t raw_value() const { return value_; }
    
    Price operator+(const Price& other) const { return Price(value_ + other.value_); }
    Price operator-(const Price& other) const { return Price(value_ - other.value_); }
    Price operator*(double factor) const { return Price(static_cast<int64_t>(value_ * factor)); }
    
    bool operator==(const Price& other) const { return value_ == other.value_; }
    bool operator!=(const Price& other) const { return value_ != other.value_; }
    bool operator<(const Price& other) const { return value_ < other.value_; }
    bool operator>(const Price& other) const { return value_ > other.value_; }
    bool operator<=(const Price& other) const { return value_ <= other.value_; }
    bool operator>=(const Price& other) const { return value_ >= other.value_; }
};

// Quantity type
using Quantity = uint64_t;

// Symbol type
using Symbol = std::string;

// Order ID type
using OrderId = uint64_t;

// Side enumeration
enum class Side : uint8_t {
    BUY = 1,
    SELL = 2
};

// Order type enumeration
enum class OrderType : uint8_t {
    MARKET = 1,
    LIMIT = 2,
    STOP = 3,
    STOP_LIMIT = 4
};

// Order status enumeration
enum class OrderStatus : uint8_t {
    NEW = 1,
    PARTIALLY_FILLED = 2,
    FILLED = 3,
    CANCELLED = 4,
    REJECTED = 5,
    PENDING_CANCEL = 6,
    PENDING_NEW = 7
};

// Lock-free atomic counter for ID generation
class AtomicIdGenerator {
private:
    std::atomic<uint64_t> counter_;

public:
    AtomicIdGenerator(uint64_t start = 1) : counter_(start) {}
    
    uint64_t next() {
        return counter_.fetch_add(1, std::memory_order_relaxed);
    }
};

// Global ID generators
extern AtomicIdGenerator order_id_generator;
extern AtomicIdGenerator trade_id_generator;

} // namespace hft::common
