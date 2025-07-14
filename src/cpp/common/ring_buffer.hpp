#pragma once

#include <atomic>
#include <memory>
#include <array>

namespace hft::common {

/**
 * Lock-free Single Producer Single Consumer (SPSC) ring buffer
 * Optimized for ultra-low latency message passing between threads
 */
template<typename T, size_t Size>
class SPSCRingBuffer {
    static_assert((Size & (Size - 1)) == 0, "Size must be power of 2");
    
private:
    alignas(64) std::atomic<size_t> head_{0};  // Producer index
    alignas(64) std::atomic<size_t> tail_{0};  // Consumer index
    alignas(64) std::array<T, Size> buffer_;
    
    static constexpr size_t MASK = Size - 1;

public:
    /**
     * Push an element to the buffer (producer side)
     * Returns true if successful, false if buffer is full
     */
    bool push(const T& item) {
        const size_t current_head = head_.load(std::memory_order_relaxed);
        const size_t next_head = (current_head + 1) & MASK;
        
        if (next_head == tail_.load(std::memory_order_acquire)) {
            return false; // Buffer full
        }
        
        buffer_[current_head] = item;
        head_.store(next_head, std::memory_order_release);
        return true;
    }
    
    /**
     * Pop an element from the buffer (consumer side)
     * Returns true if successful, false if buffer is empty
     */
    bool pop(T& item) {
        const size_t current_tail = tail_.load(std::memory_order_relaxed);
        
        if (current_tail == head_.load(std::memory_order_acquire)) {
            return false; // Buffer empty
        }
        
        item = buffer_[current_tail];
        tail_.store((current_tail + 1) & MASK, std::memory_order_release);
        return true;
    }
    
    /**
     * Check if buffer is empty
     */
    bool empty() const {
        return tail_.load(std::memory_order_acquire) == head_.load(std::memory_order_acquire);
    }
    
    /**
     * Check if buffer is full
     */
    bool full() const {
        const size_t current_head = head_.load(std::memory_order_acquire);
        const size_t next_head = (current_head + 1) & MASK;
        return next_head == tail_.load(std::memory_order_acquire);
    }
    
    /**
     * Get current size (approximate)
     */
    size_t size() const {
        const size_t current_head = head_.load(std::memory_order_acquire);
        const size_t current_tail = tail_.load(std::memory_order_acquire);
        return (current_head - current_tail) & MASK;
    }
};

/**
 * Lock-free Multiple Producer Multiple Consumer (MPMC) ring buffer
 * Uses compare-and-swap operations for thread safety
 */
template<typename T, size_t Size>
class MPMCRingBuffer {
    static_assert((Size & (Size - 1)) == 0, "Size must be power of 2");
    
private:
    struct Cell {
        alignas(64) std::atomic<size_t> sequence{0};
        T data;
    };
    
    alignas(64) std::atomic<size_t> enqueue_pos_{0};
    alignas(64) std::atomic<size_t> dequeue_pos_{0};
    alignas(64) std::array<Cell, Size> buffer_;
    
    static constexpr size_t MASK = Size - 1;

public:
    MPMCRingBuffer() {
        for (size_t i = 0; i < Size; ++i) {
            buffer_[i].sequence.store(i, std::memory_order_relaxed);
        }
    }
    
    /**
     * Push an element to the buffer
     * Returns true if successful, false if buffer is full
     */
    bool push(const T& item) {
        Cell* cell;
        size_t pos = enqueue_pos_.load(std::memory_order_relaxed);
        
        while (true) {
            cell = &buffer_[pos & MASK];
            const size_t seq = cell->sequence.load(std::memory_order_acquire);
            const intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos);
            
            if (diff == 0) {
                if (enqueue_pos_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
                    break;
                }
            } else if (diff < 0) {
                return false; // Buffer full
            } else {
                pos = enqueue_pos_.load(std::memory_order_relaxed);
            }
        }
        
        cell->data = item;
        cell->sequence.store(pos + 1, std::memory_order_release);
        return true;
    }
    
    /**
     * Pop an element from the buffer
     * Returns true if successful, false if buffer is empty
     */
    bool pop(T& item) {
        Cell* cell;
        size_t pos = dequeue_pos_.load(std::memory_order_relaxed);
        
        while (true) {
            cell = &buffer_[pos & MASK];
            const size_t seq = cell->sequence.load(std::memory_order_acquire);
            const intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos + 1);
            
            if (diff == 0) {
                if (dequeue_pos_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
                    break;
                }
            } else if (diff < 0) {
                return false; // Buffer empty
            } else {
                pos = dequeue_pos_.load(std::memory_order_relaxed);
            }
        }
        
        item = cell->data;
        cell->sequence.store(pos + MASK + 1, std::memory_order_release);
        return true;
    }
};

} // namespace hft::common
