// ================================================================
// File: include/pdpf/core/types.hpp
// ================================================================

#pragma once

#include <array>
#include <cstdint>
#include <vector>
#include <random>
#include <string>
#include <stdexcept>

namespace pdpf::core {

/**
 * Security parameters for PDPF and related constructions.
 *
 * - lambda_bits: security parameter λ (typically 128).
 * - domain_size_N: size of PDPF domain [N].
 * - epsilon: target privacy error ε (Theorem 4). :contentReference[oaicite:3]{index=3}
 */
struct SecurityParams {
    std::uint32_t lambda_bits = 128;
    std::uint64_t domain_size_N = 0;
    double        epsilon = 0.0;
};

/**
 * Seed type for PRG and GGM-based constructions.
 * λ = 128 → 16 bytes, but you can generalize later if needed.
 */
using Seed = std::array<std::uint8_t, 16>;

/**
 * Simple RAII buffer that zeroizes on destruction.
 * Used for key material and seeds.
 */
template<typename T>
class SecureBuffer {
public:
    SecureBuffer() = default;
    explicit SecureBuffer(std::size_t n) : buf_(n) {}
    ~SecureBuffer() { secure_zeroize(); }

    SecureBuffer(const SecureBuffer&) = delete;
    SecureBuffer& operator=(const SecureBuffer&) = delete;

    SecureBuffer(SecureBuffer&& other) noexcept : buf_(std::move(other.buf_)) {}
    SecureBuffer& operator=(SecureBuffer&& other) noexcept {
        if (this != &other) {
            secure_zeroize();
            buf_ = std::move(other.buf_);
        }
        return *this;
    }

    std::size_t size() const noexcept { return buf_.size(); }
    bool empty() const noexcept { return buf_.empty(); }

    T* data() noexcept { return buf_.data(); }
    const T* data() const noexcept { return buf_.data(); }

    T& operator[](std::size_t i) { return buf_[i]; }
    const T& operator[](std::size_t i) const { return buf_[i]; }

    std::vector<T>& vec() noexcept { return buf_; }
    const std::vector<T>& vec() const noexcept { return buf_; }

    void resize(std::size_t n) {
        secure_zeroize();
        buf_.resize(n);
    }

private:
    std::vector<T> buf_;

    void secure_zeroize() noexcept {
        volatile T *p = buf_.data();
        for (std::size_t i = 0; i < buf_.size(); ++i) {
            p[i] = T{};
        }
    }
};

/**
 * Cryptographically secure random generator wrapper.
 *
 * Used for:
 *  - sampling seeds (Seed)
 *  - sampling random indices, etc.
 */
class RandomDevice {
public:
    RandomDevice();

    /// Fill Seed with random bytes from OS CSPRNG.
    void random_seed(Seed &seed);

    /// Sample uint64_t in [0, bound-1]. bound must be > 0.
    std::uint64_t random_u64(std::uint64_t bound);

private:
    std::random_device rd_;
};

} // namespace pdpf::core
