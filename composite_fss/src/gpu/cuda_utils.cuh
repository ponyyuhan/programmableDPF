#pragma once

#ifdef COMPOSITE_FSS_CUDA

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace cfss {

inline void cuda_check(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
    }
}

inline dim3 make_grid(std::size_t n, int block_dim) {
    std::size_t blocks = (n + static_cast<std::size_t>(block_dim) - 1) / static_cast<std::size_t>(block_dim);
    return dim3(static_cast<unsigned>(blocks));
}

} // namespace cfss

#endif // COMPOSITE_FSS_CUDA
