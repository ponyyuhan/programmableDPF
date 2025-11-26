#ifdef COMPOSITE_FSS_CUDA

#include "pdpf_engine_cuda.hpp"
#include "cuda_utils.cuh"

#include <cuda_runtime.h>
#include <stdexcept>

namespace cfss {

__device__ inline std::uint64_t device_prg(std::uint64_t seed, std::size_t idx, unsigned word) {
    // Deterministic splitmix64-style generator seeded by (idx, word).
    std::uint64_t s = seed ^ (static_cast<std::uint64_t>(idx) * 0x9E3779B97F4A7C15ULL +
                              static_cast<std::uint64_t>(word) * 0xD1B54A32D192ED03ULL);
    s += 0x9e3779b97f4a7c15ULL;
    s = (s ^ (s >> 30)) * 0xbf58476d1ce4e5b9ULL;
    s = (s ^ (s >> 27)) * 0x94d049bb133111ebULL;
    s = s ^ (s >> 31);
    return s;
}

__global__ void lut_eval_kernel(const std::uint64_t *d_table,
                                const std::uint64_t *d_inputs,
                                std::uint64_t *d_outputs,
                                std::size_t outputs_per_point,
                                std::uint64_t domain_mask,
                                std::size_t domain_size,
                                std::size_t n_inputs,
                                int party,
                                std::uint64_t share_mask,
                                std::uint64_t seed) {
    std::size_t i = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
                    static_cast<std::size_t>(threadIdx.x);
    if (i >= n_inputs) return;
    std::uint64_t x_hat = d_inputs[i];
    std::size_t idx = static_cast<std::size_t>(x_hat & domain_mask);
    if (domain_size > 0 && idx >= domain_size) {
        idx %= domain_size;
    }
    std::size_t table_base = idx * outputs_per_point;
    std::size_t out_base = i * outputs_per_point;
    for (std::size_t w = 0; w < outputs_per_point; ++w) {
        std::uint64_t val = d_table[table_base + w] & share_mask;
        std::uint64_t r = device_prg(seed, idx, static_cast<unsigned>(w)) & share_mask;
        std::uint64_t out = (party == 0) ? r : ((val - r) & share_mask);
        d_outputs[out_base + w] = out;
    }
}

PdpfEngineCuda::PdpfEngineCuda(unsigned n_bits, std::uint64_t seed)
    : n_bits_(n_bits),
      seed_(seed) {
    share_mask_ = (n_bits_ >= 64) ? ~0ULL : ((1ULL << n_bits_) - 1ULL);
}

PdpfEngineCuda::~PdpfEngineCuda() {
    release();
}

PdpfProgramId PdpfEngineCuda::make_lut_program(const LutProgramDesc &desc,
                                               const std::vector<std::uint64_t> &table_flat) {
    return make_lut_program(desc, table_flat, std::nullopt);
}

PdpfProgramId PdpfEngineCuda::make_lut_program(const LutProgramDesc &desc,
                                               const std::vector<std::uint64_t> &table_flat,
                                               std::optional<PdpfProgramId> forced_id) {
    PdpfProgramId pid = forced_id.has_value() ? *forced_id
                                              : static_cast<PdpfProgramId>(programs_.size());
    PdpfProgramGpuMeta meta{};
    meta.domain_bits = desc.domain_bits;
    meta.outputs_per_point = desc.output_words ? desc.output_words : 1;
    meta.lut_size_words = table_flat.size();

    // Clean up any previous entry for the same pid to avoid leaks when reusing ids.
    auto it = programs_.find(pid);
    if (it != programs_.end()) {
        free_meta(it->second);
    }

    std::size_t bytes = meta.lut_size_words * sizeof(std::uint64_t);
    cuda_check(cudaMalloc(&meta.d_table, bytes), "cudaMalloc(table)");
    cuda_check(cudaMemcpy(meta.d_table, table_flat.data(), bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy(table)");
    programs_[pid] = meta;
    return pid;
}

void PdpfEngineCuda::eval_share(PdpfProgramId pid,
                                int party,
                                std::uint64_t x_hat,
                                std::vector<std::uint64_t> &out_words) {
    PdpfProgramGpuMeta &meta = lookup(pid);
    if (out_words.size() != meta.outputs_per_point) {
        out_words.resize(meta.outputs_per_point);
    }
    eval_share_batch(pid, party, &x_hat, 1, out_words.data());
}

void PdpfEngineCuda::eval_share_batch(PdpfProgramId pid,
                                      int party,
                                      const std::uint64_t *x_hats,
                                      std::size_t n_inputs,
                                      std::uint64_t *out_words) {
    if (x_hats == nullptr || out_words == nullptr || n_inputs == 0) return;
    PdpfProgramGpuMeta &meta = lookup(pid);

    std::uint64_t *d_inputs = nullptr;
    std::uint64_t *d_outputs = nullptr;
    std::size_t out_count = n_inputs * meta.outputs_per_point;
    cuda_check(cudaMalloc(&d_inputs, n_inputs * sizeof(std::uint64_t)), "cudaMalloc(inputs)");
    cuda_check(cudaMalloc(&d_outputs, out_count * sizeof(std::uint64_t)), "cudaMalloc(outputs)");
    cuda_check(cudaMemcpy(d_inputs, x_hats, n_inputs * sizeof(std::uint64_t), cudaMemcpyHostToDevice),
               "cudaMemcpy(inputs)");

    std::uint64_t domain_mask = (meta.domain_bits == 64) ? ~0ULL : ((1ULL << meta.domain_bits) - 1ULL);
    std::size_t domain_size = (meta.domain_bits == 64)
                                  ? ((meta.outputs_per_point > 0) ? (meta.lut_size_words / meta.outputs_per_point) : 0)
                                  : (1ULL << meta.domain_bits);
    int threads = 256;
    auto grid = make_grid(n_inputs, threads);
    lut_eval_kernel<<<grid, threads>>>(meta.d_table, d_inputs, d_outputs,
                                       meta.outputs_per_point, domain_mask,
                                       domain_size, n_inputs, party, share_mask_, seed_);
    cuda_check(cudaGetLastError(), "lut_eval_kernel launch");
    cuda_check(cudaMemcpy(out_words, d_outputs, out_count * sizeof(std::uint64_t), cudaMemcpyDeviceToHost),
               "cudaMemcpy(outputs)");

    cuda_check(cudaFree(d_inputs), "cudaFree(inputs)");
    cuda_check(cudaFree(d_outputs), "cudaFree(outputs)");
}

void PdpfEngineCuda::release() {
    for (auto &kv : programs_) {
        free_meta(kv.second);
    }
    programs_.clear();
}

PdpfProgramGpuMeta &PdpfEngineCuda::lookup(PdpfProgramId pid) {
    auto it = programs_.find(pid);
    if (it == programs_.end()) {
        throw std::runtime_error("PdpfEngineCuda: invalid program id");
    }
    return it->second;
}

void PdpfEngineCuda::free_meta(PdpfProgramGpuMeta &meta) {
    if (meta.d_table) {
        cudaFree(meta.d_table);
        meta.d_table = nullptr;
    }
}

} // namespace cfss

#endif // COMPOSITE_FSS_CUDA
