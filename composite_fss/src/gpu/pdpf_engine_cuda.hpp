#pragma once

#ifdef COMPOSITE_FSS_CUDA

#include "../../include/composite_fss/pdpf.hpp"
#include <cstddef>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>

namespace cfss {

struct PdpfProgramGpuMeta {
    std::size_t domain_bits = 0;
    std::size_t outputs_per_point = 0;
    std::size_t lut_size_words = 0;
    std::uint64_t *d_table = nullptr;
};

class PdpfEngineCuda {
public:
    explicit PdpfEngineCuda(unsigned n_bits, std::uint64_t seed = 0xA5A5);
    ~PdpfEngineCuda();

    PdpfProgramId make_lut_program(const LutProgramDesc &desc,
                                   const std::vector<std::uint64_t> &table_flat);

    PdpfProgramId make_lut_program(const LutProgramDesc &desc,
                                   const std::vector<std::uint64_t> &table_flat,
                                   std::optional<PdpfProgramId> forced_id);

    void eval_share(PdpfProgramId pid,
                    int party,
                    std::uint64_t x_hat,
                    std::vector<std::uint64_t> &out_words);

    void eval_share_batch(PdpfProgramId pid,
                          int party,
                          const std::uint64_t *x_hats,
                          std::size_t n_inputs,
                          std::uint64_t *out_words);

    void release();

private:
    PdpfProgramGpuMeta &lookup(PdpfProgramId pid);
    void free_meta(PdpfProgramGpuMeta &meta);

    std::unordered_map<PdpfProgramId, PdpfProgramGpuMeta> programs_;
    unsigned n_bits_ = 0;
    std::uint64_t seed_ = 0;
    std::uint64_t share_mask_ = 0;
};

} // namespace cfss

#endif // COMPOSITE_FSS_CUDA
