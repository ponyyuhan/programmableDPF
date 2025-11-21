// ================================================================
// File: include/pdpf/pprf/pprf.hpp
// ================================================================

#pragma once

#include "pdpf/core/types.hpp"
#include "pdpf/prg/prg.hpp"
#include <memory>
#include <vector>
#include <limits>

namespace pdpf::pprf {

/**
 * Parameters for GGM-based PPRF as in Theorem 3. :contentReference[oaicite:8]{index=8}
 *
 * Input domain [M], output domain [N].
 */
struct PprfParams {
    std::uint64_t M = 0;  // input domain size
    std::uint64_t N = 0;  // output domain size
};

/**
 * Master PPRF key: root seed + params.
 */
struct PprfKey {
    core::Seed   root_seed{};
    PprfParams   params{};
};

/**
 * Punctured PPRF key:
 *  - co_path_seeds: sibling seeds along the path to xp.
 *  - xp: punctured input point.
 */
struct PprfPuncturedKey {
    std::vector<core::Seed> co_path_seeds;
    std::uint64_t           xp = 0;
    PprfParams              params{};
};

/**
 * GGM-based PPRF with puncturing.
 *
 * This class only defines the *interfaces* and high-level behavior.
 * Codex should fill in the GGM tree details according to Theorem 3. :contentReference[oaicite:9]{index=9}
 */
class Pprf {
public:
    explicit Pprf(std::shared_ptr<prg::IPrg> prg);

    /**
     * PRF evaluation: y = Eval(k, x) ∈ [N].
     */
    std::uint64_t eval(const PprfKey &k, std::uint64_t x) const;

    /**
     * Full-domain evaluation: out[x] = Eval(k, x) for x ∈ [0, M-1].
     * out is resized to M.
     */
    void eval_all(const PprfKey &k, std::vector<std::uint64_t> &out) const;

    /**
     * Puncture at xp: return punctured key kp that allows evaluation for all x ≠ xp.
     */
    PprfPuncturedKey puncture(const PprfKey &k, std::uint64_t xp) const;

    /**
     * Punctured evaluation:
     *  - If x ≠ xp: returns Eval(k, x).
     *  - If x == xp: behavior is implementation-defined (e.g., return sentinel).
     */
    std::uint64_t punc_eval(const PprfPuncturedKey &kp, std::uint64_t x) const;

    /**
     * Full-domain evaluation for punctured key:
     *  out[x] = punc_eval(kp, x) for all x ∈ [0, M-1].
     */
    void punc_eval_all(const PprfPuncturedKey &kp,
                       std::vector<std::uint64_t> &out) const;

    /**
     * Sentinel value to indicate "punctured" at xp.
     * By default, we use max uint64.
     */
    static constexpr std::uint64_t PUNCTURED_SENTINEL =
        std::numeric_limits<std::uint64_t>::max();

private:
    std::shared_ptr<prg::IPrg> prg_;

    std::uint32_t tree_depth(std::uint64_t M) const;
    void seed_to_children(const core::Seed &parent,
                          core::Seed &left,
                          core::Seed &right) const;
    std::uint64_t seed_to_uint64(const core::Seed &seed) const;
};

} // namespace pdpf::pprf
