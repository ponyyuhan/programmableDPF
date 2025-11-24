#define COMPOSITE_FSS_INTERNAL 0
#include "../include/composite_fss/ring.hpp"

#include <type_traits>
#include <utility>

int main() {
    // The raw getter should be unavailable or deleted when COMPOSITE_FSS_INTERNAL=0.
    static_assert(COMPOSITE_FSS_INTERNAL == 0, "strict build must disable internal access");
    // Compilation of this TU should succeed only because we never touch raw getters;
    // any accidental call to raw_value_unsafe/value_internal would be rejected.
    return 0;
}
