cmake_minimum_required(VERSION 3.16)
project(PDPF LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_VERBOSE_MAKEFILE ON)

# Ensure Apple SDK include paths are available (fixes missing standard headers on some setups).
if(APPLE)
    execute_process(COMMAND xcrun --show-sdk-path
                    OUTPUT_VARIABLE _sdk_path
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(_sdk_path)
        set(CMAKE_OSX_SYSROOT "${_sdk_path}")
        include_directories(SYSTEM "${_sdk_path}/usr/include/c++/v1")
    endif()
endif()

if(APPLE)
    execute_process(COMMAND xcode-select -p
                    OUTPUT_VARIABLE _xcs_root
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(_xcs_root)
        include_directories(SYSTEM "${_xcs_root}/usr/include/c++/v1")
    endif()
endif()

add_library(pdpf STATIC
    pdpf/src/core/types.cpp
    pdpf/src/core/group.cpp
    pdpf/src/prg/prg.cpp
    pdpf/src/pprf/pprf.cpp
    pdpf/src/pdpf/pdpf_binary.cpp
    pdpf/src/pdpf/pdpf_group.cpp
    pdpf/src/ldc/reed_muller_ldc.cpp
    pdpf/src/pdpf/pdpf_amplified.cpp
    pdpf/src/pdpf/pdpf_lut.cpp
)

target_include_directories(pdpf PUBLIC pdpf/include)

add_executable(pdpf_demo main.cpp)
target_link_libraries(pdpf_demo PRIVATE pdpf)

add_executable(pdpf_tests pdpf/tests/test_pdpf.cpp)
target_link_libraries(pdpf_tests PRIVATE pdpf)

# Composite-FSS header-only demo/tests
add_executable(composite_fss_tests composite_fss/tests/test_composite_fss.cpp)
target_include_directories(composite_fss_tests PRIVATE composite_fss/include pdpf/include)
target_link_libraries(composite_fss_tests PRIVATE pdpf)
target_compile_definitions(composite_fss_tests PRIVATE COMPOSITE_FSS_INTERNAL=1 COMPOSITE_FSS_ENABLE_OPEN=1)

add_executable(composite_fss_test_suf_packing composite_fss/tests/test_suf_packing.cpp)
target_include_directories(composite_fss_test_suf_packing PRIVATE composite_fss/include)
target_compile_definitions(composite_fss_test_suf_packing PRIVATE COMPOSITE_FSS_INTERNAL=1)

# Compile-time check that raw opens are blocked when COMPOSITE_FSS_INTERNAL is 0.
add_executable(composite_fss_no_raw_open composite_fss/tests/test_no_raw_open.cpp)
target_include_directories(composite_fss_no_raw_open PRIVATE composite_fss/include)
target_compile_definitions(composite_fss_no_raw_open PRIVATE COMPOSITE_FSS_INTERNAL=0)

# Strict-mode SPDZ regression tests (no raw getters, share-only paths).
add_executable(composite_fss_strict_spdz composite_fss/tests/test_strict_spdz.cpp)
target_include_directories(composite_fss_strict_spdz PRIVATE composite_fss/include pdpf/include)
target_link_libraries(composite_fss_strict_spdz PRIVATE pdpf)
target_compile_definitions(composite_fss_strict_spdz PRIVATE COMPOSITE_FSS_INTERNAL=0)

# Microbenchmark harness
add_executable(bench_composite_fss composite_fss/bench/bench_composite_fss.cpp)
target_include_directories(bench_composite_fss PRIVATE composite_fss/include pdpf/include)
target_link_libraries(bench_composite_fss PRIVATE pdpf)
target_compile_definitions(bench_composite_fss PRIVATE COMPOSITE_FSS_INTERNAL=1)
