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
