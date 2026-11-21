cmake_minimum_required(VERSION 3.21)
project(velri VERSION 0.1.0 LANGUAGES C)

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

include(CTest)

# The computation library lives in this repository and also builds on its own.
set(VELRI_LIB_DEFAULT_SOURCE "")
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/velri-lib/CMakeLists.txt")
    set(VELRI_LIB_DEFAULT_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/velri-lib")
endif()
set(VELRI_LIB_SOURCE_DIR "${VELRI_LIB_DEFAULT_SOURCE}" CACHE PATH "Velri-lib source directory; empty uses an installed package")
if(VELRI_LIB_SOURCE_DIR)
    get_filename_component(VELRI_LIB_SOURCE_DIR "${VELRI_LIB_SOURCE_DIR}" ABSOLUTE
        BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
    if(NOT EXISTS "${VELRI_LIB_SOURCE_DIR}/CMakeLists.txt")
        message(FATAL_ERROR "VELRI_LIB_SOURCE_DIR must point to a Velri-lib checkout")
    endif()
    set(VELRI_LIB_BUILD_TESTS ${BUILD_TESTING} CACHE BOOL "Build dependency correctness tests")
    add_subdirectory("${VELRI_LIB_SOURCE_DIR}" velri-lib)
else()
    find_package(VelriLib 0.1.0 EXACT CONFIG REQUIRED)
endif()

# Velri engine model architectures (thin metadata-driven graphs composed from VelriLib blocks).
add_library(velri_engine STATIC
    src/model/arch/gpt2.c
)
target_include_directories(velri_engine PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
target_link_libraries(velri_engine PUBLIC VelriLib::velri)
if(MSVC OR CMAKE_C_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    target_compile_options(velri_engine PRIVATE /W4 /fp:precise /std:clatest)
else()
    target_compile_options(velri_engine PRIVATE -Wall -Wextra -Wpedantic -fno-fast-math)
endif()
if(UNIX)
    target_link_libraries(velri_engine PRIVATE m)
endif()

# Integration & End-to-End Model Tests
if(BUILD_TESTING)
    add_executable(test_velri_lib tests/integration/velri_lib.c)
    target_link_libraries(test_velri_lib PRIVATE VelriLib::velri)
    add_test(NAME velri_lib_integration COMMAND test_velri_lib)

    add_executable(test_gpt2_blocks tests/test_gpt2_blocks.c)
    target_link_libraries(test_gpt2_blocks PRIVATE velri_engine VelriLib::velri)
    if(MSVC OR CMAKE_C_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
        target_compile_options(test_gpt2_blocks PRIVATE /W4 /fp:precise /std:clatest)
    else()
        target_compile_options(test_gpt2_blocks PRIVATE -Wall -Wextra -Wpedantic -fno-fast-math)
    endif()
    if(UNIX)
        target_link_libraries(test_gpt2_blocks PRIVATE m)
    endif()
    add_test(NAME gpt2_blocks COMMAND test_gpt2_blocks)
endif()
