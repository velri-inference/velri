cmake_minimum_required(VERSION 3.21)
project(VelriLib VERSION 0.1.0 LANGUAGES C)
include(GNUInstallDirs)

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    set(VELRI_LIB_TESTS_DEFAULT ON)
else()
    set(VELRI_LIB_TESTS_DEFAULT OFF)
endif()
option(VELRI_LIB_BUILD_TESTS "Build Velri-lib's independent correctness suite" ${VELRI_LIB_TESTS_DEFAULT})
if(VELRI_LIB_BUILD_TESTS)
    enable_testing()
endif()

if(NOT CMAKE_C_COMPILER_ID MATCHES "Clang|GNU")
    message(FATAL_ERROR "Velri-lib requires Clang (including clang-cl) or GCC with C23 support")
endif()

set(VELRI_MSVC_FRONTEND OFF)
if(MSVC OR CMAKE_C_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    set(VELRI_MSVC_FRONTEND ON)
endif()
set(VELRI_WINDOWS_MSVC_ABI OFF)
if(WIN32 AND (MSVC OR CMAKE_C_SIMULATE_ID STREQUAL "MSVC"))
    set(VELRI_WINDOWS_MSVC_ABI ON)
endif()

include(CheckCCompilerFlag)
include(CheckCSourceCompiles)
if(VELRI_MSVC_FRONTEND)
    check_c_compiler_flag("/std:clatest" VELRI_HAS_C23_FLAG)
    if(NOT VELRI_HAS_C23_FLAG)
        message(FATAL_ERROR "Velri-lib requires C23 mode (/std:clatest); use Clang 18+ (clang-cl)")
    endif()
else()
    check_c_compiler_flag("-std=c23" VELRI_HAS_C23_FLAG)
    if(NOT VELRI_HAS_C23_FLAG)
        message(FATAL_ERROR "Velri-lib requires C23 mode; use Clang 18+ or GCC 14+")
    endif()
endif()

option(VELRI_CPU_MULTIVERSION "Build x86-64 AVX2 and AVX-512 AOT variants from the single Slang source" ON)
option(VELRI_SANITIZE "Enable address and undefined-behavior sanitizers" OFF)
option(VELRI_ENABLE_VULKAN "Build Slang AOT Vulkan backend (SPIR-V 1.6) and hardware-ceiling benchmark" OFF)

set(VELRI_X86_VARIANTS 0)
set(VELRI_BASE_FLAGS)
check_c_source_compiles("
#if !defined(__x86_64__) && !defined(_M_X64)
#error Not an x86-64 compiler target
#endif
int main(void) { return 0; }
" VELRI_TARGET_X86_64)
check_c_source_compiles("
#if !defined(__aarch64__) && !defined(_M_ARM64)
#error Not an AArch64 compiler target
#endif
int main(void) { return 0; }
" VELRI_TARGET_AARCH64)

if(VELRI_TARGET_X86_64)
    if(NOT VELRI_MSVC_FRONTEND AND NOT ANDROID)
        set(VELRI_BASE_FLAGS -march=x86-64)
    endif()
    if(VELRI_CPU_MULTIVERSION)
        set(VELRI_X86_VARIANTS 1)
    endif()
endif()

function(velri_strict_target target)
    if(VELRI_MSVC_FRONTEND)
        target_compile_options(${target} PRIVATE /W4 /fp:precise /std:clatest)
        target_compile_definitions(${target} PRIVATE _CRT_SECURE_NO_WARNINGS)
        if(VELRI_SANITIZE)
            target_compile_options(${target} PRIVATE -fsanitize=address)
            target_link_options(${target} PRIVATE -fsanitize=address)
        endif()
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -fno-fast-math)
        if(VELRI_WINDOWS_MSVC_ABI)
            target_compile_definitions(${target} PRIVATE _CRT_SECURE_NO_WARNINGS)
        endif()
        if(VELRI_SANITIZE)
            target_compile_options(${target} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
            target_link_options(${target} PRIVATE -fsanitize=address,undefined)
        endif()
    endif()
endfunction()

# Slang AOT CPU kernels (required) + optional Vulkan SPIR-V 1.6 backend
include(cmake/slang.cmake)

# Install CMake package exporting VelriLib::cpu (and VelriLib::vulkan when enabled)
include(CMakePackageConfigHelpers)
set(_velri_install_targets velri_cpu)
if(VELRI_ENABLE_VULKAN)
    list(APPEND _velri_install_targets velri_vulkan)
endif()
install(TARGETS ${_velri_install_targets} EXPORT VelriLibTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT VelriLibTargets NAMESPACE VelriLib::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/VelriLib)
configure_package_config_file(cmake/VelriLibConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/VelriLibConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/VelriLib)
write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/VelriLibConfigVersion.cmake"
    VERSION ${PROJECT_VERSION} COMPATIBILITY ExactVersion)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/VelriLibConfig.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/VelriLibConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/VelriLib)
