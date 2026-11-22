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

# Velri engine: public C23 runtime API, sampler, streaming JSON, cascading placement, model layers & architectures.
add_library(velri_engine STATIC
    src/format/json.c
    src/config/placement.c
    src/config/placement_resolve.c
    src/config/placement_inspect.c
    src/model/layers.c
    src/model/arch/gpt2.c
    src/model/arch/llama.c
    src/model/arch_registry.c
    src/runtime/sampler.c
    src/runtime/context.c
    src/runtime/model.c
    src/runtime/program.c
    src/runtime/session.c
)
target_include_directories(velri_engine
    PUBLIC  "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
target_link_libraries(velri_engine PRIVATE VelriLib::velri)
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
    target_include_directories(test_gpt2_blocks PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
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

    add_executable(test_config_placement tests/test_config_placement.c)
    target_include_directories(test_config_placement PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(test_config_placement PRIVATE velri_engine VelriLib::velri)
    target_compile_definitions(test_config_placement PRIVATE
        VELRI_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    )
    if(MSVC OR CMAKE_C_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
        target_compile_options(test_config_placement PRIVATE /W4 /fp:precise /std:clatest)
    else()
        target_compile_options(test_config_placement PRIVATE -Wall -Wextra -Wpedantic -fno-fast-math)
    endif()
    if(UNIX)
        target_link_libraries(test_config_placement PRIVATE m)
    endif()
    add_test(NAME config_placement COMMAND test_config_placement)

    add_executable(test_engine_runtime tests/test_engine_runtime.c)
    target_include_directories(test_engine_runtime PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(test_engine_runtime PRIVATE velri_engine)
    if(MSVC OR CMAKE_C_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
        target_compile_options(test_engine_runtime PRIVATE /W4 /fp:precise /std:clatest)
    else()
        target_compile_options(test_engine_runtime PRIVATE -Wall -Wextra -Wpedantic -fno-fast-math)
    endif()
    if(UNIX)
        target_link_libraries(test_engine_runtime PRIVATE m)
    endif()
    add_test(NAME engine_runtime COMMAND test_engine_runtime)
endif()
