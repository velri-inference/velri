# Typed CPU operations. Only their translation units receive ISA flags; callers,
# dispatch and portable kernels remain executable on the baseline architecture.
find_package(Threads REQUIRED)
option(VELRI_BUILD_CPU_BENCHMARK "Build the typed CPU comparison benchmark" OFF)

function(velri_cpu_objects name)
    cmake_parse_arguments(ARG "" "" "SOURCES;FLAGS" ${ARGN})
    add_library(${name} OBJECT ${ARG_SOURCES})
    target_include_directories(${name} PRIVATE include src/backend/cpu
        "${CMAKE_CURRENT_BINARY_DIR}/generated")
    target_compile_definitions(${name} PRIVATE VELRI_X86_VARIANTS=${VELRI_X86_VARIANTS})
    set_target_properties(${name} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    velri_strict_target(${name})
    target_compile_options(${name} PRIVATE ${VELRI_BASE_FLAGS} ${ARG_FLAGS} -Wno-overlength-strings)
    set(VELRI_CPU_OBJECTS ${VELRI_CPU_OBJECTS} $<TARGET_OBJECTS:${name}> PARENT_SCOPE)
endfunction()

velri_cpu_objects(velri_cpu_base FLAGS -ffp-contract=off SOURCES
    src/platform/cpu_caps.c src/platform/pool.c
    src/backend/cpu/convert_base.c src/backend/cpu/linear_dispatch.c
    src/backend/cpu/linear_base.c src/backend/cpu/linear_schedule.c
    src/backend/cpu/ops_dispatch.c src/backend/cpu/ops_base.c)

if(VELRI_X86_VARIANTS)
    include(cmake/gemm.cmake)
    set(avx2 -mavx2 -mfma -mf16c -ffp-contract=on)
    set(avx512 ${avx2} -mavx512f -mavx512bw -mavx512dq -mavx512vl)
    velri_cpu_objects(velri_cpu_avx2 FLAGS ${avx2} SOURCES
        src/backend/cpu/convert_avx2.c src/backend/cpu/linear_avx2.c
        src/backend/cpu/ops_avx2.c src/backend/cpu/kernels/gemv.c
        src/backend/cpu/kernels/elementwise.c)
    velri_cpu_objects(velri_cpu_avx512 FLAGS ${avx512} SOURCES
        src/backend/cpu/convert_avx512.c src/backend/cpu/linear_avx512_fp.c
        src/backend/cpu/ops_avx512.c src/backend/cpu/kernels/gemv.c
        src/backend/cpu/kernels/elementwise.c)
    velri_cpu_objects(velri_cpu_bf16 FLAGS ${avx512} -mavx512bf16 SOURCES
        src/backend/cpu/linear_avx512_bf16.c)
    velri_cpu_objects(velri_cpu_vnni FLAGS ${avx512} -mavx512vnni SOURCES
        src/backend/cpu/linear_avx512_int8.c)
endif()


add_library(velri_cpu STATIC ${VELRI_CPU_OBJECTS})
add_library(VelriLib::cpu ALIAS velri_cpu)
set_target_properties(velri_cpu PROPERTIES EXPORT_NAME cpu)
target_include_directories(velri_cpu PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(velri_cpu PUBLIC Threads::Threads)
if(UNIX)
    target_link_libraries(velri_cpu PUBLIC m)
endif()
if(VELRI_SANITIZE)
    target_link_options(velri_cpu INTERFACE -fsanitize=address,undefined)
endif()

if(VELRI_LIB_BUILD_TESTS)
    add_library(velri_typed_reference STATIC tests/cpu/typed/reference.c)
    target_include_directories(velri_typed_reference PUBLIC tests/cpu/typed
        PRIVATE src/backend/cpu include)
    target_compile_options(velri_typed_reference PRIVATE ${VELRI_BASE_FLAGS} -ffp-contract=off)
    velri_strict_target(velri_typed_reference)
    foreach(name IN ITEMS test_linear_typed test_ops test_contract)
        add_executable(${name} tests/cpu/typed/${name}.c)
        target_include_directories(${name} PRIVATE src/backend/cpu)
        target_compile_definitions(${name} PRIVATE VELRI_X86_VARIANTS=${VELRI_X86_VARIANTS})
        target_compile_options(${name} PRIVATE ${VELRI_BASE_FLAGS} -ffp-contract=off)
        target_link_libraries(${name} PRIVATE velri_cpu velri_typed_reference)
        velri_strict_target(${name})
        add_test(NAME ${name} COMMAND ${name})
    endforeach()
endif()

if(VELRI_BUILD_CPU_BENCHMARK)
    add_executable(bench_cpu tools/bench_cpu.c)
    target_compile_options(bench_cpu PRIVATE ${VELRI_BASE_FLAGS})
    target_link_libraries(bench_cpu PRIVATE velri_cpu)
    velri_strict_target(bench_cpu)
endif()
