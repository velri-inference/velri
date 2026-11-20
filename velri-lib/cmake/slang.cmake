# Opt-in Khronos Slang AOT kernel compilation for CPU (host-object-code) and Vulkan (SPIR-V 1.6).
# Both backends default to OFF so bare C23 CPU builds and CI remain dependency-free.

option(VELRI_ENABLE_SLANG_CPU "Build Slang AOT CPU kernels and comparison benchmark" OFF)
option(VELRI_ENABLE_VULKAN "Build Slang AOT Vulkan kernels and hardware-ceiling benchmark" OFF)

if(NOT VELRI_ENABLE_SLANG_CPU AND NOT VELRI_ENABLE_VULKAN)
    return()
endif()

find_program(VELRI_SLANGC NAMES slangc
    HINTS "$ENV{HOME}/.local/bin" "$ENV{HOME}/.local/opt/slang/bin")
if(NOT VELRI_SLANGC)
    message(FATAL_ERROR "VELRI_ENABLE_SLANG_CPU / VELRI_ENABLE_VULKAN require `slangc` in PATH")
endif()

set(VELRI_SLANG_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/slang")
set(VELRI_SLANG_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/slang")
file(MAKE_DIRECTORY "${VELRI_SLANG_GEN_DIR}")

set(VELRI_SLANG_COMMON_DEPS
    "${VELRI_SLANG_DIR}/common/numeric.slang"
    "${VELRI_SLANG_DIR}/common/activations.slang"
    "${VELRI_SLANG_DIR}/common/epilogue.slang"
    "${VELRI_SLANG_DIR}/common/reductions.slang"
    "${VELRI_SLANG_DIR}/ops/norm.slang"
    "${VELRI_SLANG_DIR}/ops/rope.slang"
    "${VELRI_SLANG_DIR}/ops/elementwise.slang"
    "${VELRI_SLANG_DIR}/ops/linear_cpu.slang"
)

if(NOT TARGET velri_typed_reference)
    add_library(velri_typed_reference STATIC tests/cpu/typed/reference.c)
    target_include_directories(velri_typed_reference PUBLIC tests/cpu/typed
        PRIVATE src/backend/cpu include)
    target_compile_options(velri_typed_reference PRIVATE ${VELRI_BASE_FLAGS} -ffp-contract=off)
    velri_strict_target(velri_typed_reference)
endif()

# ============================================================================
# 1. Slang AOT CPU Backend (ONE source compiled per ISA flag set)
# ============================================================================
if(VELRI_ENABLE_SLANG_CPU)
    set(VELRI_SLANG_CPU_ENTRY "${VELRI_SLANG_DIR}/entry_cpu.slang")
    set(VELRI_SLANG_CPU_OBJS)

    function(velri_add_slang_cpu_variant isa mr nr_f32 nr_f64 features)
        set(_obj "${VELRI_SLANG_GEN_DIR}/slang_cpu_${isa}.o")
        set(_cmd "${VELRI_SLANGC}" "${VELRI_SLANG_CPU_ENTRY}"
            -I "${VELRI_SLANG_DIR}/common" -I "${VELRI_SLANG_DIR}/ops"
            -target host-object-code -O3 -fp-mode fast
            "-DVELRI_ISA=${isa}" "-DVELRI_MR=${mr}"
            "-DVELRI_NR_F32=${nr_f32}" "-DVELRI_NR_F64=${nr_f64}")
        if(features)
            list(APPEND _cmd -llvm-features "${features}")
        endif()
        list(APPEND _cmd -o "${_obj}")

        add_custom_command(
            OUTPUT "${_obj}"
            COMMAND ${_cmd}
            DEPENDS "${VELRI_SLANG_CPU_ENTRY}" ${VELRI_SLANG_COMMON_DEPS}
            COMMENT "Slang AOT CPU (${isa}): entry_cpu.slang -> slang_cpu_${isa}.o"
            VERBATIM
        )
        set_source_files_properties("${_obj}" PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
        set(VELRI_SLANG_CPU_OBJS ${VELRI_SLANG_CPU_OBJS} "${_obj}" PARENT_SCOPE)
    endfunction()

    velri_add_slang_cpu_variant(base 3 16 8 "")
    if(VELRI_X86_VARIANTS)
        velri_add_slang_cpu_variant(avx2 3 32 16 "+avx2,+fma,+f16c")
        velri_add_slang_cpu_variant(avx512 6 64 32 "+avx512f,+avx512bw,+avx512dq,+avx512vl,+avx512vnni,+fma,+f16c")
    endif()

    add_library(velri_slang_cpu STATIC
        src/backend/cpu/slang_cpu_dispatch.c
        ${VELRI_SLANG_CPU_OBJS}
    )
    add_library(VelriLib::slang_cpu ALIAS velri_slang_cpu)
    set_target_properties(velri_slang_cpu PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(velri_slang_cpu PRIVATE VELRI_X86_VARIANTS=${VELRI_X86_VARIANTS})
    target_compile_options(velri_slang_cpu PRIVATE ${VELRI_BASE_FLAGS})
    velri_strict_target(velri_slang_cpu)
    target_link_libraries(velri_slang_cpu PUBLIC velri_cpu)

    add_executable(bench_slang_cpu tools/bench_slang_cpu.c)
    target_include_directories(bench_slang_cpu PRIVATE src/backend/cpu)
    target_compile_definitions(bench_slang_cpu PRIVATE VELRI_X86_VARIANTS=${VELRI_X86_VARIANTS})
    target_compile_options(bench_slang_cpu PRIVATE ${VELRI_BASE_FLAGS})
    velri_strict_target(bench_slang_cpu)
    target_link_libraries(bench_slang_cpu PRIVATE velri_slang_cpu velri_cpu velri_typed_reference)

    if(VELRI_LIB_BUILD_TESTS)
        add_test(NAME slang_cpu_verify COMMAND bench_slang_cpu --verify-only)
    endif()
endif()

# ============================================================================
# 2. Slang AOT Vulkan Backend (SPIR-V 1.6 + spirv-val + embedded header)
# ============================================================================
if(VELRI_ENABLE_VULKAN)
    find_path(VELRI_VULKAN_INCLUDE_DIR vulkan/vulkan.h
        HINTS "$ENV{HOME}/.local/opt/vulkan-headers/include" /usr/include /usr/local/include)
    if(NOT VELRI_VULKAN_INCLUDE_DIR)
        message(FATAL_ERROR "VELRI_ENABLE_VULKAN requires Vulkan-Headers (vulkan/vulkan.h)")
    endif()
    find_program(VELRI_SPIRV_VAL NAMES spirv-val)

    set(VELRI_SLANG_VK_ENTRY "${VELRI_SLANG_DIR}/entry_vulkan.slang")
    set(VELRI_VK_SPV_DIR "${VELRI_SLANG_GEN_DIR}/spv")
    file(MAKE_DIRECTORY "${VELRI_VK_SPV_DIR}")

    set(VELRI_VK_ENTRIES
        vk_rmsnorm_f32
        vk_rmsnorm_bf16
        vk_rmsnorm_f16
        vk_fused_add_rmsnorm_f32
        vk_layernorm_f32
        vk_rope_f32
        vk_rope_bf16
        vk_activation_silu_f32
        vk_activation_gelu_f32
        vk_activation_relu2_f32
        vk_gated_silu_f32
        vk_gated_gelu_f32
        vk_vec_axpby_f32
        vk_vec_mul_f32
        vk_softmax_f32
        vk_fwht_f32
        vk_quantize_row_f32_i8
        vk_gemv_f32
        vk_gemv_f16
        vk_gemv_bf16
        vk_gemv_w8a8
        vk_gemm_f32
        vk_splitk_reduce
        vk_gemm_coopmat_f16
        vk_gemm_coopmat_bf16
        vk_gemm_coopmat_w8a8
        vk_gemm_coopmat_f16_64
        vk_gemm_coopmat_bf16_64
        vk_gemm_coopmat_w8a8_64
        vk_ceiling_bw_f32x4
        vk_ceiling_bw_read_f32x4
        vk_ceiling_fma_f32
        vk_ceiling_coopmat_f16
        vk_ceiling_coopmat_bf16
        vk_ceiling_coopmat_i8
    )

    set(VELRI_VK_SPV_FILES)
    foreach(_entry IN LISTS VELRI_VK_ENTRIES)
        set(_spv "${VELRI_VK_SPV_DIR}/${_entry}.spv")
        set(_extra_caps)
        if(_entry STREQUAL "vk_gemm_coopmat_bf16" OR _entry STREQUAL "vk_gemm_coopmat_bf16_64" OR _entry STREQUAL "vk_ceiling_coopmat_bf16")
            list(APPEND _extra_caps -capability spvBFloat16KHR)
        endif()

        if(VELRI_SPIRV_VAL)
            add_custom_command(
                OUTPUT "${_spv}"
                COMMAND "${VELRI_SLANGC}" "${VELRI_SLANG_VK_ENTRY}"
                    -I "${VELRI_SLANG_DIR}/common" -I "${VELRI_SLANG_DIR}/ops"
                    -target spirv -profile spirv_1_6 -Wno-41012 -fvk-use-c-layout -O3
                    ${_extra_caps}
                    -entry "${_entry}" -stage compute -o "${_spv}"
                COMMAND "${VELRI_SPIRV_VAL}" --target-env vulkan1.3 "${_spv}"
                DEPENDS "${VELRI_SLANG_VK_ENTRY}" ${VELRI_SLANG_COMMON_DEPS}
                COMMENT "Slang AOT SPIR-V + spirv-val: ${_entry}.spv"
                VERBATIM
            )
        else()
            add_custom_command(
                OUTPUT "${_spv}"
                COMMAND "${VELRI_SLANGC}" "${VELRI_SLANG_VK_ENTRY}"
                    -I "${VELRI_SLANG_DIR}/common" -I "${VELRI_SLANG_DIR}/ops"
                    -target spirv -profile spirv_1_6 -Wno-41012 -fvk-use-c-layout -O3
                    ${_extra_caps}
                    -entry "${_entry}" -stage compute -o "${_spv}"
                DEPENDS "${VELRI_SLANG_VK_ENTRY}" ${VELRI_SLANG_COMMON_DEPS}
                COMMENT "Slang AOT SPIR-V: ${_entry}.spv"
                VERBATIM
            )
        endif()
        list(APPEND VELRI_VK_SPV_FILES "${_spv}")
    endforeach()

    set(VELRI_VK_SHADER_HEADER "${VELRI_SLANG_GEN_DIR}/velri_vk_shaders.h")
    string(JOIN ";" _entry_list_arg ${VELRI_VK_ENTRIES})
    add_custom_command(
        OUTPUT "${VELRI_VK_SHADER_HEADER}"
        COMMAND "${CMAKE_COMMAND}"
            "-DINPUT_DIR=${VELRI_VK_SPV_DIR}"
            "-DOUTPUT_HEADER=${VELRI_VK_SHADER_HEADER}"
            "-DENTRY_LIST=${_entry_list_arg}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_spv.cmake"
        DEPENDS ${VELRI_VK_SPV_FILES} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_spv.cmake"
        COMMENT "Embedding SPIR-V shaders -> velri_vk_shaders.h"
        VERBATIM
    )

    add_library(velri_vulkan STATIC
        src/backend/vulkan/vk_backend.c
        "${VELRI_VK_SHADER_HEADER}"
    )
    add_library(VelriLib::vulkan ALIAS velri_vulkan)
    set_target_properties(velri_vulkan PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_include_directories(velri_vulkan PRIVATE
        "${VELRI_SLANG_GEN_DIR}"
        "${VELRI_VULKAN_INCLUDE_DIR}"
    )
    target_compile_options(velri_vulkan PRIVATE ${VELRI_BASE_FLAGS})
    velri_strict_target(velri_vulkan)
    target_link_libraries(velri_vulkan PUBLIC velri_cpu PRIVATE ${CMAKE_DL_LIBS})

    add_executable(bench_vulkan tools/bench_vulkan.c)
    target_include_directories(bench_vulkan PRIVATE src/backend/cpu)
    target_compile_options(bench_vulkan PRIVATE ${VELRI_BASE_FLAGS})
    velri_strict_target(bench_vulkan)
    target_link_libraries(bench_vulkan PRIVATE velri_vulkan velri_cpu velri_typed_reference)

    if(VELRI_LIB_BUILD_TESTS)
        add_test(NAME vulkan_verify COMMAND bench_vulkan --verify-only)
    endif()
endif()
