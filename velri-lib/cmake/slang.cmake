# Khronos Slang AOT kernel compilation for CPU (mandatory) and Vulkan (optional).
# Pure build-time AOT via `slangc`: zero compiler/LLVM/libslang runtime dependency.
#
# Two CPU AOT lowering modes (`VELRI_SLANG_CPU_MODE`):
#   - LLVM_IR (default for compatible Clang): `slangc -target llvm-ir -O0` emits
#     target-neutral LLVM IR, stripped of host triple/datalayout, and CMake's own
#     Clang compiles it to `.o`/`.obj` for ANY OS & CPU architecture (Linux, Windows,
#     Android, macOS; x86-64, AArch64, RISC-V 64/RVV, etc.) without hardcoding triples.
#   - OBJECT (automatic fallback for GCC or older Clang): `slangc -target host-object-code`
#     emits machine objects directly via bundled `slang-llvm` (x86-64 & AArch64).

find_program(VELRI_SLANGC NAMES slangc
    HINTS "$ENV{HOME}/.local/bin" "$ENV{HOME}/.local/opt/slang/bin"
    NO_CMAKE_FIND_ROOT_PATH)
if(NOT VELRI_SLANGC)
    message(FATAL_ERROR
        "Velri-lib requires Khronos `slangc` in PATH (or ~/.local/bin) for AOT kernel compilation")
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

set(_velri_obj_ext "${CMAKE_C_OUTPUT_EXTENSION}")
if(NOT _velri_obj_ext)
    if(WIN32)
        set(_velri_obj_ext ".obj")
    else()
        set(_velri_obj_ext ".o")
    endif()
endif()

# Collect target/ABI flags from CMake so Clang `.ll -> .o` inherits the exact
# cross-compilation toolchain target without hardcoding OS or CPU architecture.
set(_velri_clang_target_flags)
if(CMAKE_C_COMPILER_TARGET)
    list(APPEND _velri_clang_target_flags "--target=${CMAKE_C_COMPILER_TARGET}")
endif()
if(CMAKE_SYSROOT)
    list(APPEND _velri_clang_target_flags "--sysroot=${CMAKE_SYSROOT}")
endif()
if(NOT WIN32)
    list(APPEND _velri_clang_target_flags -fPIC)
endif()
set(_velri_raw_c_flags "${CMAKE_C_FLAGS}")
separate_arguments(_velri_c_flag_list NATIVE_COMMAND "${_velri_raw_c_flags}")
set(_velri_next_is_target OFF)
foreach(_f IN LISTS _velri_c_flag_list)
    if(_velri_next_is_target)
        list(APPEND _velri_clang_target_flags "${_f}")
        set(_velri_next_is_target OFF)
    elseif(_f STREQUAL "-target")
        list(APPEND _velri_clang_target_flags "${_f}")
        set(_velri_next_is_target ON)
    elseif(_f MATCHES "^(--target=|--sysroot=|-march=|-mcpu=|-mtune=|-mabi=|-mfpu=|-mfloat-abi=)")
        list(APPEND _velri_clang_target_flags "${_f}")
    endif()
endforeach()

# Determine CPU lowering mode: AUTO | LLVM_IR | OBJECT
set(VELRI_SLANG_CPU_MODE "AUTO" CACHE STRING
    "Slang CPU AOT lowering mode: AUTO (LLVM_IR for compatible Clang, OBJECT for GCC/older Clang), LLVM_IR, or OBJECT")
set_property(CACHE VELRI_SLANG_CPU_MODE PROPERTY STRINGS AUTO LLVM_IR OBJECT)

set(_velri_use_llvm_ir OFF)
if(VELRI_SLANG_CPU_MODE STREQUAL "LLVM_IR" OR
   (VELRI_SLANG_CPU_MODE STREQUAL "AUTO" AND CMAKE_C_COMPILER_ID STREQUAL "Clang"))
    # Probe whether this Clang parses LLVM IR emitted by this `slangc` version
    # and targets the active toolchain architecture cleanly.
    set(_probe_dir "${VELRI_SLANG_GEN_DIR}/probe")
    file(MAKE_DIRECTORY "${_probe_dir}")
    file(WRITE "${_probe_dir}/probe.slang"
        "[DllExport] export __extern_cpp float velri_probe(float a, float b) { return a * b + 1.0f; }\n")
    execute_process(
        COMMAND "${VELRI_SLANGC}" "${_probe_dir}/probe.slang"
            -target llvm-ir -O0 -fp-mode fast -o "${_probe_dir}/probe.ll"
        RESULT_VARIABLE _probe_slang_rc
        OUTPUT_QUIET ERROR_QUIET
    )
    set(_probe_clang_rc 1)
    if(_probe_slang_rc EQUAL 0)
        execute_process(
            COMMAND "${CMAKE_COMMAND}"
                "-DINPUT_LL=${_probe_dir}/probe.ll"
                "-DOUTPUT_LL=${_probe_dir}/probe_clean.ll"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/strip_ll_target.cmake"
            RESULT_VARIABLE _probe_strip_rc
        )
        if(_probe_strip_rc EQUAL 0)
            if(VELRI_MSVC_FRONTEND)
                set(_probe_cmd "${CMAKE_C_COMPILER}" ${_velri_clang_target_flags}
                    /O2 /fp:fast -Wno-override-module -Qunused-arguments
                    /c "${_probe_dir}/probe_clean.ll" "/Fo${_probe_dir}/probe${_velri_obj_ext}")
            else()
                set(_probe_cmd "${CMAKE_C_COMPILER}" ${_velri_clang_target_flags}
                    -O3 -ffast-math -fno-math-errno -Wno-override-module -Qunused-arguments
                    -c "${_probe_dir}/probe_clean.ll" -o "${_probe_dir}/probe${_velri_obj_ext}")
            endif()
            execute_process(
                COMMAND ${_probe_cmd}
                RESULT_VARIABLE _probe_clang_rc
                OUTPUT_QUIET ERROR_VARIABLE _probe_clang_err
            )
        endif()
    endif()
    if(_probe_clang_rc EQUAL 0)
        set(_velri_use_llvm_ir ON)
    elseif(VELRI_SLANG_CPU_MODE STREQUAL "LLVM_IR")
        message(FATAL_ERROR
            "VELRI_SLANG_CPU_MODE=LLVM_IR failed probe with ${CMAKE_C_COMPILER}:\n${_probe_clang_err}\n"
            "Ensure Clang's LLVM major version is >= slangc's bundled LLVM, or use AUTO/OBJECT.")
    endif()
endif()

if(_velri_use_llvm_ir)
    message(STATUS "Velri-lib Slang CPU AOT pipeline: target-neutral LLVM IR -> ${CMAKE_C_COMPILER}")
else()
    # Fallback (`OBJECT` mode): resolve target triple for `slangc -target host-object-code`.
    set(VELRI_SLANG_TARGET_TRIPLE "" CACHE STRING
        "Explicit LLVM target triple for slangc OBJECT mode (auto-detected when empty)")
    set(_velri_triple "${VELRI_SLANG_TARGET_TRIPLE}")
    if(NOT _velri_triple)
        if(WIN32)
            if(VELRI_TARGET_AARCH64)
                set(_velri_arch "aarch64")
            else()
                set(_velri_arch "x86_64")
            endif()
            if(VELRI_WINDOWS_MSVC_ABI)
                set(_velri_triple "${_velri_arch}-pc-windows-msvc")
            else()
                set(_velri_triple "${_velri_arch}-w64-windows-gnu")
            endif()
        elseif(CMAKE_C_COMPILER_TARGET)
            set(_velri_triple "${CMAKE_C_COMPILER_TARGET}")
        else()
            execute_process(
                COMMAND "${CMAKE_C_COMPILER}" -dumpmachine
                OUTPUT_VARIABLE _velri_dumpmachine
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
                RESULT_VARIABLE _velri_dm_rc
            )
            if(_velri_dm_rc EQUAL 0 AND _velri_dumpmachine)
                set(_velri_triple "${_velri_dumpmachine}")
            elseif(VELRI_TARGET_AARCH64)
                set(_velri_triple "aarch64-linux-gnu")
            else()
                set(_velri_triple "x86_64-linux-gnu")
            endif()
        endif()
    endif()
    if(_velri_triple MATCHES "^([^-]+)-.*mingw32")
        set(_velri_triple "${CMAKE_MATCH_1}-w64-windows-gnu")
    endif()
    message(STATUS "Velri-lib Slang CPU AOT pipeline: slangc host-object-code (${_velri_triple})")
endif()

# ============================================================================
# 1. Slang AOT CPU Backend — VelriLib::cpu (ONE source compiled per ISA profile)
# ============================================================================
set(VELRI_SLANG_CPU_ENTRY "${VELRI_SLANG_DIR}/entry_cpu.slang")
set(VELRI_SLANG_CPU_OBJS)

function(velri_add_slang_cpu_variant isa mr nr_f32 nr_f64 llvm_features clang_isa_flags extra_defs)
    set(_obj "${VELRI_SLANG_GEN_DIR}/slang_cpu_${isa}${_velri_obj_ext}")

    if(_velri_use_llvm_ir)
        set(_raw_ll "${VELRI_SLANG_GEN_DIR}/slang_cpu_${isa}.raw.ll")
        set(_clean_ll "${VELRI_SLANG_GEN_DIR}/slang_cpu_${isa}.ll")
        set(_slang_cmd "${VELRI_SLANGC}" "${VELRI_SLANG_CPU_ENTRY}"
            -I "${VELRI_SLANG_DIR}/common" -I "${VELRI_SLANG_DIR}/ops"
            -target llvm-ir -O0 -fp-mode fast
            "-DVELRI_ISA=${isa}" "-DVELRI_MR=${mr}"
            "-DVELRI_NR_F32=${nr_f32}" "-DVELRI_NR_F64=${nr_f64}")
        foreach(_def IN LISTS extra_defs)
            list(APPEND _slang_cmd "-D${_def}")
        endforeach()
        list(APPEND _slang_cmd -o "${_raw_ll}")

        if(VELRI_MSVC_FRONTEND)
            set(_cc_cmd "${CMAKE_C_COMPILER}" ${_velri_clang_target_flags} ${clang_isa_flags}
                /O2 /fp:fast -Wno-override-module -Qunused-arguments
                /c "${_clean_ll}" "/Fo${_obj}")
        else()
            set(_cc_cmd "${CMAKE_C_COMPILER}" ${_velri_clang_target_flags} ${clang_isa_flags}
                -O3 -ffast-math -fno-math-errno -Wno-override-module -Qunused-arguments
                -c "${_clean_ll}" -o "${_obj}")
        endif()

        add_custom_command(
            OUTPUT "${_obj}"
            COMMAND ${_slang_cmd}
            COMMAND "${CMAKE_COMMAND}"
                "-DINPUT_LL=${_raw_ll}"
                "-DOUTPUT_LL=${_clean_ll}"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/strip_ll_target.cmake"
            COMMAND ${_cc_cmd}
            DEPENDS "${VELRI_SLANG_CPU_ENTRY}" ${VELRI_SLANG_COMMON_DEPS}
                    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/strip_ll_target.cmake"
            COMMENT "Slang AOT CPU (${isa}, LLVM IR -> Clang): entry_cpu.slang -> slang_cpu_${isa}${_velri_obj_ext}"
            VERBATIM
        )
    else()
        set(_cmd "${VELRI_SLANGC}" "${VELRI_SLANG_CPU_ENTRY}"
            -I "${VELRI_SLANG_DIR}/common" -I "${VELRI_SLANG_DIR}/ops"
            -target host-object-code -O3 -fp-mode fast
            -llvm-target-triple "${_velri_triple}"
            "-DVELRI_ISA=${isa}" "-DVELRI_MR=${mr}"
            "-DVELRI_NR_F32=${nr_f32}" "-DVELRI_NR_F64=${nr_f64}")
        foreach(_def IN LISTS extra_defs)
            list(APPEND _cmd "-D${_def}")
        endforeach()
        if(llvm_features)
            list(APPEND _cmd -llvm-features "${llvm_features}")
        endif()
        list(APPEND _cmd -o "${_obj}")

        add_custom_command(
            OUTPUT "${_obj}"
            COMMAND ${_cmd}
            DEPENDS "${VELRI_SLANG_CPU_ENTRY}" ${VELRI_SLANG_COMMON_DEPS}
            COMMENT "Slang AOT CPU (${isa}, ${_velri_triple}): entry_cpu.slang -> slang_cpu_${isa}${_velri_obj_ext}"
            VERBATIM
        )
    endif()

    set_source_files_properties("${_obj}" PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
    set(VELRI_SLANG_CPU_OBJS ${VELRI_SLANG_CPU_OBJS} "${_obj}" PARENT_SCOPE)
endfunction()

# Base profile works on ANY CPU architecture and ANY OS without arch hardcoding.
# Software FP16 on base eliminates external __extendhfsf2/__truncsfhf2 libcalls on
# targets without mandatory hardware FP16 (x86-64 SSE2, RISC-V rv64gc, Windows MSVC).
set(_velri_base_mr 3)
set(_velri_base_defs "VELRI_SOFT_F16=1")
if(VELRI_TARGET_AARCH64)
    # ARMv8.0 guarantees 32 NEON registers + native F16 fcvt.
    set(_velri_base_mr 4)
    set(_velri_base_defs "")
endif()
velri_add_slang_cpu_variant(base ${_velri_base_mr} 16 8 "" "${VELRI_BASE_FLAGS}" "${_velri_base_defs}")

# Optional x86-64 fat-binary multiversioning (runtime CPUID dispatch across SSE2/AVX2/AVX-512).
if(VELRI_X86_VARIANTS)
    if(VELRI_MSVC_FRONTEND)
        set(_avx2_cc_flags /arch:AVX2)
        set(_avx512_cc_flags /arch:AVX512)
    else()
        set(_avx2_cc_flags -mavx2 -mfma -mf16c)
        set(_avx512_cc_flags -mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx512vnni -mfma -mf16c)
    endif()
    velri_add_slang_cpu_variant(avx2 3 32 16
        "+avx2,+fma,+f16c"
        "${_avx2_cc_flags}" "")
    velri_add_slang_cpu_variant(avx512 6 64 32
        "+avx512f,+avx512bw,+avx512dq,+avx512vl,+avx512vnni,+fma,+f16c"
        "${_avx512_cc_flags}" "")
endif()

add_library(velri_cpu STATIC
    src/backend/cpu/slang_cpu_dispatch.c
    ${VELRI_SLANG_CPU_OBJS}
)
add_library(VelriLib::cpu ALIAS velri_cpu)
add_library(VelriLib::slang_cpu ALIAS velri_cpu)
set_target_properties(velri_cpu PROPERTIES
    EXPORT_NAME cpu
    POSITION_INDEPENDENT_CODE ON
)
target_include_directories(velri_cpu PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_compile_definitions(velri_cpu PRIVATE VELRI_X86_VARIANTS=${VELRI_X86_VARIANTS})
target_compile_options(velri_cpu PRIVATE ${VELRI_BASE_FLAGS})
velri_strict_target(velri_cpu)
target_link_libraries(velri_cpu PUBLIC velri_platform)
if(UNIX)
    target_link_libraries(velri_cpu PRIVATE m)
endif()

if(VELRI_LIB_BUILD_TESTS OR VELRI_ENABLE_VULKAN)
    add_library(velri_typed_reference STATIC tests/cpu/typed/reference.c)
    target_include_directories(velri_typed_reference PUBLIC tests/cpu/typed
        PRIVATE include)
    if(VELRI_MSVC_FRONTEND)
        target_compile_options(velri_typed_reference PRIVATE ${VELRI_BASE_FLAGS})
    else()
        target_compile_options(velri_typed_reference PRIVATE ${VELRI_BASE_FLAGS} -ffp-contract=off)
    endif()
    velri_strict_target(velri_typed_reference)
    target_link_libraries(velri_typed_reference PRIVATE velri_platform)
    if(UNIX)
        target_link_libraries(velri_typed_reference PUBLIC m)
    endif()
endif()

if(VELRI_LIB_BUILD_TESTS)
    add_executable(bench_slang_cpu tools/bench_slang_cpu.c)
    target_compile_definitions(bench_slang_cpu PRIVATE VELRI_X86_VARIANTS=${VELRI_X86_VARIANTS})
    target_compile_options(bench_slang_cpu PRIVATE ${VELRI_BASE_FLAGS})
    velri_strict_target(bench_slang_cpu)
    target_link_libraries(bench_slang_cpu PRIVATE velri_cpu velri_typed_reference)
    add_test(NAME slang_cpu_verify COMMAND bench_slang_cpu --verify-only)
endif()

# ============================================================================
# 2. Slang AOT Vulkan Backend — VelriLib::vulkan (SPIR-V 1.6 + spirv-val)
# ============================================================================
if(VELRI_ENABLE_VULKAN)
    # Check target sysroot first (e.g. Android NDK), then arch-independent host Vulkan-Headers.
    find_path(VELRI_VULKAN_INCLUDE_DIR vulkan/vulkan.h
        HINTS "$ENV{HOME}/.local/opt/vulkan-headers/include" "$ENV{VULKAN_SDK}/Include"
              /usr/include /usr/local/include)
    if(NOT VELRI_VULKAN_INCLUDE_DIR)
        find_path(VELRI_VULKAN_INCLUDE_DIR vulkan/vulkan.h
            HINTS "$ENV{HOME}/.local/opt/vulkan-headers/include" "$ENV{VULKAN_SDK}/Include"
                  /usr/include /usr/local/include
            NO_CMAKE_FIND_ROOT_PATH)
    endif()
    if(NOT VELRI_VULKAN_INCLUDE_DIR)
        message(FATAL_ERROR "VELRI_ENABLE_VULKAN requires Vulkan-Headers (vulkan/vulkan.h)")
    endif()
    find_program(VELRI_SPIRV_VAL NAMES spirv-val NO_CMAKE_FIND_ROOT_PATH)

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
    set_target_properties(velri_vulkan PROPERTIES
        EXPORT_NAME vulkan
        POSITION_INDEPENDENT_CODE ON
    )
    target_include_directories(velri_vulkan PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )
    target_include_directories(velri_vulkan PRIVATE
        "${VELRI_SLANG_GEN_DIR}"
        "${VELRI_VULKAN_INCLUDE_DIR}"
    )
    target_compile_options(velri_vulkan PRIVATE ${VELRI_BASE_FLAGS})
    velri_strict_target(velri_vulkan)
    target_link_libraries(velri_vulkan PUBLIC velri_cpu)

    if(VELRI_LIB_BUILD_TESTS)
        add_executable(bench_vulkan tools/bench_vulkan.c)
        target_compile_options(bench_vulkan PRIVATE ${VELRI_BASE_FLAGS})
        velri_strict_target(bench_vulkan)
        target_link_libraries(bench_vulkan PRIVATE velri_vulkan velri_cpu velri_typed_reference)
        add_test(NAME vulkan_verify COMMAND bench_vulkan --verify-only)
    endif()
endif()
