# Velri-lib Core: Storage/TensorView, Op Registry, Graph Builder, Memory Planner,
# Backend Registry/Contract, Runtime Execution Slots, and Built-in Backend Adapters.

set(VELRI_CORE_SOURCES
    src/core/tensor.c
    src/core/runtime_ctx.c
    src/ops/registry.c
    src/graph/graph.c
    src/plan/planner.c
    src/runtime/executor.c
    src/backend/cpu/cpu_backend.c
)
if(VELRI_ENABLE_VULKAN)
    list(APPEND VELRI_CORE_SOURCES src/backend/vulkan/vk_backend_adapter.c)
endif()

add_library(velri_core STATIC ${VELRI_CORE_SOURCES})
add_library(VelriLib::core ALIAS velri_core)
set_target_properties(velri_core PROPERTIES
    EXPORT_NAME core
    POSITION_INDEPENDENT_CODE ON
)
target_include_directories(velri_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src/core"
)
target_compile_options(velri_core PRIVATE ${VELRI_BASE_FLAGS})
velri_strict_target(velri_core)
target_link_libraries(velri_core PUBLIC velri_cpu)
if(VELRI_ENABLE_VULKAN)
    target_compile_definitions(velri_core PRIVATE VELRI_CORE_HAS_VULKAN=1)
    target_link_libraries(velri_core PRIVATE velri_vulkan)
endif()
