# Standalone OS Platform Library (`VelriLib::platform`).
# Isolates ALL OS-specific syscalls into `os_posix.c` (POSIX/Linux/Android/BSD/macOS)
# or `os_win32.c` (Windows x86-64/ARM64), selected cleanly at configure time.

if(WIN32)
    set(VELRI_PLATFORM_OS_SRC src/platform/os_win32.c)
else()
    find_package(Threads REQUIRED)
    set(VELRI_PLATFORM_OS_SRC src/platform/os_posix.c)
endif()

add_library(velri_platform STATIC
    ${VELRI_PLATFORM_OS_SRC}
    src/platform/cpu_caps.c
    src/platform/pool.c
)
add_library(VelriLib::platform ALIAS velri_platform)
set_target_properties(velri_platform PROPERTIES
    EXPORT_NAME platform
    POSITION_INDEPENDENT_CODE ON
)
target_include_directories(velri_platform
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/platform>
)
target_compile_definitions(velri_platform PRIVATE VELRI_X86_VARIANTS=${VELRI_X86_VARIANTS})
target_compile_options(velri_platform PRIVATE ${VELRI_BASE_FLAGS})
velri_strict_target(velri_platform)
if(NOT WIN32)
    target_link_libraries(velri_platform PUBLIC Threads::Threads PRIVATE ${CMAKE_DL_LIBS})
endif()
