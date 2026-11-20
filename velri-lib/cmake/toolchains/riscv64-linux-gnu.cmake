# Debian/Ubuntu cross tools: gcc-14-riscv64-linux-gnu,
# libc6-dev-riscv64-cross and qemu-user.
# The baseline does not require the optional RISC-V vector extension.
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)
if(NOT DEFINED CMAKE_C_COMPILER)
    set(CMAKE_C_COMPILER riscv64-linux-gnu-gcc-14)
endif()
set(CMAKE_C_FLAGS_INIT "-march=rv64gc -mabi=lp64d")
set(VELRI_QEMU_RISCV64 qemu-riscv64 CACHE FILEPATH "RISC-V user-mode emulator")
set(VELRI_RISCV_RUNTIME /usr/riscv64-linux-gnu CACHE PATH "RISC-V runtime for QEMU")
set(CMAKE_CROSSCOMPILING_EMULATOR "${VELRI_QEMU_RISCV64};-L;${VELRI_RISCV_RUNTIME}")
