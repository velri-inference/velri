#include <velri_lib/kernels.h>

/* Check the consumer header, exported CMake target (VelriLib::cpu) and Slang AOT Linear interface.
 * Numerical edge cases belong to Velri-lib's independent correctness suite. */
int main(void) {
    const double input[] = {1.0, 2.0, 3.0};
    const double weights[] = {4.0, 5.0, 6.0, -1.0, 0.0, 1.0};
    const double bias[] = {0.5, -0.5};
    double output[2] = {0.0, 0.0};
    const velri_linear_desc desc = velri_desc_f64();
    if (velri_slang_cpu_linear(&desc, 1, 2, 3, input, 3, weights, 3, bias, nullptr, nullptr,
                               output, 2, VELRI_CPU_AUTO, nullptr, 1) != VELRI_OK) {
        return 1;
    }
    return output[0] == 32.5 && output[1] == 1.5 ? 0 : 2;
}
