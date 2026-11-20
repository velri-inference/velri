#include <velri_lib/kernels.h>

/* Check the consumer header, exported CMake target and typed Linear interface.
 * Numerical edge cases belong to Velri-lib's independent correctness suite. */
int main(void) {
    const double input[] = {1.0, 2.0, 3.0};
    const double weights[] = {4.0, 5.0, 6.0, -1.0, 0.0, 1.0};
    const double bias[] = {0.5, -0.5};
    double output[2];
    velri_linear_typed_plan plan;
    const velri_linear_desc desc = velri_desc_f64();
    if (velri_linear_typed_query(&desc, 2, 3, false, VELRI_CPU_AUTO, &plan) != VELRI_OK)
        return 1;
    const velri_linear_run_args args = {
        .weights = weights, .weights_capacity = 6, .w_stride = 3,
        .x = input, .x_count = 3, .rows = 1, .x_stride = 3,
        .bias = bias, .bias_count = 2,
        .y = output, .y_count = 2, .y_stride = 2,
    };
    if (velri_linear_typed_run(&plan, &args) != VELRI_OK) return 2;
    return output[0] == 32.5 && output[1] == 1.5 ? 0 : 3;
}
