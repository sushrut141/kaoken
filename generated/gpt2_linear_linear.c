// START_METHODS
#include <arm_neon.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * Carries out a linear transformation on the input.
 * For an input I and  linear transformation matrix L, it computes
 * y = I * L.transpose
 * Shape of I -> [s, in_features]
 * Shape of L -> [out_features, in_features]
 *
 * y = I * L.transpose -> [s, in_features] * [in_features, out_features] -> [s,
 * out_features]
 */
void gpt2_linear_linear(float32_t **input, float32_t **output) {
    output[0][0] = (input[0][0] * 0.1) + (input[0][1] * 0.2) +
                   (input[0][2] * 0.3) + (input[0][3] * 0.4);
    output[0][1] = (input[0][0] * 0.1) + (input[0][1] * 0.2) +
                   (input[0][2] * 0.3) + (input[0][3] * 0.4);
    output[0][2] = (input[0][0] * 0.1) + (input[0][1] * 0.2) +
                   (input[0][2] * 0.3) + (input[0][3] * 0.4);
    output[0][3] = (input[0][0] * 0.1) + (input[0][1] * 0.2) +
                   (input[0][2] * 0.3) + (input[0][3] * 0.4);
    output[0][4] = (input[0][0] * 0.1) + (input[0][1] * 0.2) +
                   (input[0][2] * 0.3) + (input[0][3] * 0.4);
    output[0][5] = (input[0][0] * 0.1) + (input[0][1] * 0.2) +
                   (input[0][2] * 0.3) + (input[0][3] * 0.4);
    output[0][6] = (input[0][0] * 0.1) + (input[0][1] * 0.2) +
                   (input[0][2] * 0.3) + (input[0][3] * 0.4);
    output[0][7] = (input[0][0] * 0.1) + (input[0][1] * 0.2) +
                   (input[0][2] * 0.3) + (input[0][3] * 0.4);
}

// START_TEST
int linear_test(int argc, char **argv) {
    float32_t **input = (float32_t **)malloc(sizeof(float32_t *) * 1);
    float32_t **output = (float32_t **)malloc(sizeof(float32_t *) * 4);

    for (int i = 0; i < 1; i += 1) {
        input[i] = (float32_t *)malloc(sizeof(float32_t) * 4);
        // Set input as [1.0, 2.0, 3.0, 4.0 ... ]
        for (int j = 0; j < 4; j += 1) {
            input[i][j] = (j + 1) * 1.0;
        }
    }
    for (int i = 0; i < 4; i += 1) {
        output[i] = (float32_t *)malloc(sizeof(float32_t) * 8);
        memset(output[i], 0.0, sizeof(float32_t) * 8);
    }

    gpt2_linear_linear(input, output);

    for (int i = 0; i < 1; i += 1) {
        printf("Sequence %d Input\n", i);
        for (int j = 0; j < 4; j += 1) {
            printf("idx %d -> %f\n", j, input[i][j]);
        }

        printf("Sequence %d Output\n", i);
        for (int j = 0; j < 8; j += 1) {
            printf("idx %d -> %f\n", j, output[i][j]);
        }
    }

    return 0;
}
// END_TEST
