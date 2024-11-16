// START_METHODS
#include <arm_neon.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Matrix of size [4, 4 * 4]
// used for storing intermediate values in c_fc conv1d step
float32_t **gpt2_mlp_c_fc_temp = NULL;
// Matrix of size [4, 4 * 4]
// used for storing intermediate values in gelu activation step
float32_t **gpt2_mlp_gelu_temp = NULL;

// Initializes the weights used in the attention layer.
// Must be called before calling gpt2_mlp_attention.
void gpt2_mlp_initialize_heap_memory() {
    gpt2_mlp_c_fc_temp = (float32_t **)malloc(sizeof(float32_t *) * 4);
    gpt2_mlp_gelu_temp = (float32_t **)malloc(sizeof(float32_t *) * 4);

    gpt2_mlp_c_fc_temp[0] = (float32_t *)malloc(sizeof(float32_t) * 4 * 4);
    gpt2_mlp_gelu_temp[0] = (float32_t *)malloc(sizeof(float32_t) * 4 * 4);
    gpt2_mlp_c_fc_temp[1] = (float32_t *)malloc(sizeof(float32_t) * 4 * 4);
    gpt2_mlp_gelu_temp[1] = (float32_t *)malloc(sizeof(float32_t) * 4 * 4);
    gpt2_mlp_c_fc_temp[2] = (float32_t *)malloc(sizeof(float32_t) * 4 * 4);
    gpt2_mlp_gelu_temp[2] = (float32_t *)malloc(sizeof(float32_t) * 4 * 4);
    gpt2_mlp_c_fc_temp[3] = (float32_t *)malloc(sizeof(float32_t) * 4 * 4);
    gpt2_mlp_gelu_temp[3] = (float32_t *)malloc(sizeof(float32_t) * 4 * 4);
}

// 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715
// * torch.pow(input, 3.0))))
void gpt2_mlp_new_gelu_activation(float32_t *input, float32_t *output) {
    output[0] = input[0] + (0.044715 * pow(input[0], 3));
    output[0] *= sqrt(2.0 / 3.14159);
    output[0] = 1.0 + tanh(output[0]);
    output[0] = 0.5 * input[0] * output[0];
    output[1] = input[1] + (0.044715 * pow(input[1], 3));
    output[1] *= sqrt(2.0 / 3.14159);
    output[1] = 1.0 + tanh(output[1]);
    output[1] = 0.5 * input[1] * output[1];
    output[2] = input[2] + (0.044715 * pow(input[2], 3));
    output[2] *= sqrt(2.0 / 3.14159);
    output[2] = 1.0 + tanh(output[2]);
    output[2] = 0.5 * input[2] * output[2];
    output[3] = input[3] + (0.044715 * pow(input[3], 3));
    output[3] *= sqrt(2.0 / 3.14159);
    output[3] = 1.0 + tanh(output[3]);
    output[3] = 0.5 * input[3] * output[3];
    output[4] = input[4] + (0.044715 * pow(input[4], 3));
    output[4] *= sqrt(2.0 / 3.14159);
    output[4] = 1.0 + tanh(output[4]);
    output[4] = 0.5 * input[4] * output[4];
    output[5] = input[5] + (0.044715 * pow(input[5], 3));
    output[5] *= sqrt(2.0 / 3.14159);
    output[5] = 1.0 + tanh(output[5]);
    output[5] = 0.5 * input[5] * output[5];
    output[6] = input[6] + (0.044715 * pow(input[6], 3));
    output[6] *= sqrt(2.0 / 3.14159);
    output[6] = 1.0 + tanh(output[6]);
    output[6] = 0.5 * input[6] * output[6];
    output[7] = input[7] + (0.044715 * pow(input[7], 3));
    output[7] *= sqrt(2.0 / 3.14159);
    output[7] = 1.0 + tanh(output[7]);
    output[7] = 0.5 * input[7] * output[7];
    output[8] = input[8] + (0.044715 * pow(input[8], 3));
    output[8] *= sqrt(2.0 / 3.14159);
    output[8] = 1.0 + tanh(output[8]);
    output[8] = 0.5 * input[8] * output[8];
    output[9] = input[9] + (0.044715 * pow(input[9], 3));
    output[9] *= sqrt(2.0 / 3.14159);
    output[9] = 1.0 + tanh(output[9]);
    output[9] = 0.5 * input[9] * output[9];
    output[10] = input[10] + (0.044715 * pow(input[10], 3));
    output[10] *= sqrt(2.0 / 3.14159);
    output[10] = 1.0 + tanh(output[10]);
    output[10] = 0.5 * input[10] * output[10];
    output[11] = input[11] + (0.044715 * pow(input[11], 3));
    output[11] *= sqrt(2.0 / 3.14159);
    output[11] = 1.0 + tanh(output[11]);
    output[11] = 0.5 * input[11] * output[11];
    output[12] = input[12] + (0.044715 * pow(input[12], 3));
    output[12] *= sqrt(2.0 / 3.14159);
    output[12] = 1.0 + tanh(output[12]);
    output[12] = 0.5 * input[12] * output[12];
    output[13] = input[13] + (0.044715 * pow(input[13], 3));
    output[13] *= sqrt(2.0 / 3.14159);
    output[13] = 1.0 + tanh(output[13]);
    output[13] = 0.5 * input[13] * output[13];
    output[14] = input[14] + (0.044715 * pow(input[14], 3));
    output[14] *= sqrt(2.0 / 3.14159);
    output[14] = 1.0 + tanh(output[14]);
    output[14] = 0.5 * input[14] * output[14];
    output[15] = input[15] + (0.044715 * pow(input[15], 3));
    output[15] *= sqrt(2.0 / 3.14159);
    output[15] = 1.0 + tanh(output[15]);
    output[15] = 0.5 * input[15] * output[15];
}

/**
 * Carries out transformations on the inout per the MLP block in transformer
 * which is composed of two conv1d transformations and an activation.
 */
void gpt2_mlp_mlp(float32_t **input, float32_t **output) {
    // Conv1D c_fc
    // Matrix with shape [4, 4 * 4]
    gpt2_mlp_c_fc_temp[0][0] = (input[0][0] * 0.0) + (input[0][1] * 0.0) +
                               (input[0][2] * 0.0) + (input[0][3] * 0.0);
    gpt2_mlp_c_fc_temp[0][1] = (input[0][0] * 0.001) + (input[0][1] * 0.001) +
                               (input[0][2] * 0.001) + (input[0][3] * 0.001);
    gpt2_mlp_c_fc_temp[0][2] = (input[0][0] * 0.002) + (input[0][1] * 0.002) +
                               (input[0][2] * 0.002) + (input[0][3] * 0.002);
    gpt2_mlp_c_fc_temp[0][3] = (input[0][0] * 0.003) + (input[0][1] * 0.003) +
                               (input[0][2] * 0.003) + (input[0][3] * 0.003);
    gpt2_mlp_c_fc_temp[0][4] = (input[0][0] * 0.004) + (input[0][1] * 0.004) +
                               (input[0][2] * 0.004) + (input[0][3] * 0.004);
    gpt2_mlp_c_fc_temp[0][5] = (input[0][0] * 0.005) + (input[0][1] * 0.005) +
                               (input[0][2] * 0.005) + (input[0][3] * 0.005);
    gpt2_mlp_c_fc_temp[0][6] = (input[0][0] * 0.006) + (input[0][1] * 0.006) +
                               (input[0][2] * 0.006) + (input[0][3] * 0.006);
    gpt2_mlp_c_fc_temp[0][7] = (input[0][0] * 0.007) + (input[0][1] * 0.007) +
                               (input[0][2] * 0.007) + (input[0][3] * 0.007);
    gpt2_mlp_c_fc_temp[0][8] = (input[0][0] * 0.008) + (input[0][1] * 0.008) +
                               (input[0][2] * 0.008) + (input[0][3] * 0.008);
    gpt2_mlp_c_fc_temp[0][9] = (input[0][0] * 0.009000000000000001) +
                               (input[0][1] * 0.009000000000000001) +
                               (input[0][2] * 0.009000000000000001) +
                               (input[0][3] * 0.009000000000000001);
    gpt2_mlp_c_fc_temp[0][10] = (input[0][0] * 0.01) + (input[0][1] * 0.01) +
                                (input[0][2] * 0.01) + (input[0][3] * 0.01);
    gpt2_mlp_c_fc_temp[0][11] = (input[0][0] * 0.011) + (input[0][1] * 0.011) +
                                (input[0][2] * 0.011) + (input[0][3] * 0.011);
    gpt2_mlp_c_fc_temp[0][12] = (input[0][0] * 0.012) + (input[0][1] * 0.012) +
                                (input[0][2] * 0.012) + (input[0][3] * 0.012);
    gpt2_mlp_c_fc_temp[0][13] = (input[0][0] * 0.013000000000000001) +
                                (input[0][1] * 0.013000000000000001) +
                                (input[0][2] * 0.013000000000000001) +
                                (input[0][3] * 0.013000000000000001);
    gpt2_mlp_c_fc_temp[0][14] = (input[0][0] * 0.014) + (input[0][1] * 0.014) +
                                (input[0][2] * 0.014) + (input[0][3] * 0.014);
    gpt2_mlp_c_fc_temp[0][15] = (input[0][0] * 0.015) + (input[0][1] * 0.015) +
                                (input[0][2] * 0.015) + (input[0][3] * 0.015);
    gpt2_mlp_c_fc_temp[1][0] = (input[1][0] * 0.0) + (input[1][1] * 0.0) +
                               (input[1][2] * 0.0) + (input[1][3] * 0.0);
    gpt2_mlp_c_fc_temp[1][1] = (input[1][0] * 0.001) + (input[1][1] * 0.001) +
                               (input[1][2] * 0.001) + (input[1][3] * 0.001);
    gpt2_mlp_c_fc_temp[1][2] = (input[1][0] * 0.002) + (input[1][1] * 0.002) +
                               (input[1][2] * 0.002) + (input[1][3] * 0.002);
    gpt2_mlp_c_fc_temp[1][3] = (input[1][0] * 0.003) + (input[1][1] * 0.003) +
                               (input[1][2] * 0.003) + (input[1][3] * 0.003);
    gpt2_mlp_c_fc_temp[1][4] = (input[1][0] * 0.004) + (input[1][1] * 0.004) +
                               (input[1][2] * 0.004) + (input[1][3] * 0.004);
    gpt2_mlp_c_fc_temp[1][5] = (input[1][0] * 0.005) + (input[1][1] * 0.005) +
                               (input[1][2] * 0.005) + (input[1][3] * 0.005);
    gpt2_mlp_c_fc_temp[1][6] = (input[1][0] * 0.006) + (input[1][1] * 0.006) +
                               (input[1][2] * 0.006) + (input[1][3] * 0.006);
    gpt2_mlp_c_fc_temp[1][7] = (input[1][0] * 0.007) + (input[1][1] * 0.007) +
                               (input[1][2] * 0.007) + (input[1][3] * 0.007);
    gpt2_mlp_c_fc_temp[1][8] = (input[1][0] * 0.008) + (input[1][1] * 0.008) +
                               (input[1][2] * 0.008) + (input[1][3] * 0.008);
    gpt2_mlp_c_fc_temp[1][9] = (input[1][0] * 0.009000000000000001) +
                               (input[1][1] * 0.009000000000000001) +
                               (input[1][2] * 0.009000000000000001) +
                               (input[1][3] * 0.009000000000000001);
    gpt2_mlp_c_fc_temp[1][10] = (input[1][0] * 0.01) + (input[1][1] * 0.01) +
                                (input[1][2] * 0.01) + (input[1][3] * 0.01);
    gpt2_mlp_c_fc_temp[1][11] = (input[1][0] * 0.011) + (input[1][1] * 0.011) +
                                (input[1][2] * 0.011) + (input[1][3] * 0.011);
    gpt2_mlp_c_fc_temp[1][12] = (input[1][0] * 0.012) + (input[1][1] * 0.012) +
                                (input[1][2] * 0.012) + (input[1][3] * 0.012);
    gpt2_mlp_c_fc_temp[1][13] = (input[1][0] * 0.013000000000000001) +
                                (input[1][1] * 0.013000000000000001) +
                                (input[1][2] * 0.013000000000000001) +
                                (input[1][3] * 0.013000000000000001);
    gpt2_mlp_c_fc_temp[1][14] = (input[1][0] * 0.014) + (input[1][1] * 0.014) +
                                (input[1][2] * 0.014) + (input[1][3] * 0.014);
    gpt2_mlp_c_fc_temp[1][15] = (input[1][0] * 0.015) + (input[1][1] * 0.015) +
                                (input[1][2] * 0.015) + (input[1][3] * 0.015);
    gpt2_mlp_c_fc_temp[2][0] = (input[2][0] * 0.0) + (input[2][1] * 0.0) +
                               (input[2][2] * 0.0) + (input[2][3] * 0.0);
    gpt2_mlp_c_fc_temp[2][1] = (input[2][0] * 0.001) + (input[2][1] * 0.001) +
                               (input[2][2] * 0.001) + (input[2][3] * 0.001);
    gpt2_mlp_c_fc_temp[2][2] = (input[2][0] * 0.002) + (input[2][1] * 0.002) +
                               (input[2][2] * 0.002) + (input[2][3] * 0.002);
    gpt2_mlp_c_fc_temp[2][3] = (input[2][0] * 0.003) + (input[2][1] * 0.003) +
                               (input[2][2] * 0.003) + (input[2][3] * 0.003);
    gpt2_mlp_c_fc_temp[2][4] = (input[2][0] * 0.004) + (input[2][1] * 0.004) +
                               (input[2][2] * 0.004) + (input[2][3] * 0.004);
    gpt2_mlp_c_fc_temp[2][5] = (input[2][0] * 0.005) + (input[2][1] * 0.005) +
                               (input[2][2] * 0.005) + (input[2][3] * 0.005);
    gpt2_mlp_c_fc_temp[2][6] = (input[2][0] * 0.006) + (input[2][1] * 0.006) +
                               (input[2][2] * 0.006) + (input[2][3] * 0.006);
    gpt2_mlp_c_fc_temp[2][7] = (input[2][0] * 0.007) + (input[2][1] * 0.007) +
                               (input[2][2] * 0.007) + (input[2][3] * 0.007);
    gpt2_mlp_c_fc_temp[2][8] = (input[2][0] * 0.008) + (input[2][1] * 0.008) +
                               (input[2][2] * 0.008) + (input[2][3] * 0.008);
    gpt2_mlp_c_fc_temp[2][9] = (input[2][0] * 0.009000000000000001) +
                               (input[2][1] * 0.009000000000000001) +
                               (input[2][2] * 0.009000000000000001) +
                               (input[2][3] * 0.009000000000000001);
    gpt2_mlp_c_fc_temp[2][10] = (input[2][0] * 0.01) + (input[2][1] * 0.01) +
                                (input[2][2] * 0.01) + (input[2][3] * 0.01);
    gpt2_mlp_c_fc_temp[2][11] = (input[2][0] * 0.011) + (input[2][1] * 0.011) +
                                (input[2][2] * 0.011) + (input[2][3] * 0.011);
    gpt2_mlp_c_fc_temp[2][12] = (input[2][0] * 0.012) + (input[2][1] * 0.012) +
                                (input[2][2] * 0.012) + (input[2][3] * 0.012);
    gpt2_mlp_c_fc_temp[2][13] = (input[2][0] * 0.013000000000000001) +
                                (input[2][1] * 0.013000000000000001) +
                                (input[2][2] * 0.013000000000000001) +
                                (input[2][3] * 0.013000000000000001);
    gpt2_mlp_c_fc_temp[2][14] = (input[2][0] * 0.014) + (input[2][1] * 0.014) +
                                (input[2][2] * 0.014) + (input[2][3] * 0.014);
    gpt2_mlp_c_fc_temp[2][15] = (input[2][0] * 0.015) + (input[2][1] * 0.015) +
                                (input[2][2] * 0.015) + (input[2][3] * 0.015);
    gpt2_mlp_c_fc_temp[3][0] = (input[3][0] * 0.0) + (input[3][1] * 0.0) +
                               (input[3][2] * 0.0) + (input[3][3] * 0.0);
    gpt2_mlp_c_fc_temp[3][1] = (input[3][0] * 0.001) + (input[3][1] * 0.001) +
                               (input[3][2] * 0.001) + (input[3][3] * 0.001);
    gpt2_mlp_c_fc_temp[3][2] = (input[3][0] * 0.002) + (input[3][1] * 0.002) +
                               (input[3][2] * 0.002) + (input[3][3] * 0.002);
    gpt2_mlp_c_fc_temp[3][3] = (input[3][0] * 0.003) + (input[3][1] * 0.003) +
                               (input[3][2] * 0.003) + (input[3][3] * 0.003);
    gpt2_mlp_c_fc_temp[3][4] = (input[3][0] * 0.004) + (input[3][1] * 0.004) +
                               (input[3][2] * 0.004) + (input[3][3] * 0.004);
    gpt2_mlp_c_fc_temp[3][5] = (input[3][0] * 0.005) + (input[3][1] * 0.005) +
                               (input[3][2] * 0.005) + (input[3][3] * 0.005);
    gpt2_mlp_c_fc_temp[3][6] = (input[3][0] * 0.006) + (input[3][1] * 0.006) +
                               (input[3][2] * 0.006) + (input[3][3] * 0.006);
    gpt2_mlp_c_fc_temp[3][7] = (input[3][0] * 0.007) + (input[3][1] * 0.007) +
                               (input[3][2] * 0.007) + (input[3][3] * 0.007);
    gpt2_mlp_c_fc_temp[3][8] = (input[3][0] * 0.008) + (input[3][1] * 0.008) +
                               (input[3][2] * 0.008) + (input[3][3] * 0.008);
    gpt2_mlp_c_fc_temp[3][9] = (input[3][0] * 0.009000000000000001) +
                               (input[3][1] * 0.009000000000000001) +
                               (input[3][2] * 0.009000000000000001) +
                               (input[3][3] * 0.009000000000000001);
    gpt2_mlp_c_fc_temp[3][10] = (input[3][0] * 0.01) + (input[3][1] * 0.01) +
                                (input[3][2] * 0.01) + (input[3][3] * 0.01);
    gpt2_mlp_c_fc_temp[3][11] = (input[3][0] * 0.011) + (input[3][1] * 0.011) +
                                (input[3][2] * 0.011) + (input[3][3] * 0.011);
    gpt2_mlp_c_fc_temp[3][12] = (input[3][0] * 0.012) + (input[3][1] * 0.012) +
                                (input[3][2] * 0.012) + (input[3][3] * 0.012);
    gpt2_mlp_c_fc_temp[3][13] = (input[3][0] * 0.013000000000000001) +
                                (input[3][1] * 0.013000000000000001) +
                                (input[3][2] * 0.013000000000000001) +
                                (input[3][3] * 0.013000000000000001);
    gpt2_mlp_c_fc_temp[3][14] = (input[3][0] * 0.014) + (input[3][1] * 0.014) +
                                (input[3][2] * 0.014) + (input[3][3] * 0.014);
    gpt2_mlp_c_fc_temp[3][15] = (input[3][0] * 0.015) + (input[3][1] * 0.015) +
                                (input[3][2] * 0.015) + (input[3][3] * 0.015);

    // Vector of size (4*4
    gpt2_mlp_c_fc_temp[0][0] += 0;
    gpt2_mlp_c_fc_temp[0][1] += 0;
    gpt2_mlp_c_fc_temp[0][2] += 0;
    gpt2_mlp_c_fc_temp[0][3] += 0;
    gpt2_mlp_c_fc_temp[0][4] += 0;
    gpt2_mlp_c_fc_temp[0][5] += 0;
    gpt2_mlp_c_fc_temp[0][6] += 0;
    gpt2_mlp_c_fc_temp[0][7] += 0;
    gpt2_mlp_c_fc_temp[0][8] += 0;
    gpt2_mlp_c_fc_temp[0][9] += 0;
    gpt2_mlp_c_fc_temp[0][10] += 0;
    gpt2_mlp_c_fc_temp[0][11] += 0;
    gpt2_mlp_c_fc_temp[0][12] += 0;
    gpt2_mlp_c_fc_temp[0][13] += 0;
    gpt2_mlp_c_fc_temp[0][14] += 0;
    gpt2_mlp_c_fc_temp[0][15] += 0;
    gpt2_mlp_c_fc_temp[1][0] += 0;
    gpt2_mlp_c_fc_temp[1][1] += 0;
    gpt2_mlp_c_fc_temp[1][2] += 0;
    gpt2_mlp_c_fc_temp[1][3] += 0;
    gpt2_mlp_c_fc_temp[1][4] += 0;
    gpt2_mlp_c_fc_temp[1][5] += 0;
    gpt2_mlp_c_fc_temp[1][6] += 0;
    gpt2_mlp_c_fc_temp[1][7] += 0;
    gpt2_mlp_c_fc_temp[1][8] += 0;
    gpt2_mlp_c_fc_temp[1][9] += 0;
    gpt2_mlp_c_fc_temp[1][10] += 0;
    gpt2_mlp_c_fc_temp[1][11] += 0;
    gpt2_mlp_c_fc_temp[1][12] += 0;
    gpt2_mlp_c_fc_temp[1][13] += 0;
    gpt2_mlp_c_fc_temp[1][14] += 0;
    gpt2_mlp_c_fc_temp[1][15] += 0;
    gpt2_mlp_c_fc_temp[2][0] += 0;
    gpt2_mlp_c_fc_temp[2][1] += 0;
    gpt2_mlp_c_fc_temp[2][2] += 0;
    gpt2_mlp_c_fc_temp[2][3] += 0;
    gpt2_mlp_c_fc_temp[2][4] += 0;
    gpt2_mlp_c_fc_temp[2][5] += 0;
    gpt2_mlp_c_fc_temp[2][6] += 0;
    gpt2_mlp_c_fc_temp[2][7] += 0;
    gpt2_mlp_c_fc_temp[2][8] += 0;
    gpt2_mlp_c_fc_temp[2][9] += 0;
    gpt2_mlp_c_fc_temp[2][10] += 0;
    gpt2_mlp_c_fc_temp[2][11] += 0;
    gpt2_mlp_c_fc_temp[2][12] += 0;
    gpt2_mlp_c_fc_temp[2][13] += 0;
    gpt2_mlp_c_fc_temp[2][14] += 0;
    gpt2_mlp_c_fc_temp[2][15] += 0;
    gpt2_mlp_c_fc_temp[3][0] += 0;
    gpt2_mlp_c_fc_temp[3][1] += 0;
    gpt2_mlp_c_fc_temp[3][2] += 0;
    gpt2_mlp_c_fc_temp[3][3] += 0;
    gpt2_mlp_c_fc_temp[3][4] += 0;
    gpt2_mlp_c_fc_temp[3][5] += 0;
    gpt2_mlp_c_fc_temp[3][6] += 0;
    gpt2_mlp_c_fc_temp[3][7] += 0;
    gpt2_mlp_c_fc_temp[3][8] += 0;
    gpt2_mlp_c_fc_temp[3][9] += 0;
    gpt2_mlp_c_fc_temp[3][10] += 0;
    gpt2_mlp_c_fc_temp[3][11] += 0;
    gpt2_mlp_c_fc_temp[3][12] += 0;
    gpt2_mlp_c_fc_temp[3][13] += 0;
    gpt2_mlp_c_fc_temp[3][14] += 0;
    gpt2_mlp_c_fc_temp[3][15] += 0;

    gpt2_mlp_new_gelu_activation(gpt2_mlp_c_fc_temp[0], gpt2_mlp_gelu_temp[0]);
    gpt2_mlp_new_gelu_activation(gpt2_mlp_c_fc_temp[1], gpt2_mlp_gelu_temp[1]);
    gpt2_mlp_new_gelu_activation(gpt2_mlp_c_fc_temp[2], gpt2_mlp_gelu_temp[2]);
    gpt2_mlp_new_gelu_activation(gpt2_mlp_c_fc_temp[3], gpt2_mlp_gelu_temp[3]);

    // Conv1D c_proj
    // Matrix with shape [4 * 4, 4]
    output[0][0] =
        (gpt2_mlp_gelu_temp[0][0] * 0.0) + (gpt2_mlp_gelu_temp[0][1] * 0.0) +
        (gpt2_mlp_gelu_temp[0][2] * 0.0) + (gpt2_mlp_gelu_temp[0][3] * 0.0) +
        (gpt2_mlp_gelu_temp[0][4] * 0.0) + (gpt2_mlp_gelu_temp[0][5] * 0.0) +
        (gpt2_mlp_gelu_temp[0][6] * 0.0) + (gpt2_mlp_gelu_temp[0][7] * 0.0) +
        (gpt2_mlp_gelu_temp[0][8] * 0.0) + (gpt2_mlp_gelu_temp[0][9] * 0.0) +
        (gpt2_mlp_gelu_temp[0][10] * 0.0) + (gpt2_mlp_gelu_temp[0][11] * 0.0) +
        (gpt2_mlp_gelu_temp[0][12] * 0.0) + (gpt2_mlp_gelu_temp[0][13] * 0.0) +
        (gpt2_mlp_gelu_temp[0][14] * 0.0) + (gpt2_mlp_gelu_temp[0][15] * 0.0);
    output[0][1] = (gpt2_mlp_gelu_temp[0][0] * 0.001) +
                   (gpt2_mlp_gelu_temp[0][1] * 0.001) +
                   (gpt2_mlp_gelu_temp[0][2] * 0.001) +
                   (gpt2_mlp_gelu_temp[0][3] * 0.001) +
                   (gpt2_mlp_gelu_temp[0][4] * 0.001) +
                   (gpt2_mlp_gelu_temp[0][5] * 0.001) +
                   (gpt2_mlp_gelu_temp[0][6] * 0.001) +
                   (gpt2_mlp_gelu_temp[0][7] * 0.001) +
                   (gpt2_mlp_gelu_temp[0][8] * 0.001) +
                   (gpt2_mlp_gelu_temp[0][9] * 0.001) +
                   (gpt2_mlp_gelu_temp[0][10] * 0.001) +
                   (gpt2_mlp_gelu_temp[0][11] * 0.001) +
                   (gpt2_mlp_gelu_temp[0][12] * 0.001) +
                   (gpt2_mlp_gelu_temp[0][13] * 0.001) +
                   (gpt2_mlp_gelu_temp[0][14] * 0.001) +
                   (gpt2_mlp_gelu_temp[0][15] * 0.001);
    output[0][2] = (gpt2_mlp_gelu_temp[0][0] * 0.002) +
                   (gpt2_mlp_gelu_temp[0][1] * 0.002) +
                   (gpt2_mlp_gelu_temp[0][2] * 0.002) +
                   (gpt2_mlp_gelu_temp[0][3] * 0.002) +
                   (gpt2_mlp_gelu_temp[0][4] * 0.002) +
                   (gpt2_mlp_gelu_temp[0][5] * 0.002) +
                   (gpt2_mlp_gelu_temp[0][6] * 0.002) +
                   (gpt2_mlp_gelu_temp[0][7] * 0.002) +
                   (gpt2_mlp_gelu_temp[0][8] * 0.002) +
                   (gpt2_mlp_gelu_temp[0][9] * 0.002) +
                   (gpt2_mlp_gelu_temp[0][10] * 0.002) +
                   (gpt2_mlp_gelu_temp[0][11] * 0.002) +
                   (gpt2_mlp_gelu_temp[0][12] * 0.002) +
                   (gpt2_mlp_gelu_temp[0][13] * 0.002) +
                   (gpt2_mlp_gelu_temp[0][14] * 0.002) +
                   (gpt2_mlp_gelu_temp[0][15] * 0.002);
    output[0][3] = (gpt2_mlp_gelu_temp[0][0] * 0.003) +
                   (gpt2_mlp_gelu_temp[0][1] * 0.003) +
                   (gpt2_mlp_gelu_temp[0][2] * 0.003) +
                   (gpt2_mlp_gelu_temp[0][3] * 0.003) +
                   (gpt2_mlp_gelu_temp[0][4] * 0.003) +
                   (gpt2_mlp_gelu_temp[0][5] * 0.003) +
                   (gpt2_mlp_gelu_temp[0][6] * 0.003) +
                   (gpt2_mlp_gelu_temp[0][7] * 0.003) +
                   (gpt2_mlp_gelu_temp[0][8] * 0.003) +
                   (gpt2_mlp_gelu_temp[0][9] * 0.003) +
                   (gpt2_mlp_gelu_temp[0][10] * 0.003) +
                   (gpt2_mlp_gelu_temp[0][11] * 0.003) +
                   (gpt2_mlp_gelu_temp[0][12] * 0.003) +
                   (gpt2_mlp_gelu_temp[0][13] * 0.003) +
                   (gpt2_mlp_gelu_temp[0][14] * 0.003) +
                   (gpt2_mlp_gelu_temp[0][15] * 0.003);
    output[1][0] =
        (gpt2_mlp_gelu_temp[1][0] * 0.0) + (gpt2_mlp_gelu_temp[1][1] * 0.0) +
        (gpt2_mlp_gelu_temp[1][2] * 0.0) + (gpt2_mlp_gelu_temp[1][3] * 0.0) +
        (gpt2_mlp_gelu_temp[1][4] * 0.0) + (gpt2_mlp_gelu_temp[1][5] * 0.0) +
        (gpt2_mlp_gelu_temp[1][6] * 0.0) + (gpt2_mlp_gelu_temp[1][7] * 0.0) +
        (gpt2_mlp_gelu_temp[1][8] * 0.0) + (gpt2_mlp_gelu_temp[1][9] * 0.0) +
        (gpt2_mlp_gelu_temp[1][10] * 0.0) + (gpt2_mlp_gelu_temp[1][11] * 0.0) +
        (gpt2_mlp_gelu_temp[1][12] * 0.0) + (gpt2_mlp_gelu_temp[1][13] * 0.0) +
        (gpt2_mlp_gelu_temp[1][14] * 0.0) + (gpt2_mlp_gelu_temp[1][15] * 0.0);
    output[1][1] = (gpt2_mlp_gelu_temp[1][0] * 0.001) +
                   (gpt2_mlp_gelu_temp[1][1] * 0.001) +
                   (gpt2_mlp_gelu_temp[1][2] * 0.001) +
                   (gpt2_mlp_gelu_temp[1][3] * 0.001) +
                   (gpt2_mlp_gelu_temp[1][4] * 0.001) +
                   (gpt2_mlp_gelu_temp[1][5] * 0.001) +
                   (gpt2_mlp_gelu_temp[1][6] * 0.001) +
                   (gpt2_mlp_gelu_temp[1][7] * 0.001) +
                   (gpt2_mlp_gelu_temp[1][8] * 0.001) +
                   (gpt2_mlp_gelu_temp[1][9] * 0.001) +
                   (gpt2_mlp_gelu_temp[1][10] * 0.001) +
                   (gpt2_mlp_gelu_temp[1][11] * 0.001) +
                   (gpt2_mlp_gelu_temp[1][12] * 0.001) +
                   (gpt2_mlp_gelu_temp[1][13] * 0.001) +
                   (gpt2_mlp_gelu_temp[1][14] * 0.001) +
                   (gpt2_mlp_gelu_temp[1][15] * 0.001);
    output[1][2] = (gpt2_mlp_gelu_temp[1][0] * 0.002) +
                   (gpt2_mlp_gelu_temp[1][1] * 0.002) +
                   (gpt2_mlp_gelu_temp[1][2] * 0.002) +
                   (gpt2_mlp_gelu_temp[1][3] * 0.002) +
                   (gpt2_mlp_gelu_temp[1][4] * 0.002) +
                   (gpt2_mlp_gelu_temp[1][5] * 0.002) +
                   (gpt2_mlp_gelu_temp[1][6] * 0.002) +
                   (gpt2_mlp_gelu_temp[1][7] * 0.002) +
                   (gpt2_mlp_gelu_temp[1][8] * 0.002) +
                   (gpt2_mlp_gelu_temp[1][9] * 0.002) +
                   (gpt2_mlp_gelu_temp[1][10] * 0.002) +
                   (gpt2_mlp_gelu_temp[1][11] * 0.002) +
                   (gpt2_mlp_gelu_temp[1][12] * 0.002) +
                   (gpt2_mlp_gelu_temp[1][13] * 0.002) +
                   (gpt2_mlp_gelu_temp[1][14] * 0.002) +
                   (gpt2_mlp_gelu_temp[1][15] * 0.002);
    output[1][3] = (gpt2_mlp_gelu_temp[1][0] * 0.003) +
                   (gpt2_mlp_gelu_temp[1][1] * 0.003) +
                   (gpt2_mlp_gelu_temp[1][2] * 0.003) +
                   (gpt2_mlp_gelu_temp[1][3] * 0.003) +
                   (gpt2_mlp_gelu_temp[1][4] * 0.003) +
                   (gpt2_mlp_gelu_temp[1][5] * 0.003) +
                   (gpt2_mlp_gelu_temp[1][6] * 0.003) +
                   (gpt2_mlp_gelu_temp[1][7] * 0.003) +
                   (gpt2_mlp_gelu_temp[1][8] * 0.003) +
                   (gpt2_mlp_gelu_temp[1][9] * 0.003) +
                   (gpt2_mlp_gelu_temp[1][10] * 0.003) +
                   (gpt2_mlp_gelu_temp[1][11] * 0.003) +
                   (gpt2_mlp_gelu_temp[1][12] * 0.003) +
                   (gpt2_mlp_gelu_temp[1][13] * 0.003) +
                   (gpt2_mlp_gelu_temp[1][14] * 0.003) +
                   (gpt2_mlp_gelu_temp[1][15] * 0.003);
    output[2][0] =
        (gpt2_mlp_gelu_temp[2][0] * 0.0) + (gpt2_mlp_gelu_temp[2][1] * 0.0) +
        (gpt2_mlp_gelu_temp[2][2] * 0.0) + (gpt2_mlp_gelu_temp[2][3] * 0.0) +
        (gpt2_mlp_gelu_temp[2][4] * 0.0) + (gpt2_mlp_gelu_temp[2][5] * 0.0) +
        (gpt2_mlp_gelu_temp[2][6] * 0.0) + (gpt2_mlp_gelu_temp[2][7] * 0.0) +
        (gpt2_mlp_gelu_temp[2][8] * 0.0) + (gpt2_mlp_gelu_temp[2][9] * 0.0) +
        (gpt2_mlp_gelu_temp[2][10] * 0.0) + (gpt2_mlp_gelu_temp[2][11] * 0.0) +
        (gpt2_mlp_gelu_temp[2][12] * 0.0) + (gpt2_mlp_gelu_temp[2][13] * 0.0) +
        (gpt2_mlp_gelu_temp[2][14] * 0.0) + (gpt2_mlp_gelu_temp[2][15] * 0.0);
    output[2][1] = (gpt2_mlp_gelu_temp[2][0] * 0.001) +
                   (gpt2_mlp_gelu_temp[2][1] * 0.001) +
                   (gpt2_mlp_gelu_temp[2][2] * 0.001) +
                   (gpt2_mlp_gelu_temp[2][3] * 0.001) +
                   (gpt2_mlp_gelu_temp[2][4] * 0.001) +
                   (gpt2_mlp_gelu_temp[2][5] * 0.001) +
                   (gpt2_mlp_gelu_temp[2][6] * 0.001) +
                   (gpt2_mlp_gelu_temp[2][7] * 0.001) +
                   (gpt2_mlp_gelu_temp[2][8] * 0.001) +
                   (gpt2_mlp_gelu_temp[2][9] * 0.001) +
                   (gpt2_mlp_gelu_temp[2][10] * 0.001) +
                   (gpt2_mlp_gelu_temp[2][11] * 0.001) +
                   (gpt2_mlp_gelu_temp[2][12] * 0.001) +
                   (gpt2_mlp_gelu_temp[2][13] * 0.001) +
                   (gpt2_mlp_gelu_temp[2][14] * 0.001) +
                   (gpt2_mlp_gelu_temp[2][15] * 0.001);
    output[2][2] = (gpt2_mlp_gelu_temp[2][0] * 0.002) +
                   (gpt2_mlp_gelu_temp[2][1] * 0.002) +
                   (gpt2_mlp_gelu_temp[2][2] * 0.002) +
                   (gpt2_mlp_gelu_temp[2][3] * 0.002) +
                   (gpt2_mlp_gelu_temp[2][4] * 0.002) +
                   (gpt2_mlp_gelu_temp[2][5] * 0.002) +
                   (gpt2_mlp_gelu_temp[2][6] * 0.002) +
                   (gpt2_mlp_gelu_temp[2][7] * 0.002) +
                   (gpt2_mlp_gelu_temp[2][8] * 0.002) +
                   (gpt2_mlp_gelu_temp[2][9] * 0.002) +
                   (gpt2_mlp_gelu_temp[2][10] * 0.002) +
                   (gpt2_mlp_gelu_temp[2][11] * 0.002) +
                   (gpt2_mlp_gelu_temp[2][12] * 0.002) +
                   (gpt2_mlp_gelu_temp[2][13] * 0.002) +
                   (gpt2_mlp_gelu_temp[2][14] * 0.002) +
                   (gpt2_mlp_gelu_temp[2][15] * 0.002);
    output[2][3] = (gpt2_mlp_gelu_temp[2][0] * 0.003) +
                   (gpt2_mlp_gelu_temp[2][1] * 0.003) +
                   (gpt2_mlp_gelu_temp[2][2] * 0.003) +
                   (gpt2_mlp_gelu_temp[2][3] * 0.003) +
                   (gpt2_mlp_gelu_temp[2][4] * 0.003) +
                   (gpt2_mlp_gelu_temp[2][5] * 0.003) +
                   (gpt2_mlp_gelu_temp[2][6] * 0.003) +
                   (gpt2_mlp_gelu_temp[2][7] * 0.003) +
                   (gpt2_mlp_gelu_temp[2][8] * 0.003) +
                   (gpt2_mlp_gelu_temp[2][9] * 0.003) +
                   (gpt2_mlp_gelu_temp[2][10] * 0.003) +
                   (gpt2_mlp_gelu_temp[2][11] * 0.003) +
                   (gpt2_mlp_gelu_temp[2][12] * 0.003) +
                   (gpt2_mlp_gelu_temp[2][13] * 0.003) +
                   (gpt2_mlp_gelu_temp[2][14] * 0.003) +
                   (gpt2_mlp_gelu_temp[2][15] * 0.003);
    output[3][0] =
        (gpt2_mlp_gelu_temp[3][0] * 0.0) + (gpt2_mlp_gelu_temp[3][1] * 0.0) +
        (gpt2_mlp_gelu_temp[3][2] * 0.0) + (gpt2_mlp_gelu_temp[3][3] * 0.0) +
        (gpt2_mlp_gelu_temp[3][4] * 0.0) + (gpt2_mlp_gelu_temp[3][5] * 0.0) +
        (gpt2_mlp_gelu_temp[3][6] * 0.0) + (gpt2_mlp_gelu_temp[3][7] * 0.0) +
        (gpt2_mlp_gelu_temp[3][8] * 0.0) + (gpt2_mlp_gelu_temp[3][9] * 0.0) +
        (gpt2_mlp_gelu_temp[3][10] * 0.0) + (gpt2_mlp_gelu_temp[3][11] * 0.0) +
        (gpt2_mlp_gelu_temp[3][12] * 0.0) + (gpt2_mlp_gelu_temp[3][13] * 0.0) +
        (gpt2_mlp_gelu_temp[3][14] * 0.0) + (gpt2_mlp_gelu_temp[3][15] * 0.0);
    output[3][1] = (gpt2_mlp_gelu_temp[3][0] * 0.001) +
                   (gpt2_mlp_gelu_temp[3][1] * 0.001) +
                   (gpt2_mlp_gelu_temp[3][2] * 0.001) +
                   (gpt2_mlp_gelu_temp[3][3] * 0.001) +
                   (gpt2_mlp_gelu_temp[3][4] * 0.001) +
                   (gpt2_mlp_gelu_temp[3][5] * 0.001) +
                   (gpt2_mlp_gelu_temp[3][6] * 0.001) +
                   (gpt2_mlp_gelu_temp[3][7] * 0.001) +
                   (gpt2_mlp_gelu_temp[3][8] * 0.001) +
                   (gpt2_mlp_gelu_temp[3][9] * 0.001) +
                   (gpt2_mlp_gelu_temp[3][10] * 0.001) +
                   (gpt2_mlp_gelu_temp[3][11] * 0.001) +
                   (gpt2_mlp_gelu_temp[3][12] * 0.001) +
                   (gpt2_mlp_gelu_temp[3][13] * 0.001) +
                   (gpt2_mlp_gelu_temp[3][14] * 0.001) +
                   (gpt2_mlp_gelu_temp[3][15] * 0.001);
    output[3][2] = (gpt2_mlp_gelu_temp[3][0] * 0.002) +
                   (gpt2_mlp_gelu_temp[3][1] * 0.002) +
                   (gpt2_mlp_gelu_temp[3][2] * 0.002) +
                   (gpt2_mlp_gelu_temp[3][3] * 0.002) +
                   (gpt2_mlp_gelu_temp[3][4] * 0.002) +
                   (gpt2_mlp_gelu_temp[3][5] * 0.002) +
                   (gpt2_mlp_gelu_temp[3][6] * 0.002) +
                   (gpt2_mlp_gelu_temp[3][7] * 0.002) +
                   (gpt2_mlp_gelu_temp[3][8] * 0.002) +
                   (gpt2_mlp_gelu_temp[3][9] * 0.002) +
                   (gpt2_mlp_gelu_temp[3][10] * 0.002) +
                   (gpt2_mlp_gelu_temp[3][11] * 0.002) +
                   (gpt2_mlp_gelu_temp[3][12] * 0.002) +
                   (gpt2_mlp_gelu_temp[3][13] * 0.002) +
                   (gpt2_mlp_gelu_temp[3][14] * 0.002) +
                   (gpt2_mlp_gelu_temp[3][15] * 0.002);
    output[3][3] = (gpt2_mlp_gelu_temp[3][0] * 0.003) +
                   (gpt2_mlp_gelu_temp[3][1] * 0.003) +
                   (gpt2_mlp_gelu_temp[3][2] * 0.003) +
                   (gpt2_mlp_gelu_temp[3][3] * 0.003) +
                   (gpt2_mlp_gelu_temp[3][4] * 0.003) +
                   (gpt2_mlp_gelu_temp[3][5] * 0.003) +
                   (gpt2_mlp_gelu_temp[3][6] * 0.003) +
                   (gpt2_mlp_gelu_temp[3][7] * 0.003) +
                   (gpt2_mlp_gelu_temp[3][8] * 0.003) +
                   (gpt2_mlp_gelu_temp[3][9] * 0.003) +
                   (gpt2_mlp_gelu_temp[3][10] * 0.003) +
                   (gpt2_mlp_gelu_temp[3][11] * 0.003) +
                   (gpt2_mlp_gelu_temp[3][12] * 0.003) +
                   (gpt2_mlp_gelu_temp[3][13] * 0.003) +
                   (gpt2_mlp_gelu_temp[3][14] * 0.003) +
                   (gpt2_mlp_gelu_temp[3][15] * 0.003);

    // Vector of size 4
    output[0][0] += 0;
    output[0][1] += 0;
    output[0][2] += 0;
    output[0][3] += 0;
    output[1][0] += 0;
    output[1][1] += 0;
    output[1][2] += 0;
    output[1][3] += 0;
    output[2][0] += 0;
    output[2][1] += 0;
    output[2][2] += 0;
    output[2][3] += 0;
    output[3][0] += 0;
    output[3][1] += 0;
    output[3][2] += 0;
    output[3][3] += 0;
}

// START_TEST
int main(int argc, char **argv) {
    float32_t **input = (float32_t **)malloc(sizeof(float32_t *) * 4);
    float32_t **output = (float32_t **)malloc(sizeof(float32_t *) * 4);

    for (int i = 0; i < 4; i += 1) {
        input[i] = (float32_t *)malloc(sizeof(float32_t) * 4);
        output[i] = (float32_t *)malloc(sizeof(float32_t) * 4);
        // Set input as [1.0, 2.0, 3.0, 4.0 ... ]
        for (int j = 0; j < 4; j += 1) {
            input[i][j] = (j + 1) * 1.0;
        }
    }

    gpt2_mlp_initialize_heap_memory();

    gpt2_mlp_mlp(input, output);

    for (int i = 0; i < 4; i += 1) {
        printf("Sequence %d Input\n", i);
        for (int j = 0; j < 4; j += 1) {
            printf("idx %d -> %f\n", j, input[i][j]);
        }

        printf("Sequence %d Output\n", i);
        for (int j = 0; j < 4; j += 1) {
            printf("idx %d -> %f\n", j, output[i][j]);
        }
    }

    return 0;
}
// END_TEST
