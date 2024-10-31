// START_METHODS
#include <arm_neon.h>
#include <math.h>
#include <stdio.h>

// 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
inline void gpt2_gelu_1_new_gelu_activation(float32_t* input, float32_t* output) {
    output[0] = input[0] + (0.044715* pow(input[0], 3));
    output[0] *= sqrt(2.0 / 3.14159);
    output[0] = 1.0 + tanh(output[0]);
    output[0] = 0.5 * input[0] * output[0];
    output[1] = input[1] + (0.044715* pow(input[1], 3));
    output[1] *= sqrt(2.0 / 3.14159);
    output[1] = 1.0 + tanh(output[1]);
    output[1] = 0.5 * input[1] * output[1];
    output[2] = input[2] + (0.044715* pow(input[2], 3));
    output[2] *= sqrt(2.0 / 3.14159);
    output[2] = 1.0 + tanh(output[2]);
    output[2] = 0.5 * input[2] * output[2];
    output[3] = input[3] + (0.044715* pow(input[3], 3));
    output[3] *= sqrt(2.0 / 3.14159);
    output[3] = 1.0 + tanh(output[3]);
    output[3] = 0.5 * input[3] * output[3];
}

// START_TEST
int main(int argc, char** argv) {
    float input[4];
    float output[4];

    for (int i = 0; i < 4; i += 1) {
        input[i] = 0.1 * (i + 1);
    }

    gpt2_gelu_1_new_gelu_activation(input, output);

    for (int i = 0; i < 4; i += 1) {
        printf("%f\n", output[i]);
    }

    return 0;
}
// END_TEST

