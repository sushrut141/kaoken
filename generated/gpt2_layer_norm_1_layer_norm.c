// START_METHODS
#include <arm_neon.h>
#include <math.h>
#include <stdio.h>

const float WEIGHT_TABLE[4]={0.1, 0.1, 0.1, 0.1};
const float BIAS_TABLE[4]={0.1, 0.1, 0.1, 0.1};


inline void gpt2_layer_norm_1_layer_normalize(float32_t* input, float32_t* output) {
    // calculate mean
    float32x4_t sum_vec = vld1q_f32(input);
    float32_t sum = vaddvq_f32(sum_vec);
    float32x4_t vec_mean = vdupq_n_f32(sum / 4);

    // calculate standard deviation
    // substract mean from values in input
    vst1q_f32(output+(4*0), vsubq_f32(vld1q_f32(input+(4*0)), vec_mean));

    // square the values
    float32x4_t vec = vld1q_f32(output);
    vec = vld1q_f32(output+(4*0));
    vec = vmulq_f32(vec, vec);
    vst1q_f32(output+(4*0), vec);

    // add the squared values
    float32x4_t sum_vec2 = vdupq_n_f32(0.0f);
    sum_vec2 = vaddq_f32(sum_vec2, vld1q_f32(output+(4*0)));
    float32_t sum_of_squares = vaddvq_f32(sum_vec2);
    float32_t std_deviation = sqrt((sum_of_squares / 4) + 1e-05);
    float32x4_t vec_std_deviation = vdupq_n_f32(std_deviation);

    float32x4_t temp;

    // y = (input - mean) / std_deviation
    temp = vsubq_f32(vld1q_f32(input+(4*0)), vec_mean);
    temp = vdivq_f32(temp, vec_std_deviation);
    vst1q_f32(output+(4*0), temp);

    // output = (gamma * y) + beta
    temp = vmulq_f32(vld1q_f32(output+(4*0)), vld1q_f32(WEIGHT_TABLE+(4*0)));
    temp = vaddq_f32(temp, vld1q_f32(BIAS_TABLE+(4*0)));
    vst1q_f32(output+(4*0), temp);
}

