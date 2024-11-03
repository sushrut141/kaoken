// START_METHODS
#include <arm_neon.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Matrix of size [1, 24]
// used for storing intermediate values in c_attn conv1d step
float32_t **gpt2_attention_0_c_attn_temp = NULL;
// Matrix of size [1, 4]
// used for storing intermediate values in multi-head attention
float32_t **gpt2_attention_0_multi_head_temp = NULL;
// Matrix to store mask used in multi head attention  of size
// [1, 1]
float32_t **gpt2_attention_0_multi_head_attention_mask = NULL;

// TODO - precompute and make this const;
int64_t C_ATTN_OUTPUT_Q_OFFSET = 0;
int64_t C_ATTN_OUTPUT_K_OFFSET = 8;
int64_t C_ATTN_OUTPUT_V_OFFSET = 2 * 8;

// Called only once at startup to populate the attention mask
// so we can skip unrolling
void gpt2_attention_0_populate_attention_mask() {
    for (int i = 0; i < 1; i += 1) {
        for (int j = 0; j < 1; j += 1) {
            float32_t value = (j <= i ? 1.0 : 0.0);
            gpt2_attention_0_multi_head_attention_mask[i][j] = value;
        }
    }
}

// Initializes the weights used in the attention layer.
// Must be called before calling gpt2_attention_0_attention.
void gpt2_attention_0_initialize_heap_memory() {
    gpt2_attention_0_c_attn_temp =
        (float32_t **)malloc(sizeof(float32_t *) * 1);
    gpt2_attention_0_multi_head_temp =
        (float32_t **)malloc(sizeof(float32_t *) * 1);
    gpt2_attention_0_multi_head_attention_mask =
        (float32_t **)malloc(sizeof(float32_t *) * 1);

    gpt2_attention_0_c_attn_temp[0] =
        (float32_t *)malloc(sizeof(float32_t) * 24);
    gpt2_attention_0_multi_head_temp[0] =
        (float32_t *)malloc(sizeof(float32_t) * 4);
    gpt2_attention_0_multi_head_attention_mask[0] =
        (float32_t *)malloc(sizeof(float32_t) * 1);

    // create the attention mask
    gpt2_attention_0_populate_attention_mask();
}

static inline void gpt2_attention_0_apply_c_attn_conv1d(float32_t **input,
                                                        float32_t **output) {
    // Conv1D c_attn
    // Matrix with shape [8, 24]
    output[0][0] = (input[0][0] * 0.0) + (input[0][1] * 0.0) +
                   (input[0][2] * 0.0) + (input[0][3] * 0.0) +
                   (input[0][4] * 0.0) + (input[0][5] * 0.0) +
                   (input[0][6] * 0.0) + (input[0][7] * 0.0);
    output[0][1] = (input[0][0] * 0.001) + (input[0][1] * 0.001) +
                   (input[0][2] * 0.001) + (input[0][3] * 0.001) +
                   (input[0][4] * 0.001) + (input[0][5] * 0.001) +
                   (input[0][6] * 0.001) + (input[0][7] * 0.001);
    output[0][2] = (input[0][0] * 0.002) + (input[0][1] * 0.002) +
                   (input[0][2] * 0.002) + (input[0][3] * 0.002) +
                   (input[0][4] * 0.002) + (input[0][5] * 0.002) +
                   (input[0][6] * 0.002) + (input[0][7] * 0.002);
    output[0][3] = (input[0][0] * 0.003) + (input[0][1] * 0.003) +
                   (input[0][2] * 0.003) + (input[0][3] * 0.003) +
                   (input[0][4] * 0.003) + (input[0][5] * 0.003) +
                   (input[0][6] * 0.003) + (input[0][7] * 0.003);
    output[0][4] = (input[0][0] * 0.004) + (input[0][1] * 0.004) +
                   (input[0][2] * 0.004) + (input[0][3] * 0.004) +
                   (input[0][4] * 0.004) + (input[0][5] * 0.004) +
                   (input[0][6] * 0.004) + (input[0][7] * 0.004);
    output[0][5] = (input[0][0] * 0.005) + (input[0][1] * 0.005) +
                   (input[0][2] * 0.005) + (input[0][3] * 0.005) +
                   (input[0][4] * 0.005) + (input[0][5] * 0.005) +
                   (input[0][6] * 0.005) + (input[0][7] * 0.005);
    output[0][6] = (input[0][0] * 0.006) + (input[0][1] * 0.006) +
                   (input[0][2] * 0.006) + (input[0][3] * 0.006) +
                   (input[0][4] * 0.006) + (input[0][5] * 0.006) +
                   (input[0][6] * 0.006) + (input[0][7] * 0.006);
    output[0][7] = (input[0][0] * 0.007) + (input[0][1] * 0.007) +
                   (input[0][2] * 0.007) + (input[0][3] * 0.007) +
                   (input[0][4] * 0.007) + (input[0][5] * 0.007) +
                   (input[0][6] * 0.007) + (input[0][7] * 0.007);
    output[0][8] = (input[0][0] * 0.008) + (input[0][1] * 0.008) +
                   (input[0][2] * 0.008) + (input[0][3] * 0.008) +
                   (input[0][4] * 0.008) + (input[0][5] * 0.008) +
                   (input[0][6] * 0.008) + (input[0][7] * 0.008);
    output[0][9] = (input[0][0] * 0.009000000000000001) +
                   (input[0][1] * 0.009000000000000001) +
                   (input[0][2] * 0.009000000000000001) +
                   (input[0][3] * 0.009000000000000001) +
                   (input[0][4] * 0.009000000000000001) +
                   (input[0][5] * 0.009000000000000001) +
                   (input[0][6] * 0.009000000000000001) +
                   (input[0][7] * 0.009000000000000001);
    output[0][10] = (input[0][0] * 0.01) + (input[0][1] * 0.01) +
                    (input[0][2] * 0.01) + (input[0][3] * 0.01) +
                    (input[0][4] * 0.01) + (input[0][5] * 0.01) +
                    (input[0][6] * 0.01) + (input[0][7] * 0.01);
    output[0][11] = (input[0][0] * 0.011) + (input[0][1] * 0.011) +
                    (input[0][2] * 0.011) + (input[0][3] * 0.011) +
                    (input[0][4] * 0.011) + (input[0][5] * 0.011) +
                    (input[0][6] * 0.011) + (input[0][7] * 0.011);
    output[0][12] = (input[0][0] * 0.012) + (input[0][1] * 0.012) +
                    (input[0][2] * 0.012) + (input[0][3] * 0.012) +
                    (input[0][4] * 0.012) + (input[0][5] * 0.012) +
                    (input[0][6] * 0.012) + (input[0][7] * 0.012);
    output[0][13] = (input[0][0] * 0.013000000000000001) +
                    (input[0][1] * 0.013000000000000001) +
                    (input[0][2] * 0.013000000000000001) +
                    (input[0][3] * 0.013000000000000001) +
                    (input[0][4] * 0.013000000000000001) +
                    (input[0][5] * 0.013000000000000001) +
                    (input[0][6] * 0.013000000000000001) +
                    (input[0][7] * 0.013000000000000001);
    output[0][14] = (input[0][0] * 0.014) + (input[0][1] * 0.014) +
                    (input[0][2] * 0.014) + (input[0][3] * 0.014) +
                    (input[0][4] * 0.014) + (input[0][5] * 0.014) +
                    (input[0][6] * 0.014) + (input[0][7] * 0.014);
    output[0][15] = (input[0][0] * 0.015) + (input[0][1] * 0.015) +
                    (input[0][2] * 0.015) + (input[0][3] * 0.015) +
                    (input[0][4] * 0.015) + (input[0][5] * 0.015) +
                    (input[0][6] * 0.015) + (input[0][7] * 0.015);
    output[0][16] = (input[0][0] * 0.016) + (input[0][1] * 0.016) +
                    (input[0][2] * 0.016) + (input[0][3] * 0.016) +
                    (input[0][4] * 0.016) + (input[0][5] * 0.016) +
                    (input[0][6] * 0.016) + (input[0][7] * 0.016);
    output[0][17] = (input[0][0] * 0.017) + (input[0][1] * 0.017) +
                    (input[0][2] * 0.017) + (input[0][3] * 0.017) +
                    (input[0][4] * 0.017) + (input[0][5] * 0.017) +
                    (input[0][6] * 0.017) + (input[0][7] * 0.017);
    output[0][18] = (input[0][0] * 0.018000000000000002) +
                    (input[0][1] * 0.018000000000000002) +
                    (input[0][2] * 0.018000000000000002) +
                    (input[0][3] * 0.018000000000000002) +
                    (input[0][4] * 0.018000000000000002) +
                    (input[0][5] * 0.018000000000000002) +
                    (input[0][6] * 0.018000000000000002) +
                    (input[0][7] * 0.018000000000000002);
    output[0][19] = (input[0][0] * 0.019) + (input[0][1] * 0.019) +
                    (input[0][2] * 0.019) + (input[0][3] * 0.019) +
                    (input[0][4] * 0.019) + (input[0][5] * 0.019) +
                    (input[0][6] * 0.019) + (input[0][7] * 0.019);
    output[0][20] = (input[0][0] * 0.02) + (input[0][1] * 0.02) +
                    (input[0][2] * 0.02) + (input[0][3] * 0.02) +
                    (input[0][4] * 0.02) + (input[0][5] * 0.02) +
                    (input[0][6] * 0.02) + (input[0][7] * 0.02);
    output[0][21] = (input[0][0] * 0.021) + (input[0][1] * 0.021) +
                    (input[0][2] * 0.021) + (input[0][3] * 0.021) +
                    (input[0][4] * 0.021) + (input[0][5] * 0.021) +
                    (input[0][6] * 0.021) + (input[0][7] * 0.021);
    output[0][22] = (input[0][0] * 0.022) + (input[0][1] * 0.022) +
                    (input[0][2] * 0.022) + (input[0][3] * 0.022) +
                    (input[0][4] * 0.022) + (input[0][5] * 0.022) +
                    (input[0][6] * 0.022) + (input[0][7] * 0.022);
    output[0][23] = (input[0][0] * 0.023) + (input[0][1] * 0.023) +
                    (input[0][2] * 0.023) + (input[0][3] * 0.023) +
                    (input[0][4] * 0.023) + (input[0][5] * 0.023) +
                    (input[0][6] * 0.023) + (input[0][7] * 0.023);

    // Vector of size 24
    output[0][0] += 0.0;
    output[0][1] += 0.0;
    output[0][2] += 0.0;
    output[0][3] += 0.0;
    output[0][4] += 0.0;
    output[0][5] += 0.0;
    output[0][6] += 0.0;
    output[0][7] += 0.0;
    output[0][8] += 0.0;
    output[0][9] += 0.0;
    output[0][10] += 0.0;
    output[0][11] += 0.0;
    output[0][12] += 0.0;
    output[0][13] += 0.0;
    output[0][14] += 0.0;
    output[0][15] += 0.0;
    output[0][16] += 0.0;
    output[0][17] += 0.0;
    output[0][18] += 0.0;
    output[0][19] += 0.0;
    output[0][20] += 0.0;
    output[0][21] += 0.0;
    output[0][22] += 0.0;
    output[0][23] += 0.0;
}

static inline void gpt2_attention_0_populate_q_k_cell(float32_t **input,
                                                      float32_t *output,
                                                      int16_t head_idx,
                                                      int16_t seq_idx,
                                                      int16_t cell_idx) {
    float32_t *q =
        input[seq_idx] + C_ATTN_OUTPUT_Q_OFFSET + (head_idx * (8 / 2));
    float32_t *k =
        input[cell_idx] + C_ATTN_OUTPUT_K_OFFSET + (head_idx * (8 / 2));

    float64_t temp = 0.0;
    // q * k.transpose
    // [s, 64] * [s, 64].t
    temp += (*(q + 0)) * (*(k + 0));
    temp += (*(q + 1)) * (*(k + 1));
    temp += (*(q + 2)) * (*(k + 2));
    temp += (*(q + 3)) * (*(k + 3));
    // Scale attention weights
    temp = temp / pow(8 / 2, 0.5);
    output[cell_idx + (seq_idx * 1)] = temp;
}

static inline void gpt2_attention_0_populate_q_k_sequence(float32_t **input,
                                                          float32_t *output,
                                                          int16_t head_idx,
                                                          int16_t seq_idx) {
    gpt2_attention_0_populate_q_k_cell(input, output, head_idx, seq_idx, 0);
}

static inline void gpt2_attention_0_softmax_compute(float32_t *input,
                                                    int16_t seq_idx) {
    // We are assuming that softmax is to be computed for each sequence,
    // not across sequences in the [s, s] matrix
    // Use 64 bit floats to avoid overflow when computing exp(x).
    float64_t seq_sum = 0.0;
    // Compute the sum of exponents for each elemt in sequence that is not
    // masked
    seq_sum += expl((float64_t)input[0 + (seq_idx * 1)]) *
               gpt2_attention_0_multi_head_attention_mask[seq_idx][0];

    // Divide each unmasked elem by row sum
    // softmax(xi) = exp(xi) / sum([exp(xj) for xj in sequence])
    input[0 + (seq_idx * 1)] =
        (expl((float64_t)input[0 + (seq_idx * 1)]) *
         gpt2_attention_0_multi_head_attention_mask[seq_idx][0]) /
        seq_sum;
}

static inline void
gpt2_attention_0_populate_qk_v_cell(float32_t **input, float32_t *qk,
                                    float32_t **output, int16_t head_idx,
                                    int16_t seq_idx, int16_t cell_idx) {
    int64_t offset = (head_idx * 8 / 2);
    float32_t temp = 0.0;
    temp += qk[0 + (seq_idx * 1)] *
            (*(input[0] + C_ATTN_OUTPUT_V_OFFSET + offset + cell_idx));

    // populate in the output for given sequence and head
    output[seq_idx][cell_idx + offset] = temp;
}

static inline void gpt2_attention_0_populate_qk_v_sequence(float32_t **input,
                                                           float32_t *qk,
                                                           float32_t **output,
                                                           int16_t head_idx,
                                                           int16_t seq_idx) {
    gpt2_attention_0_populate_qk_v_cell(input, qk, output, head_idx, seq_idx,
                                        0);
    gpt2_attention_0_populate_qk_v_cell(input, qk, output, head_idx, seq_idx,
                                        1);
    gpt2_attention_0_populate_qk_v_cell(input, qk, output, head_idx, seq_idx,
                                        2);
    gpt2_attention_0_populate_qk_v_cell(input, qk, output, head_idx, seq_idx,
                                        3);
}

static inline void gpt2_attention_0_compute_attention_for_head(
    float32_t **input, float32_t **output, int16_t head_idx) {
    float32_t temp[1][1];
    gpt2_attention_0_populate_q_k_sequence(input, &temp[0][0], head_idx, 0);

    // compute softmax ensuring each sequence only attends to sequences that
    // came before it
    gpt2_attention_0_softmax_compute(&temp[0][0], 0);

    // multiple by v [s, s] * [s, 64] -> [s, 64]
    gpt2_attention_0_populate_qk_v_sequence(input, &temp[0][0], output,
                                            head_idx, 0);
}

static inline void
gpt2_attention_0_compute_multihead_attention(float32_t **input,
                                             float32_t **output) {
    gpt2_attention_0_compute_attention_for_head(input, output, 0);
    gpt2_attention_0_compute_attention_for_head(input, output, 1);
}

static void gpt2_attention_0_apply_c_proj_conv1d(float32_t **input,
                                                 float32_t **output) {
    // Conv1D c_proj
    // Matrix with shape [8, 4]
    output[0][0] = (input[0][0] * 0.0) + (input[0][1] * 0.0) +
                   (input[0][2] * 0.0) + (input[0][3] * 0.0) +
                   (input[0][4] * 0.0) + (input[0][5] * 0.0) +
                   (input[0][6] * 0.0) + (input[0][7] * 0.0);
    output[0][1] = (input[0][0] * 0.001) + (input[0][1] * 0.001) +
                   (input[0][2] * 0.001) + (input[0][3] * 0.001) +
                   (input[0][4] * 0.001) + (input[0][5] * 0.001) +
                   (input[0][6] * 0.001) + (input[0][7] * 0.001);
    output[0][2] = (input[0][0] * 0.002) + (input[0][1] * 0.002) +
                   (input[0][2] * 0.002) + (input[0][3] * 0.002) +
                   (input[0][4] * 0.002) + (input[0][5] * 0.002) +
                   (input[0][6] * 0.002) + (input[0][7] * 0.002);
    output[0][3] = (input[0][0] * 0.003) + (input[0][1] * 0.003) +
                   (input[0][2] * 0.003) + (input[0][3] * 0.003) +
                   (input[0][4] * 0.003) + (input[0][5] * 0.003) +
                   (input[0][6] * 0.003) + (input[0][7] * 0.003);
    output[0][4] = (input[0][0] * 0.004) + (input[0][1] * 0.004) +
                   (input[0][2] * 0.004) + (input[0][3] * 0.004) +
                   (input[0][4] * 0.004) + (input[0][5] * 0.004) +
                   (input[0][6] * 0.004) + (input[0][7] * 0.004);
    output[0][5] = (input[0][0] * 0.005) + (input[0][1] * 0.005) +
                   (input[0][2] * 0.005) + (input[0][3] * 0.005) +
                   (input[0][4] * 0.005) + (input[0][5] * 0.005) +
                   (input[0][6] * 0.005) + (input[0][7] * 0.005);
    output[0][6] = (input[0][0] * 0.006) + (input[0][1] * 0.006) +
                   (input[0][2] * 0.006) + (input[0][3] * 0.006) +
                   (input[0][4] * 0.006) + (input[0][5] * 0.006) +
                   (input[0][6] * 0.006) + (input[0][7] * 0.006);
    output[0][7] = (input[0][0] * 0.007) + (input[0][1] * 0.007) +
                   (input[0][2] * 0.007) + (input[0][3] * 0.007) +
                   (input[0][4] * 0.007) + (input[0][5] * 0.007) +
                   (input[0][6] * 0.007) + (input[0][7] * 0.007);

    // Vector of size 4
    output[0][0] += 0.0;
    output[0][1] += 0.0;
    output[0][2] += 0.0;
    output[0][3] += 0.0;
}

/**
 * Computes multi-head attention on the supplied sequence of embeddings of shape
 * [s, w] where s is the the number of elements in the sequence and w is the
 * size of the embedding.
 *
 * @param input List[List[float]]: [s, w] mapping of sequence to embedding.
 * @param output Output of shape [s, w]
 *
 * Interpolated Params:
 *
 *  - sequence length s: 1
 *  - embedding size w: 8
 *  - num heads h: 2
 */
void gpt2_attention_0_attention(float32_t **input, float32_t **output) {
    // apply c_attn conv1d
    // [s, w] * [w * c] -> [s, c]
    gpt2_attention_0_apply_c_attn_conv1d(input, gpt2_attention_0_c_attn_temp);

    // split into Q, K, V for each sequence of shape [s,
    // {C_ATTN_EMBEDDING_SIZE_COL / 3}] split into 2 heads such that for each
    // head we have q, k, v = Q.split(2), K.split(2), V.split(2) calculate
    // attention for each head finally combine value from all attention heads to
    // give [s, 4]
    gpt2_attention_0_compute_multihead_attention(
        gpt2_attention_0_c_attn_temp, gpt2_attention_0_multi_head_temp);

    // apply c_proj conv1d
    // [s, w] * [w, w]
    gpt2_attention_0_apply_c_proj_conv1d(gpt2_attention_0_multi_head_temp,
                                         output);
}

// START_TEST
int main(int argc, char **argv) {
    float32_t **input = (float32_t **)malloc(sizeof(float32_t *) * 1);
    float32_t **output = (float32_t **)malloc(sizeof(float32_t *) * 1);

    for (int i = 0; i < 1; i += 1) {
        input[i] = (float32_t *)malloc(sizeof(float32_t) * 8);
        // Set input as [1.0, 2.0, 3.0, 4.0 ... ]
        for (int j = 0; j < 8; j += 1) {
            input[i][j] = (j + 1) * 1.0;
        }

        output[i] = (float32_t *)malloc(sizeof(float32_t) * 8);
        memset(output[i], 0.0, sizeof(float32_t) * 8);
    }

    gpt2_attention_0_initialize_heap_memory();

    gpt2_attention_0_attention(input, output);

    for (int i = 0; i < 1; i += 1) {
        printf("Sequence %d Input\n", i);
        for (int j = 0; j < 8; j += 1) {
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
