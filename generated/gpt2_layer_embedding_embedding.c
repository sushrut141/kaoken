// START_METHODS
#include <arm_neon.h>
#include <stdlib.h>
#include <string.h>

// Initialize heap memory for embedding table.
static float32_t **gpt2_layer_embedding_EMBEDDING_TABLE = NULL;

/**
 * Initializes the value in the embedding matrix.
 * Must be called before calling gpt2_layer_embedding_embedding_get.
 */
void gpt2_layer_embedding_initialize_embedding() {
    gpt2_layer_embedding_EMBEDDING_TABLE =
        (float32_t **)malloc(sizeof(float32_t *) * 4);

    // Initialize nested array memory to store embeddings.
    gpt2_layer_embedding_EMBEDDING_TABLE[0] =
        (float32_t *)malloc(sizeof(float32_t) * 4);
    gpt2_layer_embedding_EMBEDDING_TABLE[1] =
        (float32_t *)malloc(sizeof(float32_t) * 4);
    gpt2_layer_embedding_EMBEDDING_TABLE[2] =
        (float32_t *)malloc(sizeof(float32_t) * 4);
    gpt2_layer_embedding_EMBEDDING_TABLE[3] =
        (float32_t *)malloc(sizeof(float32_t) * 4);

    // Copy embedding values in to allocated memory

    float32_t temp[4];
    float32_t temp_0[4] = {0.1, 0.2, 0.3, 0.4};
    memcpy(gpt2_layer_embedding_EMBEDDING_TABLE[0], temp_0,
           sizeof(float32_t) * 4);
    float32_t temp_1[4] = {0.1, 0.2, 0.3, 0.4};
    memcpy(gpt2_layer_embedding_EMBEDDING_TABLE[1], temp_1,
           sizeof(float32_t) * 4);
    float32_t temp_2[4] = {0.1, 0.2, 0.3, 0.4};
    memcpy(gpt2_layer_embedding_EMBEDDING_TABLE[2], temp_2,
           sizeof(float32_t) * 4);
    float32_t temp_3[4] = {0.1, 0.2, 0.3, 0.4};
    memcpy(gpt2_layer_embedding_EMBEDDING_TABLE[3], temp_3,
           sizeof(float32_t) * 4);
}

/**
 * Returns the vector representation of a token by looking up the embedding
 * table. The embedding table is baked into the source.
 * gpt2_layer_embedding__initialize_embedding must be called once before this
 * method is invoked.
 *
 * It is expected that the caller knows the exact number of tokens in a context
 * window and generates source code with repeated calls to this method to avoid
 * iterating over the list of tokens.
 */
inline float *gpt2_layer_embedding_embedding_get(int idx) {
    return gpt2_layer_embedding_EMBEDDING_TABLE[idx];
}
