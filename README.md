## Kaoken


## GTP2 architecture and layer shapes

The sequence of compuations carried out and the input / output shapes of tensors at each
stage can be viewed [here](https://amaarora.github.io/posts/2020-02-18-annotatedGPT2.html).

## Reference for Neon Vector operations

### vld1q_f32

Loads 4 32bit float values into an SIMD register for processing.

### vaddvq_f32

Adds the values in all the lanes in an SIMD register and returns a single
float32_t value.

### vdupq_n_f32

Loads the same floating point value in all the lanes of an SIMD register and returns a float32x4_t value.

### vaddvq_f32

Simultaneously adds the 4 32bit float values in to separate SIMD registers and
returns the results as a float32X4_t.

### vst1q_f32

Stores the individual values in a 4 lane SIMD register into a floating point array of size 4.
Usefull for getting data out of SIMD registers into an array.

### vmulq_f32

Multiplies float32 values in two SIMD registers and returns a float32x4_t value.

### vsubq_f32

Subtracts float32 values across two SIMD registers and returns a float32x4_t value.

### vdivq_f32

Divides float32 values across two SIMD registers and returns a float32x4_t value.


# TODO
 - generate c files for each layer in GPT2
 - verify output of attention layer
    - benchmark performance of attention layer
    - compare sequential vs vectorized performance of attention layer
 - get inference working using binary built from c files
 - generalize generation of c files and building binary for any model