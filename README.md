## Kaoken

Speed up CPU inference on small language models(< 500M params) like [BERT](https://huggingface.co/google-bert/bert-base-uncased), [T5](https://huggingface.co/google-t5/t5-base) by X%.

There are many usecases which demand on-device inference for text completion, image generation / classification which cannot rely on communication with large foundation models like Claude, Gemini etc. The reasons for this limitation can arise from privacy requirements, lack of internet connectivity or the need for fast on-device inference for a better user experience.

In some cases invoking a foundation model for the task would be over kill.
Consider a small bot running on an embedded ARM device that only needs to carry out image
segmentation to traverse the environment without communications with a server.
Being able to carry out this segmentation on-device would drastically improve the speed and reliability of the bot.

Another concern is cost. Hosting a even a small model backed by GPUs for inference is extremely expensive with prices ranging from $0.5-0.7 / hour of GPU, if you can get them.
CPUs however are readily available in high quantity, are much cheaper than GPUs and
can be used for ther tasks as well such hosting webservers.

With Kaoken, we attempt to apply standard optimization techniques to speed up inference
of small models on the CPU and diversify the options available to developers when deploying smaller models.

### Baked Models

The [huggingface transformers library](https://github.com/huggingface/transformers), which is backed by PyTorch allows us to pull
any model from Huggingface and run inference on it. Each model can be thought of as a sequence of Tensor operations that are applied on the input.

The increased popularity of Transformer based models means that majority of models are built on the same building blocks like Attention, Layer Normalization and activation.
PyTorch allows us to inspect the building blocks of the model by merely printing it to console.
Below we see the blocks that make up the GPT2 model.
```
#   (transformer): GPT2Model(
#     (wte): Embedding(50257, 768)
#     (wpe): Embedding(1024, 768)
#     (drop): Dropout(p=0.1, inplace=False)
#     (h): ModuleList(
#       (0-11): 12 x GPT2Block(
#         (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
#         (attn): GPT2Attention(
#           (c_attn): Conv1D()
#           (c_proj): Conv1D()
#           (attn_dropout): Dropout(p=0.1, inplace=False)
#           (resid_dropout): Dropout(p=0.1, inplace=False)
#         )
#         (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
#         (mlp): GPT2MLP(
#           (c_fc): Conv1D()
#           (c_proj): Conv1D()
#           (act): NewGELUActivation()
#           (dropout): Dropout(p=0.1, inplace=False)
#         )
#       )
#     )
#     (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
#   )
#   (lm_head): Linear(in_features=768, out_features=50257, bias=False)
# )>
```

While PyTorch is great **generalized framework** for training models, during inference we can
do with something much simpler. We draw on work prior work done in Query Processing namely [JIT Compiled models](https://15721.courses.cs.cmu.edu/spring2018/slides/03-compilation.pdf) as inspiration for Kaoken.

When running inference on a pretrained model, we already know
 - The specification of the layers involed
 - The input and output dimensions of each layer.
 - The exact compuation (mainly matrix multiplcation and addition) to be carried out in each layer.

The way to speed up inference then would be to generated specialized code that compiles
to the smallest set of instructions in bytecode. Wherever possible, we can also leverage
vectorized instructions if the architecture supports it.

Consider the simple PyTorch operation below where we multiply two matrices of known dimensions, the most common operation in language models.

```
a = torch.rand(2, 2)
weights = torch.rand(2, 3)

output = torch.matmul(a, weights)
```

The above steps can be compiled to simple C code that explodes the matrix multiplication
into simple operations backed by pre-allocated memory.

```
output[0][0] = (a[0][0] * weights[0][0]) + (a[0][1] * weights[1][0]);
output[0][1] = (a[0][0] * weights[0][1]) + (a[0][1] * weights[1][1]);
...
output[1][0] = (a[1][0] * weights[0][0]) + (a[1][1] * weights[1][0]);
...
```

During model inference, operations usually involve the input sequence which is dynamic and weights which are fixed. Since the weights are already known we can bake them into the generated source.

```
output[0][0] = (a[0][0] * 0.077) + (a[0][1] * -0.087);
output[0][1] = (a[0][0] * 0.076) + (a[0][1] * 0.65);
...
output[1][0] = (a[1][0] * 0.077) + (a[1][1] * -0.087);
...
```

Another optimization(maybe premature) that has been applied is to avoid the use of loops
to compute the matrix multiplication. The sequence of operations to multiply matrices of known dimensions is simple and thus we can explode the operations into a known set of simple multiply / add instructions.

In CUDA programming for GPUs, loop unrolling is a common optimization.
For CPU's however, it is possible that the clang / gcc compiler would be smart enough to 
unroll the loops itself. The effects of these optimizations will be benchmarked in the future.

The methods defined above will be used to create what we will call **baked models**.
Parse the model specification from pretrained weights and generate specialized instructions
for each layer. The **baked layers** can then be stitched together per the model specification.
Based on the current state of the project, we can safely say that any model on hugging face can be baked.

### Show me the code

This repo focuses on generating a baked representation of the GPT2 model and evaluating performance of the baked model compared to standard inference using the transformers library.

The following layers are involved in reproducing the GPT2 model
 - [Layer Normalization](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)
 - [Embedding Lookup Table](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html)
 - [Attention](https://www.youtube.com/watch?v=ekg-hoob0SM&list=PLTl9hO2Oobd97qfWC40gOSU8C0iu0m2l4&index=7)
 - [Conv1d](https://arxiv.org/abs/1511.06434)
 - [Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)

 For each layer, we have created a generation template backed by a generator that consumes
 the layer params like weights, sizes etc and generates c code for the layer.
 The generated sourced in the generated directory can be viewed to see samples of generated source based on test weights. The templates are housed in the generation_templates and generator directory.

 We leverage custom macros in the templates to bake model weights in source, unroll loops
 and explode matrix multiplication operations. Not all optimization may be necessary but we are adopting "if it is easy to do, do it" strategy.

 Step through the code in the generation template and the corresponding generator to understand how the code is generated.

 ### Does it work?

 Yes, it works. The validation directory contains python scripts used to generate
 output for each of the above layers based on the standard PyTorch implementation.
 The output of the baked source has been validated to match the output of the validation scripts.

 ### Challenges

 The main challenge is compiling the generated source. The attention layer when baked using actual weights from GPT2 yields 16k lines of c code. Compiling this source takes too much memory and is not possible on a Macbook Air.

 To overcome this we plan to generate source, compile and run benchmarks on cloud hosted machines which have a decent amount of RAM (atleast 32GiB).
 Note that this memory is only required for compiling the source, during inference the memory used should be comparable to the amount used during inference with Pytorch.

 ## Benchmark Results

 Benchmarks will be carrried out in two stages, benchmarks for individual layers on inputs with
 the same dimensions used in GPT2 followed by end to end benchmark that evaluates time taken to complete one generation cycle by calling `model.generate()`.

 Benchmarks have been recorded by instrumenting the scripts in the validation directory with `time.time()` calls to record the time taken to only execute layer transformation excluding setup code. Numbers below are the average of five runs.

 **NOTE: Ensure the device you are running on has no form of hardware acceleration like GPU or the results will be skewed**

| Layer               | Input Shape   | PyTorch (ms) | Kaoken (ms) |
|---------------------|---------------|--------------|-------------|
| Layer Normalization | [1, 768]      | 0.052        | 0.0052      |
| GELU Activation     | [1, 768]      | 0.174        | 0.084       |
| Linear              | [768, 50257]  | 22.4         | 2.378       |
| Attention           | [1, 768]      | 14.484       | 9.386       |
| MLP                 | [3072, 768]   | 32.51        |             |

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
 - read embedding weights by memory mapping file fom disk
 - get inference working using binary built from c files
 - generalize generation of c files and building binary for any model