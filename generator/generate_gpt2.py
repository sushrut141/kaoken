import json
import os

from generate_layer_normalization import generate_layer_normalization
from generate_attention import generate_attention
from generate_mlp import generate_mlp
from generate_embedding import generate_embedding
from generate_linear import generate_linear


def load_weights(path):
    f = open(f"./weights/{path}", 'r')
    return json.load(f)


# Reference for blocks: https://huggingface.co/transformers/v4.11.3/_modules/transformers/models/gpt2/modeling_gpt2.html
#
# <bound method GenerationMixin.generate of GPT2LMHeadModel(
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
def generate(
        sequence_length: int,
        num_of_blocks: int, 
        num_of_heads: int):
    assert os.path.isdir('./weights'), "Layer weights must be present in the weights directory. Try running generate_specification.py with DUMP_WEIGHTS=True."

    generate_embedding(
        name="transformer.wte.weight",
        embeddings=load_weights("transformer.wte.weight.json")
    )
    generate_embedding(
        name="transformer.wpe.weight",
        embeddings=load_weights("transformer.wpe.weight.json")
    )
    generate_linear(
        name="lm_head",
        linear_weights=load_weights("lm_head.weight.json"),
        sequence_length=sequence_length,
        embedding_size=768
    )

    for i in range(num_of_blocks):
        layer_norm_1 = f"transformer.h.{i}.ln_1"
        layer_norm_2 = f"transformer.h.{i}.ln_2"
        attn = f"transformer.h.{i}.attn"
        c_attn = f"transformer.h.{i}.attn.c_attn"
        c_proj = f"transformer.h.{i}.attn.c_proj"
        mlp = f"transformer.h.{i}.mlp"
        c_mlp_fc = f"transformer.h.{i}.mlp.c_fc"
        c_mlp_proj = f"transformer.h.{i}.mlp.c_proj"

        generate_layer_normalization(
            name=layer_norm_1,
            weights=load_weights(f"{layer_norm_1}.weight.json"),
            bias=load_weights(f"{layer_norm_1}.bias.json")
        )
        generate_layer_normalization(
            name=layer_norm_2,
            weights=load_weights(f"{layer_norm_2}.weight.json"),
            bias=load_weights(f"{layer_norm_2}.bias.json")
        )

        generate_attention(
            name=attn,
            c_attn_weight=load_weights(f"{c_attn}.weight.json"),
            c_attn_bias=load_weights(f"{c_attn}.bias.json"),
            c_proj_weight=load_weights(f"{c_proj}.weight.json"),
            c_proj_bias=load_weights(f"{c_proj}.bias.json"),
            num_heads=num_of_heads,
            sequence_length=sequence_length
        )
        generate_mlp(
            name=mlp,
            c_fc_weight=load_weights(f"{c_mlp_fc}.weight.json"),
            c_fc_bias=load_weights(f"{c_mlp_fc}.bias.json"),
            c_proj_weight=load_weights(f"{c_mlp_proj}.weight.json"),
            c_proj_bias=load_weights(f"{c_mlp_proj}.bias.json"),
            sequence_length=sequence_length
        )
        

if __name__ == "__main__":
    generate(
        sequence_length=4,
        num_of_blocks=1,
        num_of_heads=1
    )