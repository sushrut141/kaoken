from typing import List, Mapping

import os
import re
import torch
import utils
from transformers import AutoTokenizer, AutoModelForCausalLM

OUTPUT_DIR = "./generated"

ATTENTION_INPUT_EMBEDDING = [0.1 for _ in range(768)]
ATTENTION_INPUT_SEQUENCE = [ATTENTION_INPUT_EMBEDDING for _ in range(4)]


def generate_attention(
        name: str,
        c_attn_weight: List[List[float]],
        c_attn_bias: List[float],
        c_proj_weight: List[List[float]],
        c_proj_bias: List[float],
        num_heads: int,
        sequence_length: int,
        generate_test_main: bool = False):
    assert len(c_attn_weight) > 0
    assert len(c_attn_weight[0]) == 3 * len(c_attn_weight)
    embedding_size = len(c_attn_weight)

    output_file_path = f"{OUTPUT_DIR}/{name}_attention.c"
    template_path = "./generation_templates/attention.c.template"

    # Populate bag of tokens for inlining weights
    bag_of_tokens = {}
    utils.populate_2d_weights_in_bag(bag_of_tokens, c_attn_weight, '{c_attn_weight}')
    utils.populate_2d_weights_in_bag(bag_of_tokens, c_proj_weight, '{c_proj_weight}')
    utils.populate_1d_weights_in_bag(bag_of_tokens, c_attn_bias, '{c_attn_bias}')
    utils.populate_1d_weights_in_bag(bag_of_tokens, c_proj_bias, '{c_proj_bias}')

    with open(template_path) as template:
        template_text = template.read()

        implementation = template_text[template_text.index("// START_METHODS"):template_text.index("// END_METHODS")]
        # replace placeholder variables
        implementation = implementation.replace("{NAME}", name)
        implementation = implementation.replace("{EMBEDDING_SIZE}", str(embedding_size))
        implementation = implementation.replace("{NUM_HEADS}", str(num_heads))
        implementation = implementation.replace("{SEQUENCE_LENGTH}", str(sequence_length))
        implementation = implementation.replace("{C_ATTN_EMBEDDING_SIZE_ROW}", str(len(c_attn_weight)))
        implementation = implementation.replace("{C_ATTN_EMBEDDING_SIZE_COL}", str(len(c_attn_bias)))
        implementation = implementation.replace("{C_PROJ_EMBEDDING_SIZE_ROW}", str(len(c_proj_weight)))
        implementation = implementation.replace("{C_PROJ_EMBEDDING_SIZE_COL}", str(len(c_proj_bias)))

        source = utils.unroll_loops(implementation, bag_of_tokens)
        
        temp = source.split('\n')
        source = ''
        for line in temp:
            print("post processing line ", line)
            if "MAT_MULTIPLY" in line:
                source += utils.generate_mat_multiply_source(line, bag_of_tokens)
            elif "MAT_ROW_ADD_VEC1D" in line:
                source += utils.generate_mat_row_add_vec1D(line, bag_of_tokens)
            else:
                source += line + '\n'
        
        if not generate_test_main :
            source = utils.cleanup_test_block(source)

        with open(output_file_path, 'w+') as of:
            of.write(source)

def generate_attention_from_gpt2_pretrained():
    assert os.path.isdir('../gpt2'), "Pretrained model must be present in gpt2 directory in parent."
    model = AutoModelForCausalLM.from_pretrained('../gpt2', local_files_only = True)
    model_top_config = next(model.named_modules())[1]
    attention_layers = {
        'transformer.h.0.attn.c_attn.weight': 'c_attn_weight',
        'transformer.h.0.attn.c_attn.bias': 'c_attn_bias',
        'transformer.h.0.attn.c_proj.weight': 'c_proj_weight',
        'transformer.h.0.attn.c_proj.bias': 'c_proj_bias'
    }
    obj = {}
    for param_name, param in model_top_config.named_parameters():
        if param_name in attention_layers:
            key = attention_layers[param_name]
            obj[key] = param.tolist()
    
    generate_attention(
        name="gpt2_attention_0",
        **obj,
        num_heads=12,
        sequence_length=4,
        generate_test_main= False
    )

def get_conv1d_weights(nx, nf):
    return [
        [
            (i* 0.001) for i in range(nf)
        ] for _ in range(nx)
    ]

if __name__ == "__main__":
    c_attn_weight = get_conv1d_weights(8, 24)
    c_proj_weight = get_conv1d_weights(8, 24)

    generate_attention(
        name="gpt2_attention_0",
        c_attn_weight = c_attn_weight,
        c_attn_bias = torch.zeros(24).tolist(),
        c_proj_weight = c_proj_weight,
        c_proj_bias = torch.zeros(4).tolist(),
        num_heads=2,
        sequence_length=1,
        generate_test_main= True
    )
    # generate_attention_from_gpt2_pretrained()
