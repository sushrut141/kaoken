from typing import List

import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

OUTPUT_DIR = "./generated"

ATTENTION_INPUT_EMBEDDING = [0.1 for _ in range(768)]
ATTENTION_INPUT_SEQUENCE = [ATTENTION_INPUT_EMBEDDING for _ in range(4)]

def replace_tokens(bag_of_tokens, line):
    regexes = [
        (r"{c_attn_weight}\[(\d+)\]\[(\d+)\]", "{c_attn_weight}"),
        (r"{c_proj_weight}\[(\d+)\]\[(\d+)\]", "{c_proj_weight}"),
        (r"{c_attn_bias}\[(\d+)\]", "{c_attn_bias}"),
        (r"{c_proj_bias}\[(\d+)\]", "{c_proj_bias}")
    ]
    temp = line
    for tup in regexes:
        regex, token = tup
        match = re.search(regex, temp)
        if match:
            is_two_dimensional = len(match.groups()) == 2
            first_index = match.group(1)
            if is_two_dimensional:
                second_index = match.group(2)
                key = f"{token}[{first_index}][{second_index}]"
            else:
                key = f"{token}[{first_index}]"
            temp = temp.replace(key, bag_of_tokens[key])
    return temp


# Generates source for multiplying matrices supplied in the macro
# // MAT_MULTIPLY a=input,b={C_ATTN_WEIGHTS},c=output r=4,c=5,inner=6
def generate_mat_multiply_source(input: str, bag_of_tokens) -> str:
    temp = input.strip().split(' ')
    assert len(temp) == 4
    assert temp[1] == 'MAT_MULTIPLY'

    params = {}
    for i, tup in enumerate(temp[2].split(',')):
        key, val = tup.split('=')
        params[key.strip()] = val.strip()
    assert len(params) == 3, " Format must be a=left_matrix,b=right_matrix,c=output_matrix"
    assert 'a' in params, " Format must be a=left_matrix,b=right_matrix,c=output_matrix"
    assert 'b' in params, " Format must be a=left_matrix,b=right_matrix,c=output_matrix"
    assert 'c' in params, " Format must be a=left_matrix,b=right_matrix,c=output_matrix"
    
    ranges = {}
    for i, tup in enumerate(temp[3].split(',')):
        key, val = tup.split('=')
        ranges[key.strip()] = int(val.strip())
    assert len(params) == 3, " Format must be r=4,c=5,inner=6"
    assert 'r' in ranges, " Format must be r=4,c=5,inner=6"
    assert 'c' in ranges, " Format must be r=4,c=5,inner=6"
    assert 'inner' in ranges, " Format must be r=4,c=5,inner=6"

    source = ''
    for i in range(ranges['r']):
        for j in range(ranges['c']):
            source += f"{params['c']}[{i}][{j}] = "
            inner = []
            for k in range(ranges['inner']):
                line = f"({params['a']}[{i}][{k}] * {params['b']}[{k}][{j}])"
                line = replace_tokens(bag_of_tokens, line)
                inner.append(line)
            source += ' + '.join(inner) + ';' + '\n'
    return source

def populate_2d_weights_in_bag(bag, weights, token):
    for i in range(len(weights)):
        for j in range(len(weights[0])):
            key = f"{token}[{i}][{j}]"
            bag[key] = str(weights[i][j])

def populate_1d_weights_in_bag(bag, weights, token):
    for i in range(len(weights)):
        key = f"{token}[{i}]"
        bag[key] = str(weights[i])

def generate_attention(
        name: str,
        c_attn_weight: List[List[float]],
        c_attn_bias: List[float],
        c_proj_weight: List[List[float]],
        c_proj_bias: List[float],
        num_heads: int,
        sequence_length: int,
        embedding_size: int):
    assert len(c_attn_weight) > 0
    assert len(c_attn_weight[0]) == 3 * embedding_size

    output_file_path = f"{OUTPUT_DIR}/{name}_attention.c"
    template_path = "./generation_templates/attention.c.template"

    # Populate bag of tokens for inlining weights
    bag_of_tokens = {}
    populate_2d_weights_in_bag(bag_of_tokens, c_attn_weight, '{c_attn_weight}')
    populate_2d_weights_in_bag(bag_of_tokens, c_proj_weight, '{c_proj_weight}')
    populate_1d_weights_in_bag(bag_of_tokens, c_attn_bias, '{c_attn_bias}')
    populate_1d_weights_in_bag(bag_of_tokens, c_proj_bias, '{c_proj_bias}')

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

        lines = implementation.split("\n")

        i = 0
        source = ""
        size = len(lines)
        # TODO - reuse generation logic across layers
        while i < size:
            line = lines[i]
            print("processing line ", line)

            if "UNROLL_START" in line:
                unroll_end = None
                # unroll expression
                expr = []
                # capture the expression to unroll
                for j in range(i + 1, size):
                    if "UNROLL_END" in lines[j]:
                        unroll_end = j
                        break
                    else:
                        expr.append(lines[j])
                assert unroll_end != None, "Unroll must be closed"
                assert len(expr) > 0, "Unroll block cannot be empty"

                expr = "\n".join(expr)

                # capture params to use in unroll
                parts = [val.strip() for val in re.split(r'\s+', line) if len(val.strip()) > 0]
                assert len(parts) == 4, "Unroll block must be formatted as // UNROLL_START start=0,end=100 param1=input+idx,param2=input2+idx+1"

                start = int(eval(parts[2][(parts[2].index("start=") + 6):parts[2].index(",")]))
                end = int(eval(parts[2][(parts[2].index("end=") + 4):]))

                params = parts[3].split(",")
                params_map = {}
                for param_tup in params:
                    assert '=' in param_tup, "Unroll params must be key=value pairs"
                    param_name, value = param_tup.split('=')
                    params_map[param_name] = value

                # unroll expression and interpolate values
                for j in range(start, end):
                    unroll_block = expr
                    for key in params_map:
                        value = params_map[key]
                        value = value.replace("idx", str(j))
                        unroll_block = unroll_block.replace(key, value)
                        unroll_block = replace_tokens(bag_of_tokens, unroll_block)
                    source += unroll_block
                    source += '\n'
                i = unroll_end + 1
            else:
                source += line
                source += '\n'
                i += 1
        
        temp = source.split('\n')
        source = ''
        for line in temp:
            print("post processing line ", line)
            if "MAT_MULTIPLY" in line:
                source += generate_mat_multiply_source(line, bag_of_tokens)
            else:
                source += line + '\n'

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
        embedding_size=768
    )

if __name__ == "__main__":
    generate_attention(
        name="gpt2_attention_0",
        c_attn_weight = [
            [0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4],
            [0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4],
            [0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4],
            [0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4]
        ],
        c_attn_bias = [0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4],
        c_proj_weight = [
            [0.1, 0.2, 0.3, 0.4],
            [0.1, 0.2, 0.3, 0.4],
            [0.1, 0.2, 0.3, 0.4],
            [0.1, 0.2, 0.3, 0.4]
        ],
        c_proj_bias = [0.1, 0.2, 0.3, 0.4],
        num_heads=12,
        sequence_length=4,
        embedding_size=4
    )
    # generate_attention_from_gpt2_pretrained()
