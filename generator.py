from typing import List

import re

OUTPUT_DIR = "./generated"

def generate_layer_normalization(
        name: str, weights: List[float], bias: List[float], size: int):
    """
    Generates source code for baked layer normalization.
    """
    assert len(weights) == size
    assert len(bias) == size
    output_file_path = f"{OUTPUT_DIR}/{name}_layer_norm.c"
    template_path = "./generation_templates/layer_norm.c.template"

    with open(template_path) as template:
        template_text = template.read()
        
        # inject constants into source
        weights_str = '{' + ', '.join(map(str, weights)) + '}'
        bias_str = '{' + ', '.join(map(str, bias)) + '}'

        implementation = template_text[template_text.index("// START_METHODS"):template_text.index("// END_METHODS")]
        # replace placeholder variables
        implementation = implementation.replace("{SIZE}", str(size)).replace("{NAME}", name)
        implementation = implementation.replace("{WEIGHTS}", weights_str).replace("{BIAS}", bias_str)

        lines = implementation.split("\n")

        i = 0
        source = ""
        size = len(lines)
        # TODO - reuse generation logic across layers
        while i < size:
            line = lines[i]

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
                    source += unroll_block
                    source += '\n'
                i = unroll_end + 1
            else:
                source += line
                source += '\n'
                i += 1
        with open(output_file_path, 'w+') as of:
            of.write(source)

def generate_embedding(
        name: str, embeddings: List[List[float]], vocab_size: int, embedding_size: int):
    """
    Generates source code for embedding lookup layer.
    """
    assert len(embeddings) > 0
    assert len(embeddings) == vocab_size
    assert len(embeddings[0]) == embedding_size
    output_file_path = f"{OUTPUT_DIR}/{name}_embedding.c"
    template_path = "./generation_templates/embedding.c.template"

    with open(template_path) as template:
        template_text = template.read()

        implementation = template_text[template_text.index("// START_METHODS"):template_text.index("// END_METHODS")]
        # replace placeholder variables
        implementation = implementation.replace("{NAME}", name)
        implementation = implementation.replace("{EMBEDDING_SIZE}", str(embedding_size)).replace("{VOCAB_SIZE}", str(vocab_size))

        lines = implementation.split("\n")

        i = 0
        source = ""
        size = len(lines)
        # TODO - reuse generation logic across layers
        while i < size:
            line = lines[i]

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
                    source += unroll_block
                    source += '\n'
                i = unroll_end + 1
            else:
                source += line
                source += '\n'
                i += 1
        
        # replace all the embedding variables with actual values
        for idx, embedding in enumerate(embeddings):
            placeholder = "{EMBEDDING_" + str(idx) + "}"
            embedding_str = '{' + ', '.join(map(str, embedding)) + '}'
            source = source.replace(placeholder, embedding_str);


        with open(output_file_path, 'w+') as of:
            of.write(source)

def generate_attention(
        name: str,
        c_attn_weights: List[List[float]],
        c_attn_bias: List[float],
        c_proj_weights: List[List[float]],
        c_proj_bias: List[float],
        num_heads: int,
        sequence_length: int,
        embedding_size: int):
    assert len(c_attn_weights) > 0
    assert len(c_attn_weights[0]) == 3 * embedding_size

    output_file_path = f"{OUTPUT_DIR}/{name}_attention.c"
    template_path = "./generation_templates/attention.c.template"

    with open(template_path) as template:
        template_text = template.read()

        implementation = template_text[template_text.index("// START_METHODS"):template_text.index("// END_METHODS")]
        # replace placeholder variables
        implementation = implementation.replace("{NAME}", name)
        implementation = implementation.replace("{EMBEDDING_SIZE}", str(embedding_size))
        implementation = implementation.replace("{NUM_HEADS}", str(num_heads))
        implementation = implementation.replace("{SEQUENCE_LENGTH}", str(sequence_length))
        implementation = implementation.replace("{C_ATTN_EMBEDDING_SIZE_ROW}", str(len(c_attn_weights)))
        implementation = implementation.replace("{C_ATTN_EMBEDDING_SIZE_COL}", str(len(c_attn_bias)))
        implementation = implementation.replace("{C_PROJ_EMBEDDING_SIZE_ROW}", str(len(c_proj_weights)))
        implementation = implementation.replace("{C_PROJ_EMBEDDING_SIZE_COL}", str(len(c_proj_bias)))

        lines = implementation.split("\n")

        i = 0
        source = ""
        size = len(lines)
        # TODO - reuse generation logic across layers
        while i < size:
            line = lines[i]

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
                    source += unroll_block
                    source += '\n'
                i = unroll_end + 1
            else:
                source += line
                source += '\n'
                i += 1
        
        # replace all the embedding variables with actual values
        for idx, embedding in enumerate(c_attn_weights):
            placeholder = "{C_ATTN_EMBEDDING_" + str(idx) + "}"
            embedding_str = '{' + ', '.join(map(str, embedding)) + '}'
            source = source.replace(placeholder, embedding_str)
        for idx, embedding in enumerate(c_proj_weights):
            placeholder = "{C_PROJ_EMBEDDING_" + str(idx) + "}"
            embedding_str = '{' + ', '.join(map(str, embedding)) + '}'
            source = source.replace(placeholder, embedding_str)
        source = source.replace('{C_ATTN_BIAS}', '{' + ', '.join(map(str, c_attn_bias)) + '}')
        source = source.replace('{C_PROJ_BIAS}', '{' + ', '.join(map(str, c_proj_bias)) + '}')

        with open(output_file_path, 'w+') as of:
            of.write(source)

if __name__ == "__main__":
    # generate_layer_normalization(
    #     name= "gpt2_layer_norm_1",
    #     weights=[0.1, 0.2, 0.3, 0.4],
    #     bias=[0.20, 0.12, 0.77, 0.56],
    #     size=4
    # )
    # generate_embedding(
    #     name="gpt2_layer_embedding",
    #     embeddings=[
    #         [0.1, 0.2, 0.3, 0.4],
    #         [0.1, 0.2, 0.3, 0.4],
    #         [0.1, 0.2, 0.3, 0.4],
    #         [0.1, 0.2, 0.3, 0.4]
    #     ],
    #     vocab_size=4,
    #     embedding_size=4
    # )
    generate_attention(
        name="gpt2_attention_0",
        c_attn_weights = [
            [0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4],
            [0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4],
            [0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4],
            [0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4]
        ],
        c_attn_bias = [0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4],
        c_proj_weights = [
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

