from typing import List

import utils

OUTPUT_DIR = "./generated"


def generate_mlp(
        name: str,
        c_fc_weight: List[List[float]],
        c_fc_bias: List[float],
        c_proj_weight: List[List[float]],
        c_proj_bias: List[float],
        sequence_length: int,
        generate_test_main: bool = False):
    """
    Generates source code for MLP layer.
    """
    assert len(c_fc_weight) > 0
    assert len(c_proj_weight) > 0
    assert len(c_fc_weight[0]) == len(c_fc_bias)
    assert len(c_proj_weight[0]) == len(c_proj_bias)

    embedding_size = len(c_fc_weight)

    output_file_path = f"{OUTPUT_DIR}/{name}_mlp.c"
    template_path = "./generation_templates/mlp.c.template"

    bag_of_tokens = {}
    utils.populate_2d_weights_in_bag(bag_of_tokens, c_fc_weight, '{c_fc_weight}')
    utils.populate_2d_weights_in_bag(bag_of_tokens, c_proj_weight, '{c_proj_weight}')
    utils.populate_1d_weights_in_bag(bag_of_tokens, c_fc_bias, '{c_fc_bias}')
    utils.populate_1d_weights_in_bag(bag_of_tokens, c_proj_bias, '{c_proj_bias}')

    with open(template_path) as template:
        template_text = template.read()

        implementation = template_text[template_text.index("// START_METHODS"):template_text.index("// END_METHODS")]
        # replace placeholder variables
        implementation = implementation.replace("{NAME}", name)
        implementation = implementation.replace("{SEQUENCE_LENGTH}", str(sequence_length))
        implementation = implementation.replace("{EMBEDDING_SIZE}", str(embedding_size))

        source = utils.unroll_loops(implementation, bag_of_tokens)
        
        temp = source.split('\n')
        source = ''
        for line in temp:
            print("post processing line ", line)
            if "MAT_MULTIPLY_TRANSPOSE" in line:
                source += utils.generate_mat_multiply_transpose_source(line, bag_of_tokens)
            elif "MAT_ROW_ADD_VEC1D" in line:
                source += utils.generate_mat_row_add_vec1D(line, bag_of_tokens)
            elif "MAT_MULTIPLY" in line:
                source += utils.generate_mat_multiply_source(line, bag_of_tokens)
            else:
                source += line + '\n'
        
        if not generate_test_main :
            source = utils.cleanup_test_block(source)

        with open(output_file_path, 'w+') as of:
            of.write(source)


def get_conv1d_weights(nx, nf):
    return [
        [
            (i* 0.001) for i in range(nf)
        ] for _ in range(nx)
    ]

if __name__ == "__main__":
    generate_mlp(
        name="gpt2_mlp",
        c_fc_weight=get_conv1d_weights(4, 16),
        c_fc_bias=[0 for _ in range(16)],
        c_proj_weight=get_conv1d_weights(16, 4),
        c_proj_bias=[0 for _ in range(4)],
        sequence_length=4,
        generate_test_main = True
    )
