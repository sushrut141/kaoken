from typing import List

import re
import utils

OUTPUT_DIR = "./generated"


def generate_linear(
        name: str,
        linear_weights: List[List[float]],
        sequence_length: int,
        out_features: int,
        embedding_size: int,
        generate_test_main: bool):
    """
    Generates source code for linear transformation layer.
    """
    assert len(linear_weights) > 0
    assert len(linear_weights) == out_features

    output_file_path = f"{OUTPUT_DIR}/{name}_linear.c"
    template_path = "./generation_templates/linear.c.template"

    bag_of_tokens = {}
    utils.populate_2d_weights_in_bag(bag_of_tokens, linear_weights, '{linear_weights}')

    with open(template_path) as template:
        template_text = template.read()

        implementation = template_text[template_text.index("// START_METHODS"):template_text.index("// END_METHODS")]
        # replace placeholder variables
        implementation = implementation.replace("{NAME}", name)
        implementation = implementation.replace("{SEQUENCE_LENGTH}", str(sequence_length))
        implementation = implementation.replace("{LINEAR_OUT_FEATURES}", str(out_features))
        implementation = implementation.replace("{EMBEDDING_SIZE}", str(embedding_size))

        source = utils.unroll_loops(implementation, bag_of_tokens)
        
        temp = source.split('\n')
        source = ''
        for line in temp:
            print("post processing line ", line)
            if "MAT_MULTIPLY_TRANSPOSE" in line:
                source += utils.generate_mat_multiply_transpose_source(line, bag_of_tokens)
            elif "MAT_MULTIPLY" in line:
                source += utils.generate_mat_multiply_source(line, bag_of_tokens)
            else:
                source += line + '\n'
        
        if not generate_test_main :
            source = utils.cleanup_test_block(source)

        with open(output_file_path, 'w+') as of:
            of.write(source)


if __name__ == "__main__":
    generate_linear(
        name="gpt2_linear",
        linear_weights=[
            [0.1, 0.2, 0.3, 0.4],
            [0.1, 0.2, 0.3, 0.4],
            [0.1, 0.2, 0.3, 0.4],
            [0.1, 0.2, 0.3, 0.4],
            [0.1, 0.2, 0.3, 0.4],
            [0.1, 0.2, 0.3, 0.4],
            [0.1, 0.2, 0.3, 0.4],
            [0.1, 0.2, 0.3, 0.4],
        ],
        sequence_length=1,
        embedding_size=4,
        out_features=8,
        generate_test_main = True
    )
