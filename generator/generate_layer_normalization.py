from typing import List

import re
import utils

OUTPUT_DIR = "./generated"


def generate_layer_normalization(
        name: str, weights: List[float], bias: List[float], size: int, generate_test_main: bool):
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

        source = utils.unroll_loops(implementation)

        if not generate_test_main :
            source = utils.cleanup_test_block(source)

        with open(output_file_path, 'w+') as of:
            of.write(source)

if __name__ == "__main__":
    generate_layer_normalization(
        name= "gpt2_layer_norm_1",
        weights=[0.1, 0.1, 0.1, 0.1],
        bias=[0.1, 0.1, 0.1, 0.1],
        size=4,
        generate_test_main=True
    )
