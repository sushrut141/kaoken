from typing import List

import re
import utils

OUTPUT_DIR = "./generated"


def generate_gelu_activation(
        name: str, size: int, generate_test_main: bool = False):
    """
    Generates source code for GELU activation.
    """

    output_file_path = f"{OUTPUT_DIR}/{name}_new_gelu_activation.c"
    template_path = "./generation_templates/new_gelu_activation.c.template"

    with open(template_path) as template:
        template_text = template.read()
        
      

        implementation = template_text[template_text.index("// START_METHODS"):template_text.index("// END_METHODS")]
        # replace placeholder variables
        implementation = implementation.replace("{SIZE}", str(size)).replace("{NAME}", name)
        
        source = utils.unroll_loops(implementation)

        if not generate_test_main :
            source = utils.cleanup_test_block(source)

        with open(output_file_path, 'w+') as of:
            of.write(source)

if __name__ == "__main__":
    generate_gelu_activation(
        name= "gpt2_gelu_1",
        size=4,
        generate_test_main=True
    )
