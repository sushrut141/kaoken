from typing import List

import re

OUTPUT_DIR = "./generated"


def generate_gelu_activation(
        name: str, size: int, generate_test_main: bool):
    """
    Generates source code for baked layer normalization.
    """

    output_file_path = f"{OUTPUT_DIR}/{name}_new_gelu_activation.c"
    template_path = "./generation_templates/new_gelu_activation.c.template"

    with open(template_path) as template:
        template_text = template.read()
        
      

        implementation = template_text[template_text.index("// START_METHODS"):template_text.index("// END_METHODS")]
        # replace placeholder variables
        implementation = implementation.replace("{SIZE}", str(size)).replace("{NAME}", name)
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
        if not generate_test_main and '// START_TEST' in source:
            assert '// END_TEST' in source, "Testing block must be closed with // END_TEST"
            test_start_idx = source.index('// START_TEST') - 1
            test_end_idx = source.index('// END_TEST') + len('// END_TEST')
            source = source[:test_start_idx] + source[test_end_idx+1:]
        with open(output_file_path, 'w+') as of:
            of.write(source)

if __name__ == "__main__":
    generate_gelu_activation(
        name= "gpt2_gelu_1",
        size=4,
        generate_test_main=True
    )
