from typing import Mapping, Union

import re

# List of regexes to check for when processing source.
REGEXES = [
    (r"{c_attn_weight}\[(\d+)\]\[(\d+)\]", "{c_attn_weight}"),
    (r"{c_proj_weight}\[(\d+)\]\[(\d+)\]", "{c_proj_weight}"),
    (r"{linear_weights}\[(\d+)]\[(\d+)\]", "{linear_weights}"),
    (r"{c_attn_bias}\[(\d+)\]", "{c_attn_bias}"),
    (r"{c_proj_bias}\[(\d+)\]", "{c_proj_bias}")
]

def populate_2d_weights_in_bag(bag, weights, token):
    """
    Associate each value in the 2d array with a key that 
    represents 2d array access like `arr[i][j]`.
    """
    for i in range(len(weights)):
        for j in range(len(weights[0])):
            key = f"{token}[{i}][{j}]"
            bag[key] = str(weights[i][j])

def populate_1d_weights_in_bag(bag, weights, token):
    """
    Associate each value in the 1d array with a key that 
    represents 1d array access like `arr[i]`.
    """
    for i in range(len(weights)):
        key = f"{token}[{i}]"
        bag[key] = str(weights[i])


def replace_tokens(bag_of_tokens, line):
    """
    Replace array access tokens in the generated source with 
    actual weights.
    """
    temp = line
    for tup in REGEXES:
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


def generate_mat_multiply_source(input: str, bag_of_tokens) -> str:
    """
    Generates source for multiplying matrices supplied in the macro.
    // MAT_MULTIPLY a=input,b={C_ATTN_WEIGHTS},c=output r=4,c=5,inner=6

    The value is computed as c = a * b.transpose

    The matrix multiplication is exploded to populate each array cell as sum
    of row / col multiplications.
    """
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

def generate_mat_multiply_transpose_source(input: str, bag_of_tokens) -> str:
    """
    Generates source for multiplying matrices supplied in the macro but transposing the
    second marox before multiplication.

    The value is computed as c = a * b.transpose

    // MAT_MULTIPLY_TRANSPOSE a=input,b={C_ATTN_WEIGHTS},c=output r=4,c=5,inner=6

    The matrix multiplication is exploded to populate each array cell as sum
    of row / col multiplications.
    """
    temp = input.strip().split(' ')
    assert len(temp) == 4
    assert temp[1] == 'MAT_MULTIPLY_TRANSPOSE'

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
                line = f"({params['a']}[{i}][{k}] * {params['b']}[{i}][{k}])"
                line = replace_tokens(bag_of_tokens, line)
                inner.append(line)
            source += ' + '.join(inner) + ';' + '\n'
    return source


def generate_mat_row_add_vec1D(line: str, bag_of_tokens: Mapping[str, str]) -> str:
    """
    Explodes addition of a vector to each row in a matrix to individual addition operations.
    """
    temp = line.strip().split(' ')
    assert len(temp) == 4
    assert temp[1] == 'MAT_ROW_ADD_VEC1D'

    params = {}
    for _, tup in enumerate(temp[2].split(',')):
        key, val = tup.split('=')
        params[key.strip()] = val.strip()
    assert len(params) == 2, " Format must be a=matrix,b=vector"
    assert 'a' in params, " Format must be a=matrix,b=vector"
    assert 'b' in params, " Format must be a=matrix,b=vector"

    ranges = {}
    for _, tup in enumerate(temp[3].split(',')):
        key, val = tup.split('=')
        ranges[key.strip()] = int(val.strip())
    assert len(params) == 2, " Format must be r=4,c=5"
    assert 'r' in ranges, " Format must be r=4,c=5"
    assert 'c' in ranges, " Format must be r=4,c=5"

    source = ''
    for i in range(ranges['r']):
        for j in range(ranges['c']):
            line = f"{params['a']}[{i}][{j}] += {params['b']}[{j}];\n"
            line = replace_tokens(bag_of_tokens, line)
            source += line
    return source


def unroll_loops(input: str, bag_of_tokens: Mapping[str, Union[str, float]] = {}) -> str:
    """
    Processes the supplied input source and unrolls all loops and
    generate individual instructions for content within each
    // UNROLL_START / UNROLL_END blocks.

    Tokens such as {token}[i][j] will be reolaced with the value from bag of tokens
    if the {token} regex is present in the list. 
    """
    lines = input.split("\n")
    i = 0
    source = ""
    size = len(lines)
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
                    unroll_block = replace_tokens(bag_of_tokens, unroll_block)
                source += unroll_block
                source += '\n'
            i = unroll_end + 1
        else:
            source += line
            source += '\n'
            i += 1
    return source

def cleanup_test_block(source: str) -> str:
    """
    Find the testing block in the source and removes it.
    Testing block is code added n=between // START_TEST / END_TEST block.
    Only one testing block can exist within the source.
    """
    if '// START_TEST' in source:
        assert '// END_TEST' in source, "Testing block must be closed with // END_TEST"
        test_start_idx = source.index('// START_TEST')
        test_end_idx = source.index('// END_TEST') + len('// END_TEST')
        source = source[:test_start_idx] + source[test_end_idx+1:]
    return source