import torch
from torch import nn


def get_input_tensor(r, c):
    return torch.tensor([
        [
            1.0 * (i + 1) for i in range(c)
        ] for _ in range(r)
    ])

def get_linear_weights_tensor(r, c):
    return torch.tensor([
        [
            0.1 * (i + 1) for i in range(c)
        ] for _ in range(r)
    ])


def compute_linear_transformation(input, in_features, out_features):
    linear = nn.Linear(in_features, out_features, bias=False)
    linear.weight.data = get_linear_weights_tensor(out_features, in_features)
    return linear(input)

if __name__ == "__main__":
    input = get_input_tensor(1, 4)
    
    output = compute_linear_transformation(input, 4, 8)
    print(output.tolist())