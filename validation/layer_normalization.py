import torch
from torch import nn

def compute_layer_norm(tensor, weights, bias):
    layer_norm = nn.LayerNorm(tensor.shape)
    layer_norm.weight.data = weights
    layer_norm.bias.data = bias

    return layer_norm(tensor)


if __name__ == "__main__":
    input = torch.tensor([0.1, 0.2, 0.3, 0.4])
    weights = torch.tensor([0.1, 0.1, 0.1, 0.1])
    bias = torch.tensor([0.1, 0.1, 0.1, 0.1])
    output = compute_layer_norm(input, weights, bias)
    print(output.tolist())