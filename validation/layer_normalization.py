import time
import torch
from torch import nn

if __name__ == "__main__":
    input = torch.tensor([(0.1 * (i + 1)) for i in range(768)])
    weights = torch.tensor([(0.1 * i) for i in range(768)])
    bias = torch.tensor([(0.2 * i) for i in range(768)])

    layer_norm = nn.LayerNorm(input.shape)
    layer_norm.weight.data = weights
    layer_norm.bias.data = bias

    start_time = time.time()
    output = layer_norm(input)
    end_time = time.time()

    execution_time_ms = (end_time - start_time) * 1000
    print("Layer Normalization Execution time:", execution_time_ms, "milli seconds")
    # print(output.tolist()[:10])