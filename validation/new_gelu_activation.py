import time
import torch
import math
from torch import nn


class NewGELUActivation(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    

if __name__ == "__main__":
    input = torch.tensor([0.1 for _ in range(768)])
    activation = NewGELUActivation()
    start_time = time.time()
    output = activation(input)
    end_time = time.time()
    execution_time_ms = (end_time - start_time) * 1000
    print("Layer Normalization Execution time:", execution_time_ms, "milli seconds")
    print(output.tolist()[:10])