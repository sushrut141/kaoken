import torch
from torch import nn
import math

def get_conv1d_weights(nx, nf):
    return [
        [
            (i* 0.001) for i in range(nf)
        ] for _ in range(nx)
    ]
class Conv1D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

    Basically works like a linear layer but the weights are transposed.

    Args:
        nf (:obj:`int`): The number of output features.
        nx (:obj:`int`): The number of input features.
    """

    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        # hard coding weights for deterministic tests.
        w = torch.tensor(get_conv1d_weights(nx, nf))
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x
    
class NewGELUActivation(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    

class GPT2MLP(nn.Module):
    def __init__(self, intermediate_size, embed_dim):
        super().__init__()
        self.c_fc = Conv1D(intermediate_size, embed_dim)
        self.c_proj = Conv1D(embed_dim, intermediate_size)
        self.act = NewGELUActivation()

    def forward(self, hidden_states) -> torch.FloatTensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        return hidden_states


def create_mlp():
    mlp = GPT2MLP(16, 4)
    return mlp


# Generates input output pair for the GPT2MLP layer that
# can be used to validate output of baked attention layer.
def generate_base_output(attention):
    sequence = torch.tensor(
        # Batch Size 1
        [
            # Sequence Length 1
            [
                # Embedding Size 4
                [1.0, 2.0, 3.0, 4.0]
            ]
        ]
    )
    output = attention.forward(sequence)
    return (sequence.tolist(), output)

if __name__ == "__main__":
    attention = create_mlp()

    input, output = generate_base_output(attention)
    print('Input', input)
    print('Output', output[0].tolist())