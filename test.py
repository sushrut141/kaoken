import json
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

ATTENTION_INPUT_EMBEDDING = [0.1 for _ in range(768)]
ATTENTION_INPUT_SEQUENCE = [ATTENTION_INPUT_EMBEDDING for _ in range(4)]

def compute_attention_output(name, module):
    if name == 'transformer.h.0.attn':
        # [1, 4, 768] -> b, s, w
        with_batch_dimension = [ATTENTION_INPUT_SEQUENCE]
        input = torch.tensor(with_batch_dimension)
        output, _ = module.forward(input)
        output = output.tolist()[0]
        temp = {}
        temp['input'] = ATTENTION_INPUT_SEQUENCE
        temp['output'] = output
        temp = json.dumps(temp, indent=4)
        with open('./attention_0.json', 'w+') as f:
            f.write(temp)

def create_layer_config(model):
    """
    Creates a JSON configuration of all layers in the given model, including weights and operations.

    Args:
        model: The Hugging Face model object.

    Returns:
        A JSON string representing the layer configuration.
    """
    layer_config = []
    module_config = []
    for name, module in model.named_modules():
        compute_attention_output(name, module)
        temp = {}
        if name:
            temp[name] = {
                'name': name,
                'type': module.__class__.__name__,
                # "parameters": {param_name: param.shape.tolist() for param_name, param in module.named_parameters()}
            }
            module_config.append(temp)
    with open('./modules.json', 'w+') as f:
        f.write(json.dumps(module_config, indent=4))

    model_top_config = next(model.named_modules())[1]
    for param_name, param in model_top_config.named_parameters():
        obj = {}
        obj['name'] = param_name
        obj['dimensions'] = list(param.shape)
        layer_config.append(obj)
    return json.dumps(layer_config, indent=4)

# Reference for blocks: https://huggingface.co/transformers/v4.11.3/_modules/transformers/models/gpt2/modeling_gpt2.html
#
# <bound method GenerationMixin.generate of GPT2LMHeadModel(
#   (transformer): GPT2Model(
#     (wte): Embedding(50257, 768)
#     (wpe): Embedding(1024, 768)
#     (drop): Dropout(p=0.1, inplace=False)
#     (h): ModuleList(
#       (0-11): 12 x GPT2Block(
#         (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
#         (attn): GPT2Attention(
#           (c_attn): Conv1D()
#           (c_proj): Conv1D()
#           (attn_dropout): Dropout(p=0.1, inplace=False)
#           (resid_dropout): Dropout(p=0.1, inplace=False)
#         )
#         (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
#         (mlp): GPT2MLP(
#           (c_fc): Conv1D()
#           (c_proj): Conv1D()
#           (act): NewGELUActivation()
#           (dropout): Dropout(p=0.1, inplace=False)
#         )
#       )
#     )
#     (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
#   )
#   (lm_head): Linear(in_features=768, out_features=50257, bias=False)
# )>

def main():
    tokenizer = GPT2Tokenizer.from_pretrained('./gpt2', local_files_only = True)
    model = GPT2LMHeadModel.from_pretrained('./gpt2', local_files_only = True)

    text = "Replace me by any text you'd like Replace me by any text you'd like Replace me by any text you'd like Replace me by any text you'd like Replace me by any text you'd like"
    encoded_input = tokenizer(text, return_tensors='pt')

    output = model(**encoded_input)

    # output_text = tokenizer.decode(output)

    print(output)

    # layer_config = create_layer_config(model)
    # with open('./layers.json', 'w+') as f:
    #     f.write(layer_config)

if __name__ == "__main__":
    main()