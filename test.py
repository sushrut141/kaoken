import json
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def create_layer_config(model):
    """
    Creates a JSON configuration of all layers in the given model, including weights and operations
    and writes it to disk.

    Args:
        model: The Hugging Face model object.
    """
    module_config = []
    for name, module in model.named_modules():
        temp = {}
        if name:
            temp[name] = {
                'name': name,
                'type': module.__class__.__name__,
            }
            temp['params'] = []
            for param_name, param in module.named_parameters():
                param_temp = {}
                param_temp['param_name'] = param_name
                param_temp['dimensions'] = list(param.shape)
                temp['params'].append(param_temp)
            module_config.append(temp)
    with open('./modules.json', 'w+') as f:
        f.write(json.dumps(module_config, indent=4))

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

    text = "how are you?"
    encoded_input = tokenizer.encode(text, return_tensors='pt')

    output = model.generate(encoded_input)

    output_text = tokenizer.decode(output[0])

    print(output_text)

    _ = create_layer_config(model)

if __name__ == "__main__":
    main()