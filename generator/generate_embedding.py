from typing import List

import re
import utils

OUTPUT_DIR = "./generated"

def generate_embedding(
        name: str, embeddings: List[List[float]]):
    """
    Generates source code for embedding lookup layer.
    """
    assert len(embeddings) > 0
    vocab_size = len(embeddings)
    embedding_size = len(embeddings[0])

    output_file_path = f"{OUTPUT_DIR}/{name}_embedding.c"
    template_path = "./generation_templates/embedding.c.template"

    with open(template_path) as template:
        template_text = template.read()

        implementation = template_text[template_text.index("// START_METHODS"):template_text.index("// END_METHODS")]
        # replace placeholder variables
        implementation = implementation.replace("{NAME}", name)
        implementation = implementation.replace("{EMBEDDING_SIZE}", str(embedding_size)).replace("{VOCAB_SIZE}", str(vocab_size))

        source = utils.unroll_loops(implementation)
        
        # replace all the embedding variables with actual values
        for idx, embedding in enumerate(embeddings):
            placeholder = "{EMBEDDING_" + str(idx) + "}"
            embedding_str = '{' + ', '.join(map(str, embedding)) + '}'
            source = source.replace(placeholder, embedding_str);


        with open(output_file_path, 'w+') as of:
            of.write(source)


if __name__ == "__main__":
    generate_embedding(
        name="gpt2_layer_embedding",
        embeddings=[
            [0.1 for _ in range(768)] for _ in range(50257)
        ]
    )
