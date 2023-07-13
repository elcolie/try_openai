"""
Pick the smallest parameters model.
https://huggingface.co/databricks/dolly-v2-3b
Run the model follow this.
https://github.com/databrickslabs/dolly#getting-started-with-response-generation
"""

import torch
from transformers import pipeline

generate_text = pipeline(model="databricks/dolly-v2-3b", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
res = generate_text("Explain to me the difference between nuclear fission and fusion.")
print(res[0]["generated_text"])
