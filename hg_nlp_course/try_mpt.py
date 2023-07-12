"""
https://learning.edx.org/course/course-v1:Databricks+LLM101x+2T2023/block-v1:Databricks+LLM101x+2T2023+type@sequential+block@e57f7ee0973643318091f3b9f4a83911/block-v1:Databricks+LLM101x+2T2023+type@vertical+block@859b3255960844bba90a148ccb8fcac1
"""

import torch
import transformers
from transformers import AutoTokenizer
from transformers import pipeline

name: str = 'mosaicml/mpt-7b'
device: str = "cpu"

config = transformers.AutoConfig.from_pretrained(name, trust_remote_code=True)
# config.attn_config['attn_impl'] = 'triton'
config.init_device = device  # For fast initialization directly on GPU!

model = transformers.AutoModelForCausalLM.from_pretrained(
    name,
    config=config,
    torch_dtype=torch.bfloat16,  # Load model weights in bfloat16
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')

pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, device=device)

with torch.autocast(device, dtype=torch.bfloat16):
    print(
        pipe('Here is a recipe for vegan banana bread:\n',
             max_new_tokens=100,
             do_sample=True,
             use_cache=True))
