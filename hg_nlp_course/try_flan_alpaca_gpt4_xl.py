"""
https://crfm.stanford.edu/2023/03/13/alpaca.html
We introduce Alpaca 7B, a model fine-tuned from the LLaMA 7B model on 52K instruction-following demonstrations
https://github.com/tatsu-lab/stanford_alpaca
Choose ready-made from huggingface
https://huggingface.co/declare-lab/flan-alpaca-gpt4-xl
"""
from transformers import pipeline

# model_id: str = "declare-lab/flan-alpaca-gpt4-xl"
model_id: str = "chavinlo/gpt4-x-alpaca"  # Too long to wait downloading the model.
prompt = "Write an instruction how to do gaslift in Texasls"
model = pipeline(model=model_id)
result = model(prompt, max_length=128, do_sample=True)
print(result)

"""
[{'generated_text': "Gaslift is one of the fastest growing eco systems in the world. It is the process of extracting gases and carbon dioxide from the Earth's atmosphere through the process of rising above the ground. It provides a clean and sustainable way to generate electricity, reducing emissions and supporting the development of clean energy. Here is how gaslift works in Texasls: 1. Drive to the start of the gaslift trail.ARESA gas lift trail starts just west of the TEXAS VALdirigeants border crossing. If you are hiking from the Colorado state line, you can enter the first gate, just before the start of"}]
[{'generated_text': 'Gaslift is the process of lifting a car off the ground using the force of two gas lifts attached to the vehicle. It involves lifting the car up and lower it back down into the ground along an accessible slope, often used for road work or access to areas not easily accessible by vehicle. The gas lifts operate by pushing gas upward into an empty vehicle through the gap between the two vehicles. The gas that accumulates inside the car is then released into a designated area where it can be used to drive off the ground. Once the car is lifted off the ground, it is landed on the surface with the gas'}]
"""
