"""Chapter 6: https://huggingface.co/learn/nlp-course/chapter6/3?fw=pt"""
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
example = "My name is Sylvain and I work at Hugging Face in Brooklyn."
encoding = tokenizer(example)
print(type(encoding))

# Try it out! Create a tokenizer from the bert-base-cased and roberta-base checkpoints and tokenize ”81s” with them.
# What do you observe? What are the word IDs?
tokenizer_names = [
    "bert",
    "roberta"
]
tokenizers = (
    AutoTokenizer.from_pretrained("bert-base-cased"),
    AutoTokenizer.from_pretrained("roberta-base")
)
for name, tokenizer in zip(tokenizer_names, tokenizers):
    print(name)
    example = "81s"
    encoding = tokenizer(example)
    print(encoding.tokens())
    print(encoding.word_ids())
    print(type(encoding))

for name, tokenizer in zip(tokenizer_names, tokenizers):
    print(name)
    example = "Thai monarchy is a good organization. It is being cursed by Thai people."
    encoding = tokenizer(example)
    print(encoding.tokens())
    print(encoding.word_ids())
    print(encoding.sequence_ids())
    print(type(encoding))

#
from transformers import AutoTokenizer, AutoModelForTokenClassification

model_checkpoint = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint)

example = "My name is Sylvain and I work at Hugging Face in Brooklyn."
inputs = tokenizer(example, return_tensors="pt")
outputs = model(**inputs)


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
print(type(tokenizer.backend_tokenizer))
print(tokenizer.backend_tokenizer.normalizer.normalize_str("Héllò hôw are ü?"))


tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
print(type(tokenizer.backend_tokenizer))
print(tokenizer.backend_tokenizer.normalizer.normalize_str("Héllò hôw are ü?"))
