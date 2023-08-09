"""Image to Text experiments."""
import typing as typ

import torch
from PIL import Image
from diffusers.utils import load_image
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer


device = "mps" if torch.cuda.is_available() else "cpu"

def vit_gpt2_image_captioning(files: typ.List[str]) -> typ.List[str]:
    """https://huggingface.co/nlpconnect/vit-gpt2-image-captioning"""
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    model.to(device)

    max_length = 16
    num_beams = 4
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

    def predict_step(image_paths):
        images = []
        for image_path in image_paths:
            i_image = Image.open(image_path)
            if i_image.mode != "RGB":
                i_image = i_image.convert(mode="RGB")

            images.append(i_image)

        pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)

        output_ids = model.generate(pixel_values, **gen_kwargs)

        preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        preds = [pred.strip() for pred in preds]
        return preds

    # predict_step(['doctor.e16ba4e4.jpg'])  # ['a woman in a hospital bed with a woman in a hospital bed']
    return predict_step(files)


def blip_image_captioning_large(file_path: str) -> str:
    """https://huggingface.co/Salesforce/blip-image-captioning-large"""
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)

    raw_image = load_image(file_path)
    # img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
    # raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

    # conditional image captioning
    text = "a photography of"
    inputs = processor(raw_image, text, return_tensors="pt").to(device)
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

    # unconditional image captioning
    # inputs = processor(raw_image, return_tensors="pt").to(device)
    # out = model.generate(**inputs)
    # print(processor.decode(out[0], skip_special_tokens=True))


def blip_image_captioning_base(file_path: str) -> str:
    """https://huggingface.co/Salesforce/blip-image-captioning-base"""
    from transformers import BlipProcessor, BlipForConditionalGeneration
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

    raw_image = load_image(file_path)

    # conditional image captioning
    text = "a photography of"
    inputs = processor(raw_image, text, return_tensors="pt").to(device)

    out = model.generate(**inputs)
    print(processor.decode(out[0], skip_special_tokens=True))
    # >>> a photography of a woman and her dog

    # unconditional image captioning
    inputs = processor(raw_image, return_tensors="pt").to(device)

    out = model.generate(**inputs)
    print(processor.decode(out[0], skip_special_tokens=True))


def git_large_coco(file_path) -> str:
    """https://huggingface.co/microsoft/git-large-coco"""
    from transformers import AutoProcessor, AutoModelForCausalLM

    processor = AutoProcessor.from_pretrained("microsoft/git-base-coco")
    model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco")

    image = load_image(file_path)

    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return (generated_caption)


def blip2_flan_t5_xl(file_path: str) -> str:
    """https://huggingface.co/Salesforce/blip2-flan-t5-xl"""
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl", device_map="auto")

    raw_image = load_image(file_path)

    question = "A picture "
    inputs = processor(raw_image, question, return_tensors="pt").to(device)

    out = model.generate(**inputs)
    return question + (processor.decode(out[0], skip_special_tokens=True))


def blip_image_captioning(file_path: str) -> str:
    """https://huggingface.co/prasanna2003/blip-image-captioning"""
    from transformers import BlipProcessor, BlipForConditionalGeneration

    processor = BlipProcessor.from_pretrained("prasanna2003/blip-image-captioning")
    if processor.tokenizer.eos_token is None:
        processor.tokenizer.eos_token = '<|eos|>'
    model = BlipForConditionalGeneration.from_pretrained("prasanna2003/blip-image-captioning")

    image = load_image(file_path)

    prompt = """Instruction: Generate a single line caption of the Image.
    output: """

    inputs = processor(image, prompt, return_tensors="pt")

    output = model.generate(**inputs, max_length=100)
    return (processor.tokenizer.decode(output[0]))


def blip2_opt_27b(file_path: str) -> str:
    """https://huggingface.co/Salesforce/blip2-opt-2.7b"""
    # pip install accelerate
    import requests
    from PIL import Image
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", device_map="auto")

    raw_image = load_image(file_path)

    question = "a picture of "
    inputs = processor(raw_image, question, return_tensors="pt").to("cuda")

    out = model.generate(**inputs)
    return(processor.decode(out[0], skip_special_tokens=True))

def instructblip_flan_t5_xl(file_path: str) -> str:
    """https://huggingface.co/Salesforce/instructblip-flan-t5-xl"""
    from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
    import torch
    from PIL import Image
    import requests

    model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-flan-t5-xl")
    processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl")

    model.to(device)

    image = load_image(file_path)
    prompt = "a picture of "
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        do_sample=False,
        num_beams=5,
        max_length=256,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1.0,
        temperature=1,
    )
    generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    return(generated_text)


def instructblip_vicuna_7b(file_path: str) -> str:
    """https://huggingface.co/Salesforce/instructblip-vicuna-7b"""
    from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
    import torch
    from PIL import Image
    import requests

    model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b")
    processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")

    model.to(device)

    image = load_image(file_path)
    prompt = "describe this image "
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        do_sample=False,
        num_beams=5,
        max_length=256,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1.0,
        temperature=1,
    )
    generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    return(generated_text)


def main() -> None:
    """Run main function."""
    file_path: str = "references/IMG_6127.JPG"
    my_dict = {
        # 'blip_image_captioning_large': blip_image_captioning_large,
        # 'blip_image_captioning_base': blip_image_captioning_base,
        # 'git_large_coco': git_large_coco,  # Good
        # 'blip2_flan_t5_xl': blip2_flan_t5_xl,  # Good
        # 'blip_image_captioning': blip_image_captioning,
        # 'blip2_opt_27b': blip2_opt_27b,
        'instructblip_flan_t5_xl': instructblip_flan_t5_xl,
        'instructblip_vicuna_7b': instructblip_vicuna_7b,
    }
    # print(vit_gpt2_image_captioning([file_path]))
    print("===============")
    for key, value in my_dict.items():
        print(f"{key}: {value(file_path)}")
        print("===============")


if __name__ == "__main__":
    main()
