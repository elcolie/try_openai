# Try run checkpoint from civitAI
Search for civitAI on hf.co
https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/img2img
CharTurner V2 - For 1.5
https://civitai.com/models/3036?modelVersionId=8387

# Covert .safetensors to pre-trained model
1. Download [safetensors](https://civitai.com/models/43331/majicmix-realistic)
1. Download [python file](https://github.com/huggingface/diffusers/blob/main/scripts/convert_original_stable_diffusion_to_diffusers.py)
2. `mkdir converted`
3. `python convert_original_stable_diffusion_to_diffusers.py --checkpoint_path majicmixRealistic_v5.safetensors --from_safetensors --dump_path converted`
4. ```python
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny").to(device)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "converted",  # This line
        safety_checker=None,
        controlnet=controlnet,
    ).to(device)
```
