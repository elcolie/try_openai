from diffusers import DiffusionPipeline
import torch

torch.manual_seed(111)
device = "mps" if torch.backends.mps.is_available() else "cpu"

pipe = DiffusionPipeline.from_pretrained(
    'hakurei/waifu-diffusion',
    custom_pipeline="lpw_stable_diffusion",
)
pipe = pipe.to(device)

prompt: str = "best_quality (1girl:1.3) bow bride brown_hair closed_mouth frilled_bow frilled_hair_tubes frills (full_body:1.3) fox_ear hair_bow hair_tubes happy hood japanese_clothes kimono long_sleeves red_bow smile solo tabi uchikake white_kimono wide_sleeves cherry_blossoms"
negative_prompt: str = "lowres, bad_anatomy, error_body, error_hair, error_arm, error_hands, bad_hands, error_fingers, bad_fingers, missing_fingers, error_legs, bad_legs, multiple_legs, missing_legs, error_lighting, error_shadow, error_reflection, text, error, extra_digit, fewer_digits, cropped, worst_quality, low_quality, normal_quality, jpeg_artifacts, signature, watermark, username, blurry"
# prompt = """
# best quality, 8K, HDR, highres, blurry background, bokeh:1.3, Photography, (RAW photo:1.2, photorealistic:1.4, masterpiece:1.3, best quality:1.3, ultra highres:1.2), (((pin light, backlighting))), (depth of field), sharp focus:1.4, (camera from distance), (super_detail, hyper_detail, finely_detailed) ADDBASE blue sky ADDROW blue see ADDROW sand beach BREAK 1woman standing on sand beach, front view person. full body, ((((mature woman, solo, glamorous)))), (((gigantic breasts:0.7, narrow waist, oiledwet, large ass, thin thighs))) BREAK high detailed skin, delicate, beautiful skin, solid circle eyes, detailed beautiful face, beautiful detailed, (detailed eyes), (detailed facial features), beautiful and clear eyes, detail eye pupil, lipgloss, blush, cheek, long hair, black hair, (((expression_face, hair over one eye, beautiful fingers, beautiful hands))) BREAK bikini, earrings, necklace, sandals
# """
# negative_prompt: str = "Negative prompt: ((((BadNegAnatomyV1-neg, badhandv4)))), EasyNegative, paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), ((((skin spots, acnes, skin blemishes, age spot)))), (((water mark, username, signature, text))), (((extra fingers, extra legs, extra foot, missing body, missing face, missing arms, missing hands, missing fingers, missing legs, missing feet, missing toe, strange fingers, bad hands, fewer fingers, extra digit, fewer digits)))"

num_images_per_prompt: int = 8

result = pipe.text2img(prompt, negative_prompt=negative_prompt, width=512, height=512, max_embeddings_multiples=3, num_images_per_prompt=num_images_per_prompt)
for idx, image in enumerate(result.images):
    image.save(f"1.3/long_prompt_{idx}.png")
