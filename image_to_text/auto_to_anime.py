"""
Image to text feed to prompt and img2img to final result.
Use 2 img2txt
1. git_large_coco
2. blip2_flan_t5_xl
ControlNet
1. canny
2. scribble
3. sd-controlnet-mlsd
"""
import typing as typ

from review import git_large_coco


def double_system(file_path: str) -> str:
    """Save anime image to disk."""
    # Get prompt from image
    print(git_large_coco(file_path))

    # Feed image to controlNet

    # Calculate the anime picture

if __name__ == "__main__":
    double_system("references/IMG_6127.JPG")
