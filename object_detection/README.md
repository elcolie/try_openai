# Setup
Also use the files from https://github.com/pytorch/vision/tree/main/references/detection
https://discuss.pytorch.org/t/transforms-helper-function-has-no-attribute-totensor/153985/3
[transforms.py](transforms.py) -> Use v0.8.2
[tv-training-code.py](tv-training-code.py)
[utils.py](utils.py)
[coco_eval.py](coco_eval.py)
[coco_utils.py](coco_utils.py)
[engine.py](engine.py)

# To view the label of dataset
It is 0, 1, 2, ... then when view in image viewer will be black and human eyes can't distinguish.

```python
from PIL import Image
import numpy as np

# Load the mask image
mask_img = Image.open("PennFudanPed/PedMasks/FudanPed00001_mask.png").convert("L")  # Convert to grayscale

# Define a color mapping
color_mapping = {
    0: (0, 0, 0),     # Black
    1: (255, 0, 0),   # Red
    2: (0, 255, 0),   # Green
}

# Convert the mask image to a numpy array
mask_array = np.array(mask_img)

# Create a new image with the colored regions
colored_mask = Image.new("RGB", mask_array.shape[::-1])
pixels = colored_mask.load()

# Apply the color mapping to the mask image
for i in range(mask_array.shape[0]):
    for j in range(mask_array.shape[1]):
        pixel_value = mask_array[i, j]
        color = color_mapping.get(pixel_value, (0, 0, 0))  # Default to black if pixel_value is not in the mapping
        pixels[j, i] = color  # Note the transposed indexing here

# Display the colored mask image
colored_mask.show()
```
