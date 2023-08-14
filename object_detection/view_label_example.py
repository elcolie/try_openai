from small_utils import PennFudanDataset, get_transform
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

train_dataset = PennFudanDataset("PennFudanPed", get_transform(train=True))

data, target = train_dataset[0]

tensor_np = data.numpy()

plt.imshow(tensor_np.transpose(1, 2, 0))  # Transpose to (H, W, C) order
plt.axis('off')  # Turn off axis labels and ticks
plt.show()

mask_img = Image.open("PennFudanPed/PedMasks/FudanPed00001_mask.png").convert("RGB")

_mask = np.array(mask_img)

np.unique(_mask)

_mask

# +
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

# -


