import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import typing as typ

# +
from dataclasses import dataclass

@dataclass
class BoxData:
    """Class for keeping track of an item in inventory."""
    x_min: float
    y_min: float
    box_width_pixel: float
    box_height_pixel: float


# -

def to_box(
    x_center_norm: float, 
    y_center_norm: float, 
    width_norm: float, 
    height_norm:float, 
    image_width: float, 
    image_height: float
) -> BoxData:    
    # Convert normalized coordinates and sizes to pixel values
    x_center_pixel = x_center_norm * image_width
    y_center_pixel = y_center_norm * image_height
    box_width_pixel = width_norm * image_width
    box_height_pixel = height_norm * image_height    
    
    # Derive the corner coordinates of the bounding box
    x_min = x_center_pixel - (box_width_pixel / 2)
    y_min = y_center_pixel - (box_height_pixel / 2)
    x_max = x_center_pixel + (box_width_pixel / 2)
    y_max = y_center_pixel + (box_height_pixel / 2)
    return BoxData(x_min, y_min, box_width_pixel, box_height_pixel)


# Now, (x_min, y_min) and (x_max, y_max) represent the top-left and bottom-right
# corners of the bounding box in pixel coordinates.

def file2boxes(
    lable_path: str = "/Users/sarit/study/try_openai/appsmiths/datasets/coco128/labels/train2017/000000000034.txt",
    image_path: str = "/Users/sarit/study/try_openai/appsmiths/datasets/coco128/images/train2017/000000000034.jpg",
) -> typ.List[BoxData]:
    ans: typ.List[BoxData] = []
    img = np.asarray(Image.open(image_path))
    height, width, _ = img.shape
    
    with open(lable_path, encoding="utf8") as file:
        lines = file.readlines()
        for line in lines:
            _class, x_center_pixel, y_center_pixel, box_width_pixel, box_height_pixel = line.split(" ")
            _class, x_center_pixel, y_center_pixel, box_width_pixel, box_height_pixel = float(_class), \
            float(x_center_pixel), float(y_center_pixel), float(box_width_pixel), float(box_height_pixel)
            ans.append(to_box(
                x_center_pixel, y_center_pixel, box_width_pixel, box_height_pixel,
                width, height
            ))
    return ans



# +
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Load the image
img_path = "/Users/sarit/study/try_openai/appsmiths/datasets/coco128/images/train2017/000000000034.jpg"
img = np.asarray(Image.open(img_path))

lable_path = "/Users/sarit/study/try_openai/appsmiths/datasets/coco128/labels/train2017/000000000034.txt"
boxes = file2boxes(lable_path)

# Create a figure and axis
fig, ax = plt.subplots(1)

# Display the image
imgplot = ax.imshow(img)

boxes: typ.List[BoxData] = file2boxes(lable_path, img_path)
for _box in boxes:
    # Create a rectangle patch representing the bounding box
    # bbox = patches.Rectangle((x_min, y_min), box_width_pixel, box_height_pixel, linewidth=2, edgecolor='r', facecolor='none')
    bbox = patches.Rectangle((_box.x_min, _box.y_min), _box.box_width_pixel, _box.box_height_pixel, linewidth=2, edgecolor='r', facecolor='none')
    
    # Add the bounding box patch to the plot
    ax.add_patch(bbox)

# Show the plot with the image and bounding box
plt.show()

# -


