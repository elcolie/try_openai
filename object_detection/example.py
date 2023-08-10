import os
import numpy as np
import torch
from PIL import Image

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from engine import train_one_epoch, evaluate
import utils
import transforms as T
import math
import sys
import time
import torch

import torchvision.models.detection.mask_rcnn

from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator
import utils

from IPython.display import Image, display

from small_utils import PennFudanDataset, get_transform


device = torch.device('cpu')

# our dataset has two classes only - background and person
num_classes = 2
# use our dataset and defined transformations
dataset = PennFudanDataset('PennFudanPed', get_transform(train=True))
dataset_test = PennFudanDataset('PennFudanPed', get_transform(train=False))

el_dataset_test = PennFudanDataset('ElPed', get_transform(train=False))

# split the dataset in train and test set
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:-50])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=True, num_workers=4,
    collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=4,
    collate_fn=utils.collate_fn)

el_loader_test = torch.utils.data.DataLoader(
    el_dataset_test, batch_size=1, shuffle=False, num_workers=4,
    collate_fn=utils.collate_fn)

from small_utils import get_model_instance_segmentation
model = get_model_instance_segmentation(num_classes)
model.load_state_dict(torch.load("trained_detector_2.saved"))
model.eval()

inputs, classes = next(iter(data_loader_test))



el_inputs, el_classes = next(iter(el_loader_test))   


appsmith_inputs, appsmith_classes = next(iter(el_loader_test))

outputs = model(inputs)

# +
import matplotlib.pyplot as plt
import torch

# Replace the given tensor with your actual tensor
tensor_data = inputs

# Convert the tensor to a numpy array
image_np = tensor_data[0].numpy()

# Transpose the dimensions from (channels, height, width) to (height, width, channels)
image_np = image_np.transpose(1, 2, 0)

# Display the image using matplotlib
plt.imshow(image_np)
plt.axis('off')  # Turn off axis labels and ticks
plt.show()

# -



# +
import matplotlib.pyplot as plt
import numpy as np
import cv2

image_tensor = inputs[0] # Your image tensor
image_np = (image_tensor.numpy() * 255).astype(np.uint8)
instance = outputs[0]  # Your instance dictionary
boxes = instance['boxes'].detach().numpy()
labels = instance['labels'].detach().numpy()
scores = instance['scores'].detach().numpy()
masks = instance['masks'].detach().numpy()

# -

masks

plt.imshow(masks[0][0])

plt.imshow(masks[1][0])

plt.imshow(masks[2][0])

# # El data

# +
import matplotlib.pyplot as plt
import torch

# Replace the given tensor with your actual tensor
tensor_data = el_inputs

# Convert the tensor to a numpy array
image_np = tensor_data[0].numpy()

# Transpose the dimensions from (channels, height, width) to (height, width, channels)
image_np = image_np.transpose(1, 2, 0)

# Display the image using matplotlib
plt.imshow(image_np)
plt.axis('off')  # Turn off axis labels and ticks
plt.show()

# -

el_outputs = model(el_inputs)

el_outputs

# +
import matplotlib.pyplot as plt
import numpy as np
import cv2

el_instance = el_outputs[0]  # Your instance dictionary
el_boxes = el_instance['boxes'].detach().numpy()
el_labels = el_instance['labels'].detach().numpy()
el_scores = el_instance['scores'].detach().numpy()
el_masks = el_instance['masks'].detach().numpy()


# +
from PIL import Image

for i in range(len(el_masks)):
    print(i)
    plt.imshow(el_masks[i][0])
    plt.savefig(f"el_{i}.png")
# -

# # Appsmith

# +
import matplotlib.pyplot as plt
import torch

# Replace the given tensor with your actual tensor
tensor_data = appsmith_inputs

# Convert the tensor to a numpy array
image_np = tensor_data[0].numpy()

# Transpose the dimensions from (channels, height, width) to (height, width, channels)
image_np = image_np.transpose(1, 2, 0)

# Display the image using matplotlib
plt.imshow(image_np)
plt.axis('off')  # Turn off axis labels and ticks
plt.show()

# -

appsmith_outputs = model(appsmith_inputs)

# +
import matplotlib.pyplot as plt
import numpy as np
import cv2

appsmith_instance = appsmith_outputs[0]  # Your instance dictionary
appsmith_masks = appsmith_instance['masks'].detach().numpy()


# +
from PIL import Image

for i in range(len(appsmith_masks)):
    print(i)
    plt.imshow(appsmith_masks[i][0])
    plt.savefig(f"appsmith_{i}.png")
# -


