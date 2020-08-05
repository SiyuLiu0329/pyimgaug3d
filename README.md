# Pyimgaug3d
A 3D *GPU* augmentation library.

# Installation
```
pip install pyimgaug3d
```

# Example Usage
```
from pyimgaug3d.augmentation import GridWarp
from pyimgaug3d.augmenters import ImageSegmentationAugmenter

img = load_img... # shape=(H,W,D,C)
seg = load_seg... # shape=(H,W,D,C)(one hot encoded)

# This augmenter automatically rounds the segmentation mask
aug = ImageSegmentationAugmenter()
aug.add_augmentation(GridWarp(grid=(4, 5, 5), max_shift=10))

warpped_img, warpped_seg = aug([img, seg])
```
