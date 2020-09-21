# Pyimgaug3d
A 3D *GPU* augmentation library. This library is work-in-progress and will be constantly updated with more augmentation methods. The current supported ones are grid warp 3D, flip  3D and identity. As all the augmentation methods are implemented in TensorFlow, a Cuda compatible GPU is required to take advantage of increases speeds.

# Installation
```
pip install pyimgaug3d
```

# Example Usage
```
from pyimgaug3d.augmentation import GridWarp, Flip, Identity
from pyimgaug3d.augmenters import ImageSegmentationAugmenter

img = load_img... # shape=(H,W,D,C)
seg = load_seg... # shape=(H,W,D,C)(one hot encoded)

# This augmenter automatically rounds the segmentation mask
aug = ImageSegmentationAugmenter()
aug.add_augmentation(GridWarp(grid=(4, 5, 5), max_shift=10))
aug.add_augmentation(Flip(0))
aug.add_augmentation(Identity())

aug_img, aug_seg = aug([img, seg]) # call to perform augmentation, each time an augmentation method is sampled at random.
```
# Citation
This library is published along with the following paper
```
S.  Liu,  W.  Dai,  C.  Engstrom,  J.  Fripp,  P.  B.  Greer,  S.  Crozier,J. A. Dowling, 
and S. S. Chandra, “Fabric Image Representation Encoding Networks for Large-scale 3D Medical 
Image Analysis,”arXiv e-prints, p. arXiv:2006.15578, Jun. 2020.
```