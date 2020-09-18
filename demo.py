import nibabel as nib
import numpy as np
import os
from pyimgaug3d.augmenters import ImageSegmentationAugmenter
from pyimgaug3d.augmentation import Flip, GridWarp
from pyimgaug3d.utils import to_channels

def save_as_nifti(data, folder, name, affine=np.eye(4)):
    img = nib.Nifti1Image(data, affine)
    if not os.path.exists(folder):
        os.mkdir(folder)
    nib.save(img, os.path.join(folder, name))


img = nib.load('img.nii.gz').get_fdata()
seg = nib.load('seg.nii.gz').get_fdata()

img_in = img
seg_in = seg
print(img.shape, seg.shape)

seg = to_channels(seg)

aug = ImageSegmentationAugmenter()
aug.add_augmentation(Flip(1))
# aug.add_augmentation(GridWarp())

# seg = seg[None, ...]
img = img[..., None]

img, seg = aug([img, seg])
print(img.shape, seg.shape)

save_as_nifti(img_in, 'tmp', 'img.nii.gz')
save_as_nifti(seg_in, 'tmp', 'seg.nii.gz')

save_as_nifti(img, 'tmp', 'img_out.nii.gz')
save_as_nifti(np.argmax(seg, axis=-1).astype('int16'), 'tmp', 'seg_out.nii.gz')