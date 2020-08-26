from pyimgaug3d.augmenters.src.base_augmenter import BaseAugmenter
from pyimgaug3d.utils import classical_round_tf

"""
accepts 2 images, the second one gets roudned
"""
class ImageSegmentationAugmenter(BaseAugmenter):
    def __call__(self, imgs):
        assert len(imgs) == 2, "only pass in [img, seg]"
        x, y = super().__call__(imgs)
        return x, classical_round_tf(y)

class ImageAugmenter(BaseAugmenter):
    def __call__(self, img):
        x = super().__call__([img])
        return x[0]