class BaseAugmentation:
    def __init__(self):
        pass

    def __call__(self, img):
        # perform augmentation to image
        # img.shape must be 4D (h, w, d, c), without batch dim
        raise NotImplementedError