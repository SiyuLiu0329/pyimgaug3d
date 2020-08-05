class BaseAugmentation:
    def __init__(self):
        pass

    def __call__(self, img):
        # perform augmentation to image
        raise NotImplementedError