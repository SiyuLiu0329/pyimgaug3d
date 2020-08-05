import random

class BaseAugmenter:
    def __init__(self):
        self.augmentation = []
    
    def add_augmentation(self, augmentation):
        self.augmentation.append(augmentation)

    def __call__(self, images):
        aug = random.choice(self.augmentation)
        return aug(images)