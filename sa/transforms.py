from torchvision.transforms import *
from torchvision import transforms

class CropCityscapesArtefacts:
    """Crop Cityscapes images to remove artefacts"""

    def __init__(self, size_assertion=True):
        self.top = 64
        self.left = 128
        self.right = 128
        self.bottom = 256
        self.size_assertion = size_assertion

    def __call__(self, image):
        """Crops a PIL image.

        Args:
            image (PIL.Image): Cityscapes image (or disparity map)

        Returns:
            PIL.Image: Cropped PIL Image
        """
        h, w = image.shape[1:]
        if self.size_assertion:
            assert w == 2048 and h == 1024, f'Expected (2048, 1024) image but got ({w}, {h}). Maybe the ordering of transforms is wrong?'

        if w != 2048 or h != 1024:
            return image
        else:
            return transforms.functional.crop(image, self.top, self.left, h-self.bottom, w-self.right)

    def __str__(self):
        return 'CropCityscapesArtefacts()'


class MinimalCrop:
    """
    Performs the minimal crop such that height and width are both divisible by min_div.
    """

    def __init__(self, min_div=32):
        self.min_div = min_div

    def __call__(self, image):
        h, w = image.shape[1:]

        h_new = h - (h % self.min_div)
        w_new = w - (w % self.min_div)

        if h_new == 0 and w_new == 0:
            return image
        else:    
            h_diff = h-h_new
            w_diff = w-w_new

            top = int(h_diff//2)
            bottom = h_diff-top
            left = int(w_diff//2)
            right = w_diff-left

            return transforms.functional.crop(image, top, left, h_new, w_new)

    def __str__(self):
        return f'MinimalCrop(min_div={self.min_div})'

class RandomCrop:
    
    def __init__(self, size: tuple):
        self.h = size[0]
        self.w = size[1]
        
    def __call__(self, *args):
        w = min([a.shape[2] for a in args] + [self.w])
        h = min([a.shape[1] for a in args] + [self.h])
        
        crop_params = transforms.RandomCrop.get_params(args[0], (h, w))
        return [transforms.functional.crop(x, *crop_params) for x in args]

    def __str__(self):
        return f'RandomCrop(size=({self.h}, {self.w}))'

class TWrapper:
    
    def __init__(self, transform):
        self.transform = transform
        
    def __call__(self, *args):
        return [self.transform(x) for x in args]

    def __str__(self):
        return str(self.transform)
    
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, *args):
        for transform in self.transforms:
            args = transform(*args)
            
        return args