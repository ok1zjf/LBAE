__author__ = 'Jiri Fajtl'
__email__ = 'ok1zjf@gmail.com'
__version__= '1.8'
__status__ = "Research"
__date__ = "2/1/2020"
__license__= "MIT License"

from __future__ import print_function
from torchvision.datasets import CIFAR10 
from torchvision.datasets import MNIST
from torchvision.datasets import CelebA


def corrupt(x, corrupt_method, corrupt_args):
    """ 
        Disabled
    """ 
    return x

class CIFAR10Ex(CIFAR10):

    def __init__(self, root, train=True,
                 transform=None, target_transform=None, download=False,
                 corrupt_method='noise', corrupt_args=[0.4]):

        super().__init__(root, train, transform, target_transform, download)

        self.corrupt_method = corrupt_method
        self.corrupt_args = corrupt_args
        return

    def __getitem__(self, index):
        image, target = super().__getitem__(index)

        # Corrupt images
        corrupted = corrupt(image, self.corrupt_method, self.corrupt_args)
        return image, target, corrupted

class MNISTEx(MNIST):

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False,
                 corrupt_method='noise', corrupt_args=[0.4]):
        super().__init__(root, train, transform, target_transform, download)
        self.corrupt_method = corrupt_method
        self.corrupt_args = corrupt_args
        
    def __getitem__(self, index):
        image, target = super().__getitem__(index)

        # Corrupt images
        corrupted = corrupt(image, self.corrupt_method, self.corrupt_args)
        return image, target, corrupted


class CelebAEx(CelebA):

    def __init__(self, root,
                 split="train",
                 target_type="attr",
                 transform=None, target_transform=None,
                 download=False,
                 corrupt_method='noise', corrupt_args=[0.4]):
        super().__init__(root, split, target_type, transform, target_transform, download)
        self.corrupt_method = corrupt_method
        self.corrupt_args = corrupt_args

    def __getitem__(self, index):
        image, target = super().__getitem__(index)

        # Corrupt images
        corrupted = corrupt(image, self.corrupt_method, self.corrupt_args)
        return image, target, corrupted


#=================================================================================
if __name__ == "__main__":
    print("NOT AN EXECUTABLE!")

