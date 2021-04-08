__author__ = 'Jiri Fajtl'
__email__ = 'ok1zjf@gmail.com'
__version__= '1.8'
__status__ = "Research"
__date__ = "2/1/2020"
__license__= "MIT License"

import math
import os
import random
import numpy as np
import torch
from torchvision.utils import make_grid
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio
from PIL import Image

rnd_seed = 12345
random.seed(rnd_seed)
np.random.seed(rnd_seed)
torch.manual_seed(rnd_seed)
# torch.cuda.manual_seed(rnd_seed)

def psnr(original, reconstruction):
    if 1:
        v = peak_signal_noise_ratio(original, reconstruction, data_range=255)
        return v

    if 1:
        v = 0
        for a, b in zip(original, reconstruction):
            v += peak_signal_noise_ratio(a, b, data_range=255)
        v /= original.shape[0]
        return v

    mse = np.mean( (original - reconstruction) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def ssim(original, reconstruction):
    ssim_val = structural_similarity(original, reconstruction, 
            data_range=reconstruction.max() - reconstruction.min(),
            multichannel=True)
    return  ssim_val

def save_image(tensor, filename, nrow=8, padding=2,
               normalize=False, range=None, scale_each=False, pad_value=0):
    from PIL import Image
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(filename)
    return grid

def save_images(x, channels, img_size, path, from_id=0):
        # imgs_test_dir = os.path.join(self.logr.exp_path, 'reconstructed_test', 'eval')
        os.makedirs(path, exist_ok=True)
        # imgs_reco_dir = os.path.join(self.logr.exp_path, 'reconstructed_test', 'reco')
        # os.makedirs(imgs_reco_dir, exist_ok=True)
        for i, ximg in enumerate(x):
            ximg = ximg.view(channels, img_size, img_size)
            ximg = ximg.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            if channels == 1:
                ximg = ximg.reshape(img_size, img_size)
            im = Image.fromarray(ximg)
            im.save(path+'/'+str(i+from_id)+'.jpg')


def roundf(z, npoints=1):
    if npoints < 0:
        z = torch.sign(z)
        z[z==0] = 1
        return z

    # npoints = 1
    scale = 10**npoints
    input = z * scale
    input = torch.round(input)
    z = input / scale
    return z

#=================================================================================
if __name__ == "__main__":
    print("NOT AN EXECUTABLE!")

