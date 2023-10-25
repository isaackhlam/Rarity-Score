import torch

sigma = 1

def gaussian_noise_tensor(img, sigma=sigma):
    print(sigma)
    assert isinstance(img, torch.Tensor)
    dtype = img.dtype
    if not img.is_floating_point():
        img = img.to(torch.float32)

    out = img + sigma * torch.randn_like(img)

    if out.dtype != dtype:
        out = out.to(dtype)

    return out

class GaussianNoise(object):

    def __init__(self, sigma=sigma):
        self.sigma = sigma

    def __call__(self, sample):
        return gaussian_noise_tensor(sample, self.sigma)
