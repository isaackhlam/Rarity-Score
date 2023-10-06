import torch
import random
from torchvision import datasets, transforms, models
import numpy as np

seed = 2302
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


transform = transforms.Compose([
    transforms.ToTensor(), transforms.Normalize(mean=0.5, std=0.5)
])

data_train = datasets.CIFAR10(
        root="./data/",
        transform=transform,
        train=True,
        download=True
)

data_test = datasets.CIFAR10(
        root="./data/",
        transform=transform,
        train=False,
        download=True
)

data_loader_train = torch.utils.data.DataLoader(dataset=data_train, batch_size=64)
data_loader_test = torch.utils.data.DataLoader(dataset=data_test, batch_size=64)

model = models.vgg16(pretrained=True).eval()
