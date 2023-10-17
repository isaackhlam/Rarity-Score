import torch
import random
import numpy as np
from rarity_score import *
from cifar10_dataset import *
from torchvision import datasets, transforms, models
from tqdm import tqdm

seed = 2302
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


data = datasets.CIFAR10(
        root="./data/",
        train=True,
        download=True
)

data_train = CIFAR10Dataset(data, partition=range(1,10))
data_test = CIFAR10Dataset(data, partition=[10])

# sanity check to verify
print(len(data_train))
print(len(data_test))

data_loader_train = torch.utils.data.DataLoader(dataset=data_train, batch_size=64)
data_loader_test = torch.utils.data.DataLoader(dataset=data_test, batch_size=64)

model = models.vgg16(pretrained=True).eval()

flag = 1
with torch.no_grad():
    for data in tqdm(data_loader_train):
        x, _ = data
        features = model.features(x)
        if flag:
            real_features = features
            flag = 0
        else:
            real_features = torch.cat((real_features, features), 0)
            del features


flag = 1
with torch.no_grad():
    for data in tqdm(data_loader_test):
        x, _ = data
        features = model.features(x)
        if flag:
            fake_features = features
            flag = 0
        else:
            fake_features = torch.cat((fake_features, features), 0)
            del features

nearest_k = 3

real_features, fake_features = real_features.squeeze().numpy(), fake_features.squeeze().numpy()
manifold = MANIFOLD(real_features=real_features, fake_features=fake_features)
score, score_index = manifold.rarity(k=nearest_k)
print(score[score_index])
