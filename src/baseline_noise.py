import random
import gc
import torch
import psutil
import numpy as np
from rarity_score import *
from cifar10_dataset import *
from gaussian_noise import *
from torchvision import datasets, transforms, models
from tqdm import tqdm


# Set seed for reproducibility
seed = 2302
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


# Model and Datasets to extract features and manifold
model = models.vgg16(pretrained=True).eval()
data_train = datasets.CIFAR10(
        root="./data/",
        train=True,
        download=True
)
data_train = CIFAR10Dataset(data_train)

def get_real_manifold(data_loader_train):
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
    return real_features

def get_fake_images_feature(data_loader_test):
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
    return fake_features


if __name__ == '__main__':

    nearest_k = 3
    p = psutil.Process()
    data_loader_train = torch.utils.data.DataLoader(dataset=data_train, batch_size=64)
    real_features = get_real_manifold(data_loader_train)
    real_features = real_features.squeeze().numpy()

    for i in [i * 0.05 for i in range(0, 100)]:
        print(f"Add Gaussian Noise of sigma {i}. Calculating...\n")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            GaussianNoise(i)
        ])

        data_test = datasets.CIFAR10(
                root="./data",
                train=False,
                download=True,
                transform=transform
        )

        data_loader_test = torch.utils.data.DataLoader(dataset=data_test, batch_size=64)
        fake_features = get_fake_images_feature(data_loader_test)
        fake_features = fake_features.squeeze().numpy()

        manifold = MANIFOLD(real_features=real_features, fake_features=fake_features)
        score, score_index = manifold.rarity(k=nearest_k)
        del manifold
        gc.collect()
        print(score[score_index])
        np.savetxt(f'./result/add_gaussian_noise/baseline_sigma_{i}.txt', score)
        print(f"Memory Info: {p.memory_info()}")
