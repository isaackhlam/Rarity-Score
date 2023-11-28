import torch
import random
import numpy as np
from rarity_score import *
from torchvision import datasets, transforms, models
from tqdm import tqdm
from torch.utils.data import DataLoader,Dataset
from PIL import Image
import os

seed = 2302
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ImageDataset(datasets):
    def __item__(self, path, transform):
        self.path = path
        self.transform = transform
        self.img_names = os.listdir(path)
        
    def __len__(self):
        return len(self.img_names)

    def __getitem__(self,idx):
        img_path = os.path.join(self.path, self.img_names[idx])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image


transform = transforms.Compose([
    transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

data_train = datasets.CIFAR10(
        root="./data/",
        transform=transform,
        train=True,
        download=True
)

data_test = ImageDataset("./cifar10-stylegan2-imgs/", transform)

data_loader_train = DataLoader(dataset=data_train, batch_size=64)
data_loader_test = DataLoader(dataset=data_test, batch_size=64)

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
        x = data
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
np.savetxt('./result/vgg_stylegan2.txt', score)


#biggan

data_test = ImageDataset("./cifar10-biggan-deep-imgs/", transform)

# data_loader_train = DataLoader(dataset=data_train, batch_size=64)
data_loader_test = DataLoader(dataset=data_test, batch_size=64)

# model = models.vgg16(pretrained=True).eval()

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
        x = data
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
np.savetxt('./result/vgg_biggan_deep.txt', score)