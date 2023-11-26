import torch
import random
import numpy as np
from rarity_score import *
from torchvision import datasets, transforms, models
from tqdm import tqdm
from torch.utils.data import DataLoader,Dataset, random_split
from PIL import Image
import os

seed = 2302
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform_inception = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = datasets.CelebA(root="./celeba_data", split="all", download=False, transform=transform_inception)

# parallel processing with GPUs
inception = models.inception_v3(pretrained=True)
inception = torch.nn.DataParallel(inception)
inception.to(device)


# split into 'real' and 'fake' portions
fake_percentage = 0.3
fake_size = int(len(dataset) * fake_percentage)
real_size = len(dataset) - fake_size
real_dataset, fake_dataset = random_split(dataset, [real_size, fake_size])

real_loader = DataLoader(real_dataset, batch_size=256)
fake_loader = DataLoader(fake_dataset, batch_size=256)

def extract_features(model, data_loader):
    model.eval()
    features = []
    with torch.no_grad():
        for batch in tqdm(data_loader):
            images = batch[0].to(device)
            output = model(images)
            features.append(output.cpu())
    return torch.cat(features, dim=0)

real_features = extract_features(inception, real_loader)
fake_features = extract_features(inception, fake_loader)

real_features = real_features.numpy()
fake_features = fake_features.numpy()

nearest_k = 3
manifold = MANIFOLD(real_features=real_features, fake_features=fake_features)
score, score_index = manifold.rarity(k=nearest_k)
print(score[score_index])

np.savetxt("./result/inception_celeba.txt", score)

import torchvision.transforms as T
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

def tensor_to_pil(tensor):
    # undo the normalization
    unnormalize = T.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225], std=[1/0.229, 1/0.224, 1/0.225])
    tensor = unnormalize(tensor)
    # convert to PIL Image
    return T.ToPILImage()(tensor)

non_zero_scores = score[score != 0]
sorted_scores = np.sort(non_zero_scores)

# indcies for lowest and highest scores
lowest_indices = score_index[-3:]
highest_indices = score_index[:3]

# idx for median score
median_score = np.median(sorted_scores)
median_index = np.where(score == median_score)[0][0]

zero_indices = score[score == 0]

def save_image(dataset, index, filename):
    image_tensor, _ = dataset[index]
    image = tensor_to_pil(image_tensor)
    image.save(f'./result/inception_celeba/{filename}')

for i, index in enumerate(lowest_indices):
    save_image(dataset, index, f'lowest_{i+1}.png')

for i, index in enumerate(highest_indices):
    save_image(dataset, index, f'highest_{i+1}.png')

for i, index in enumerate(zero_indices[:3] if len(zero_indices)>=3 else zero_indices):
    save_image(dataset, index, f'zero_{i+1}.png')

save_image(dataset, median_index, 'median.png')
