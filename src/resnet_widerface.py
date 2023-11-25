import torch
import random
import numpy as np
from rarity_score import *
from torchvision import datasets, transforms, models
from tqdm import tqdm
from torch.utils.data import DataLoader

seed = 2302
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform_resnet = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# load datasets
train_wider_dataset = datasets.WIDERFace("./widerface_data", split ="train", download=True, transform = transform_resnet)
test_wider_dataset = datasets.WIDERFace("./widerface_data", split ="test", download=True, transform = transform_resnet)


# load feature extractor
resnet = models.resnet50(pretrained=True).to(device)

train_wider_loader = DataLoader(train_wider_dataset, batch_size=64)
test_wider_loader = DataLoader(test_wider_dataset, batch_size=64)

def extract_features(model, data_loader):
    model.eval()
    features = []
    with torch.no_grad():
        for batch in tqdm(data_loader):
            images = batch[0].to(device)
            output = model(images)
            features.append(output.cpu())
    return torch.cat(features, dim=0)

real_wider_features = extract_features(resnet, train_wider_loader)
fake_wider_features = extract_features(resnet, test_wider_loader)

real_wider_features = real_wider_features.numpy()
fake_wider_features = fake_wider_features.numpy()


manifold = MANIFOLD(real_features=real_wider_features, fake_features=fake_wider_features)
score, score_index = manifold.rarity(k=3)
print(score[score_index])
np.savetxt('./result/resnet_widerface.txt', score)
