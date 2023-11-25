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


transform_inception = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# transform_resnet = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])


# load datasets
# train_wider_dataset = datasets.WIDERFace("./widerface_data", split ="train", download=True, transform = transform_inception)
# test_wider_dataset = datasets.WIDERFace("./widerface_data", split ="test", download=True, transform = transform_inception)
train_pet_dataset = datasets.OxfordIIITPet(root="./oxford_pet_data/", split="trainval", download=True, transform=transform_inception)
test_pet_dataset = datasets.OxfordIIITPet(root="./oxford_pet_data/", split="test", download=True, transform=transform_inception)


# load feature extractor
inception = models.inception_v3(pretrained=True).to(device)
# resnet = models.resnet50(pretrained=True).to(device)

train_pet_loader = DataLoader(train_pet_dataset, batch_size=64)
test_pet_loader = DataLoader(test_pet_dataset, batch_size=64)
# train_wider_loader = DataLoader(train_wider_dataset, batch_size=64)
# test_wider_loader = DataLoader(test_wider_dataset, batch_size=64)

def extract_features(model, data_loader):
    model.eval()
    features = []
    with torch.no_grad():
        for batch in tqdm(data_loader):
            images = batch[0].to(device)
            output = model(images)
            features.append(output.cpu())
    return torch.cat(features, dim=0)

real_pet_features = extract_features(inception, train_pet_loader)
fake_pet_features = extract_features(inception, test_pet_loader)

# real_wider_features = extract_features(inception, train_wider_loader)
# fake_wider_features = extract_features(inception, test_wider_loader)

real_pet_features = real_pet_features.numpy()
fake_pet_features = fake_pet_features.numpy()


manifold = MANIFOLD(real_features=real_pet_features, fake_features=fake_pet_features)
score, score_index = manifold.rarity(k=3)
print(score[score_index])
np.savetxt('./result/inception_pet.txt', score)
