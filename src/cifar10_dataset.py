import numpy as np
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import Dataset

transform = transforms.Compose([
    transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class CIFAR10Dataset(Dataset):

    def __init__(self, df, partition="all", transform=transform):
        self.transform = transform
        self.data = []
        self.targets = []

        if partition == "all":
            self.data = df.data
            self.targets = df.targets
        else:
            for img, target in df:
                if target in partition:
                    img = np.expand_dims(img, axis=0)
                    if len(self.data) == 0:
                        self.data = img
                    else:
                        self.data = np.append(self.data, img, axis=0)

                    self.targets.append(target)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)
