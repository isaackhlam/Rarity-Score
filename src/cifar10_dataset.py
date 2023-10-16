from torchvision import datasets, transforms
from torch.utils.data import Dataset

transform = transforms.Compose([
    transforms.ToTensor(), transforms.Normalize(mean=0.5, std=0.5)
])

class CIFAR10Dataset(Dataset):

    def __init__(self, partition, transform=transform):
        self.transform = transform
        self.data = []
        if partition == NULL:
            self.data = Dataset.CIFAR10.data

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)
