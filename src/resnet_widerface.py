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

class WiderFaceDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.annotations = self.load_annotations(annotation_file)
        self.images = [annotation['image'] for annotation in self.annotations]

    def load_annotations(self, annotation_file):
        annotations = []
        with open(annotation_file, 'r') as file:
            lines = file.readlines()
            i = 0
            while i < len(lines):
                # read the image filename
                image_path = lines[i].strip()
                i += 1

                # the number of faces
                num_faces = int(lines[i].strip())
                i += 1  # Move to the first bounding box line or skip if num_faces is 0

                # if no faces, skip this image
                if num_faces == 0:
                    i += 1
                    continue

                boxes = []
                for _ in range(num_faces):
                    box_info = lines[i].strip().split()
                    # Extract the first four values (x, y, width, height)
                    x, y, w, h = [int(box_info[k]) for k in range(4)]
                    boxes.append([x, y, x + w, y + h])
                    i += 1

                annotations.append({'image': image_path, 'boxes': boxes})

        return annotations



    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.annotations[index]['image'])
        img = Image.open(img_path)

        original_size = img.size

        # corresponding bounding boxes
        bboxes = self.annotations[index]['boxes']

        # apply image transformations()
        if self.transform:
            img = self.transform(img)

        # scale factors
        scale_x = 299 / original_size[0]
        scale_y = 299 / original_size[1]

        # scale the bounding boxes
        scaled_bboxes = []
        for box in bboxes:
            x1, y1, x2, y2 = box
            scaled_bboxes.append([x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y])

        # convert scaled bounding boxes(annotations) to a tensor
        scaled_bboxes = torch.tensor(scaled_bboxes, dtype=torch.float32)

        return img, scaled_bboxes

transform_resnet = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# load datasets
train_wider_dataset = WiderFaceDataset(image_dir='./widerface_data/widerface/WIDER_train/images/', 
                                 annotation_file='./widerface_data/widerface/wider_face_split/wider_face_train_bbx_gt.txt', 
                                 transform=transform_resnet)
test_wider_dataset = WiderFaceDataset(image_dir='./widerface_data/widerface/WIDER_val/images/', 
                                 annotation_file='./widerface_data/widerface/wider_face_split/wider_face_val_bbx_gt.txt', 
                                 transform=transform_resnet)


# load feature extractor
resnet = models.resnet50(pretrained=True).to(device)

def collate_fn(batch):
    images = [item[0] for item in batch]
    annotations = [item[1] for item in batch]  # handle varying-size annotations

    # stack the images into a single tensor
    images = torch.stack(images, dim=0)

    return images, annotations


train_wider_loader = DataLoader(train_wider_dataset, batch_size=64, collate_fn=collate_fn)
test_wider_loader = DataLoader(test_wider_dataset, batch_size=64, collate_fn=collate_fn)

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
