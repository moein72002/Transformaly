import os
import shutil
from pathlib import Path
import os
import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import random
from PIL import Image
from glob import glob

import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.utils.data.dataset import Subset
from torchvision.transforms import Compose
from visualize.visualize_dataset import visualize_random_samples_from_clean_dataset
from visualize.count_labels import count_unique_labels_of_dataset

def prepare_br35h_dataset_files():
    normal_path35 = './Br35H/no'
    anomaly_path35 = './Br35H/yes'

    print(f"len(os.listdir(normal_path35)): {len(os.listdir(normal_path35))}")
    print(f"len(os.listdir(anomaly_path35)): {len(os.listdir(anomaly_path35))}")

    Path('./Br35H/dataset/test/anomaly').mkdir(parents=True, exist_ok=True)

    flist = [f for f in os.listdir('./Br35H/dataset/test/anomaly')]
    for f in flist:
        os.remove(os.path.join('./Br35H/dataset/test/anomaly', f))

    anom35 = os.listdir(anomaly_path35)
    for f in anom35:
        shutil.copy2(os.path.join(anomaly_path35, f), './Br35H/dataset/test/anomaly')

    len(os.listdir('./Br35H/dataset/test/anomaly'))

    normal35 = os.listdir(normal_path35)
    random.shuffle(normal35)
    ratio = 0.7
    sep = round(len(normal35) * ratio)

    Path('./Br35H/dataset/test/normal').mkdir(parents=True, exist_ok=True)
    Path('./Br35H/dataset/train/normal').mkdir(parents=True, exist_ok=True)

    flist = [f for f in os.listdir('./Br35H/dataset/test/normal')]
    for f in flist:
        os.remove(os.path.join('./Br35H/dataset/test/normal', f))

    flist = [f for f in os.listdir('./Br35H/dataset/train/normal')]
    for f in flist:
        os.remove(os.path.join('./Br35H/dataset/train/normal', f))

    for f in normal35[:sep]:
        shutil.copy2(os.path.join(normal_path35, f), './Br35H/dataset/train/normal')
    for f in normal35[sep:]:
        shutil.copy2(os.path.join(normal_path35, f), './Br35H/dataset/test/normal')

    len(os.listdir('./Br35H/dataset/test/normal')), len(os.listdir('./Br35H/dataset/train/normal'))

class Br35H_Dataset(Dataset):
    def __init__(self, image_path, labels, count=-1):

        use_imagenet = True
        val_transforms_list = [
            transforms.Resize((384, 384)) if use_imagenet else transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]
        val_transforms = Compose(val_transforms_list)
        self.transform = val_transforms

        self.image_files = image_path
        self.labels = labels
        if count != -1:
            if count<len(self.image_files):
                self.image_files = self.image_files[:count]
                self.labels = self.labels[:count]
            else:
                t = len(self.image_files)
                for i in range(count-t):
                    self.image_files.append(random.choice(self.image_files[:t]))
                    self.labels.append(random.choice(self.labels[:t]))

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[index]

    def __len__(self):
        return len(self.image_files)


def get_br35h_trainset():
    train_normal_path = glob('./Br35H/dataset/train/normal/*')

    brats_mod = glob('./brats/dataset/train/normal/*')

    random.seed(1)

    random_brats_images = random.sample(brats_mod, 50)
    train_normal_path.extend(random_brats_images)

    print('added 50 normal brat images')

    train_label = [0]*len(train_normal_path)
    print(f"len(train_normal_path): {len(train_normal_path)}")

    br35h_trainset = Br35H_Dataset(image_path=train_normal_path, labels=train_label)
    print("train_set shapes: ", br35h_trainset[0][0].shape)

    count_unique_labels_of_dataset(br35h_trainset, "br35h_trainset")
    visualize_random_samples_from_clean_dataset(br35h_trainset, "br35h_trainset")

    return br35h_trainset

def get_br35h_test_set():
    test_normal_path = glob('./Br35H/dataset/test/normal/*')
    test_anomaly_path = glob('./Br35H/dataset/test/anomaly/*')

    test_path = test_normal_path + test_anomaly_path
    test_label = [0]*len(test_normal_path) + [1]*len(test_anomaly_path)
    print(f"len(test_label): {len(test_label)}")
    print(f"len(test_path): {len(test_path)}")

    br35h_testset = Br35H_Dataset(image_path=test_path, labels=test_label)
    print("test_set shapes: ", br35h_testset[0][0].shape)

    count_unique_labels_of_dataset(br35h_testset, "br35h_testset")
    visualize_random_samples_from_clean_dataset(br35h_testset, "br35h_testset")

    return br35h_testset

def get_br35h_test_set_id():
    test_normal_path = glob('./Br35H/dataset/test/normal/*')

    test_path = test_normal_path
    test_label = [0] * len(test_normal_path)
    print(f"len(test_label): {len(test_label)}")
    print(f"len(test_path): {len(test_path)}")

    br35h_testset_id = Br35H_Dataset(image_path=test_path, labels=test_label)
    print("test_set shapes: ", br35h_testset_id[0][0].shape)

    count_unique_labels_of_dataset(br35h_testset_id, "br35h_testset_id")
    visualize_random_samples_from_clean_dataset(br35h_testset_id, "br35h_testset_id")

    return br35h_testset_id

def get_br35h_test_set_ood():
    test_anomaly_path = glob('./Br35H/dataset/test/anomaly/*')

    test_path = test_anomaly_path
    test_label = [1]*len(test_anomaly_path)
    print(f"len(test_label): {len(test_label)}")
    print(f"len(test_path): {len(test_path)}")

    br35h_testset_ood = Br35H_Dataset(image_path=test_path, labels=test_label)
    print("test_set shapes: ", br35h_testset_ood[0][0].shape)

    count_unique_labels_of_dataset(br35h_testset_ood, "br35h_testset_ood")
    visualize_random_samples_from_clean_dataset(br35h_testset_ood, "br35h_testset_ood")

    return br35h_testset_ood

def get_br35h_just_test():
    train_normal_path = glob('./Br35H/dataset/train/normal/*')
    test_normal_path = glob('./Br35H/dataset/test/normal/*')
    test_anomaly_path = glob('./Br35H/dataset/test/anomaly/*')

    test_path = test_normal_path + test_anomaly_path
    test_label = [0] * len(test_normal_path) + [1] * len(test_anomaly_path)

    just_test_br35h_path = train_normal_path + test_path
    just_test_br35h_label = [0] * len(train_normal_path) + test_label
    print(f"len(just_test_br35h_path): {len(just_test_br35h_path)}")
    print(f"len(just_test_br35h_label): {len(just_test_br35h_label)}")

    just_test_br35h = Br35H_Dataset(image_path=just_test_br35h_path, labels=just_test_br35h_label)
    print("just_test_br35h shapes: ", just_test_br35h[0][0].shape)

    count_unique_labels_of_dataset(just_test_br35h, "just_test_br35h")
    visualize_random_samples_from_clean_dataset(just_test_br35h, "just_test_br35h")

    return just_test_br35h

# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
# ])

# print("len(test_set), len(train_set), len(just_test35): ", len(test_set), len(train_set), len(just_test35test_set))

# batch_size = 128
#
# train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, batch_size=batch_size)
# x, y = next(iter(train_loader))
# x.shape, y.shape, len(train_loader)
#
# test_loader = torch.utils.data.DataLoader(test_set, shuffle=False, batch_size=batch_size)
# x, y = next(iter(test_loader))
# x.shape, y.shape, len(test_loader)
#
# just_test35test_loader = torch.utils.data.DataLoader(just_test35test_set, shuffle=False, batch_size=batch_size)
# x, y = next(iter(just_test35test_loader))
# x.shape, y.shape, len(just_test35test_loader)

