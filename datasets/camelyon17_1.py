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

class Camelyon17(Dataset):
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
            if count < len(self.image_files):
                self.image_files = self.image_files[:count]
                self.labels = self.labels[:count]
            else:
                t = len(self.image_files)
                for i in range(count - t):
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

def get_camelyon17_trainset():
    node0_train = glob('/kaggle/input/camelyon17-clean/node0/train/normal/*')
    node1_train = glob('/kaggle/input/camelyon17-clean/node1/train/normal/*')
    node2_train = glob('/kaggle/input/camelyon17-clean/node2/train/normal/*')
    
    train_normal_path = node0_train + node1_train + node2_train
    train_label = [0] * len(train_normal_path)
    print(f"len(train_normal_path): {len(train_normal_path)}")

    camelyon17_trainset = Camelyon17(image_path=train_normal_path, labels=train_label)
    print("train_set shapes: ", camelyon17_trainset[0][0].shape)

    count_unique_labels_of_dataset(camelyon17_trainset, "camelyon17 train")
    visualize_random_samples_from_clean_dataset(camelyon17_trainset, "camelyon17 train")

    return camelyon17_trainset

def get_camelyon17_test_set():
    node0_test_normal = glob('/kaggle/input/camelyon17-clean/node0/test/normal/*')
    node0_test_anomaly = glob('/kaggle/input/camelyon17-clean/node0/test/anomaly/*')

    node1_test_normal = glob('/kaggle/input/camelyon17-clean/node1/test/normal/*')
    node1_test_anomaly = glob('/kaggle/input/camelyon17-clean/node1/test/anomaly/*')

    node2_test_normal = glob('/kaggle/input/camelyon17-clean/node2/test/normal/*')
    node2_test_anomaly = glob('/kaggle/input/camelyon17-clean/node2/test/anomaly/*')

    test_path_normal = node0_test_normal + node1_test_normal + node2_test_normal
    test_path_anomaly = node0_test_anomaly + node1_test_anomaly + node2_test_anomaly

    test_path = test_path_normal + test_path_anomaly
    test_label = [0]*len(test_path_normal) + [1]*len(test_path_anomaly)
    print(f"len(test_label): {len(test_label)}")
    print(f"len(test_path): {len(test_path)}")

    camelyon17_trainset = Camelyon17(image_path=test_path, labels=test_label)
    print("test_set shapes: ", camelyon17_trainset[0][0].shape)

    count_unique_labels_of_dataset(camelyon17_trainset, "camelyon17_trainset")
    visualize_random_samples_from_clean_dataset(camelyon17_trainset, "camelyon17_trainset")

    return camelyon17_trainset

def get_camelyon_test_set_id():
    node0_test_normal = glob('/kaggle/input/camelyon17-clean/node0/test/normal/*')
    node1_test_normal = glob('/kaggle/input/camelyon17-clean/node1/test/normal/*')
    node2_test_normal = glob('/kaggle/input/camelyon17-clean/node2/test/normal/*')

    test_path = node0_test_normal + node1_test_normal + node2_test_normal
    test_label = [0] * len(test_path)
    print(f"len(test_label): {len(test_label)}")
    print(f"len(test_path): {len(test_path)}")

    camelyon17_trainset_id = Camelyon17(image_path=test_path, labels=test_label)
    print("test_set shapes: ", camelyon17_trainset_id[0][0].shape)

    count_unique_labels_of_dataset(camelyon17_trainset_id, "camelyon17_trainset_id")
    visualize_random_samples_from_clean_dataset(camelyon17_trainset_id, "camelyon17_trainset_id")

    return camelyon17_trainset_id

def get_camelyon_test_set_ood():
    node0_test_anomaly = glob('/kaggle/input/camelyon17-clean/node0/test/anomaly/*')
    node1_test_anomaly = glob('/kaggle/input/camelyon17-clean/node1/test/anomaly/*')
    node2_test_anomaly = glob('/kaggle/input/camelyon17-clean/node2/test/anomaly/*')

    test_path = node0_test_anomaly + node1_test_anomaly + node2_test_anomaly
    test_label = [1]*len(test_path)
    print(f"len(test_label): {len(test_label)}")
    print(f"len(test_path): {len(test_path)}")

    camelyon17_trainset_ood = Camelyon17(image_path=test_path, labels=test_label)
    print("test_set shapes: ", camelyon17_trainset_ood[0][0].shape)

    count_unique_labels_of_dataset(camelyon17_trainset_ood, "camelyon17_trainset_ood")
    visualize_random_samples_from_clean_dataset(camelyon17_trainset_ood, "camelyon17_trainset_ood")

    return camelyon17_trainset_ood

def get_camelyon_just_test_shifted():
    node3_test_normal = glob('/kaggle/input/camelyon17-clean/node3/test/normal/*')
    node3_test_anomaly = glob('/kaggle/input/camelyon17-clean/node3/test/anomaly/*')

    node4_test_normal = glob('/kaggle/input/camelyon17-clean/node4/test/normal/*')
    node4_test_anomaly = glob('/kaggle/input/camelyon17-clean/node4/test/anomaly/*')
    shifted_test_path_normal = node3_test_normal + node4_test_normal
    shifted_test_path_anomaly = node3_test_anomaly + node4_test_anomaly

    test_path = shifted_test_path_normal + shifted_test_path_anomaly
    test_label = [0] * len(shifted_test_path_normal) + [1] * len(shifted_test_path_anomaly)

    print(f"len(test_label_shifted): {len(test_label)}")
    print(f"len(test_path_shifted): {len(test_path)}")

    camelyon17_just_testset = Camelyon17(image_path=test_path, labels=test_label)
    print("test_set shapes: ", camelyon17_just_testset[0][0].shape)

    count_unique_labels_of_dataset(camelyon17_just_testset, "camelyon17_testset_shifted")
    visualize_random_samples_from_clean_dataset(camelyon17_just_testset, "camelyon17_testset_shifted")

    return camelyon17_just_testset