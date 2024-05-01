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
import pandas as pd
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.utils.data.dataset import Subset
from torchvision.transforms import Compose
from visualize.visualize_dataset import visualize_random_samples_from_clean_dataset
from visualize.count_labels import count_unique_labels_of_dataset


class Isic(Dataset):
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


def get_isic_trainset():
    test_normal_path = glob('/kaggle/input/isic-task3-dataset/dataset/train/NORMAL/*')

    test_path = test_normal_path
    test_label = [0] * len(test_normal_path)

    print(f"len(test_path): {len(test_path)}")

    isic_train = Isic(image_path=test_path, labels=test_label)
    print("test_set shapes: ", isic_train[0][0].shape)

    count_unique_labels_of_dataset(isic_train, "isic_testset_id")
    visualize_random_samples_from_clean_dataset(isic_train, "isic_testset_id")
    return isic_train


def get_isic_test_set():
    test_normal_path = glob('/kaggle/input/isic-task3-dataset/dataset/test/NORMAL/*')
    test_anomaly_path = glob('/kaggle/input/isic-task3-dataset/dataset/test/ABNORMAL/*')

    test_path = test_normal_path + test_anomaly_path
    test_label = [0] * len(test_normal_path) + [1] * len(test_anomaly_path)

    print(f"len(test_path): {len(test_path)}")

    isic_test = Isic(image_path=test_path, labels=test_label)
    print("test_set shapes: ", isic_test[0][0].shape)

    count_unique_labels_of_dataset(isic_test, "isic_test")
    visualize_random_samples_from_clean_dataset(isic_test, "isic_test")

    return isic_test


def get_isic_test_set_id():
    test_normal_path = glob('/kaggle/input/isic-task3-dataset/dataset/test/NORMAL/*')

    test_path = test_normal_path
    test_label = [0] * len(test_normal_path)
    print(f"len(test_label): {len(test_label)}")
    print(f"len(test_path): {len(test_path)}")

    isic_test_id = Isic(image_path=test_path, labels=test_label)
    print("test_set shapes: ", isic_test_id[0][0].shape)

    count_unique_labels_of_dataset(isic_test_id, "isic_test_id")
    visualize_random_samples_from_clean_dataset(isic_test_id, "isic_test_id")

    return isic_test_id


def get_isic_test_set_ood():
    test_anomaly_path = glob('/kaggle/input/isic-task3-dataset/dataset/test/ABNORMAL/*')

    test_path = test_anomaly_path
    test_label = [1] * len(test_anomaly_path)
    print(f"len(test_label): {len(test_label)}")
    print(f"len(test_path): {len(test_path)}")

    isic_test_ood = Isic(image_path=test_path, labels=test_label)
    print("test_set shapes: ", isic_test_ood[0][0].shape)

    count_unique_labels_of_dataset(isic_test_ood, "isic_test_ood")
    visualize_random_samples_from_clean_dataset(isic_test_ood, "isic_test_ood")

    return isic_test_ood


def get_isic_just_test_shifted():
    df = pd.read_csv('/kaggle/input/pad-ufes-20/PAD-UFES-20/metadata.csv')

    shifted_test_label = df["diagnostic"].to_numpy()
    shifted_test_label = (shifted_test_label != "NEV")

    shifted_test_path = df["img_id"].to_numpy()
    shifted_test_path = '/kaggle/input/pad-ufes-20/PAD-UFES-20/Dataset/' + shifted_test_path

    test_path = shifted_test_path
    test_label = shifted_test_label
    shifted_test = Isic(test_path, test_label)
    visualize_random_samples_from_clean_dataset(shifted_test, 'shifted test set')
    return shifted_test

