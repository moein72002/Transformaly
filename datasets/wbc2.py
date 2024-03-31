from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import torch.utils.data as data
import matplotlib.image as mpimg
from torchvision import transforms
import random
import torch
from torch.utils.data import Dataset
import pandas as pd
from torchvision import transforms as T

from visualize.count_labels import count_unique_labels_of_dataset

from torchvision.transforms import Compose

class WBC_dataset2(Dataset):
    def __init__(self, images_path="", csv_path="", resize=224, normal_class_label=1):
        self.path = images_path
        self.resize = resize
        self.normal_class_label = normal_class_label
        self.img_labels = pd.read_csv(csv_path)
        self.img_labels = self.img_labels[self.img_labels['class'] != 5]
        self.targets = self.img_labels['class']
        use_imagenet = True
        val_transforms_list = [
            transforms.Resize((384, 384)) if use_imagenet else transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]
        val_transforms = Compose(val_transforms_list)
        self.transform = val_transforms

    def __getitem__(self, idx):
        img_path = f"{self.path}/{str(self.img_labels.iloc[idx, 0]).zfill(3)}.bmp"
        # print(img_path)
        image = Image.open(img_path).convert('RGB')
        label = self.targets

        image = self.transform(image)

        target = 1 if label == self.normal_class_label else 0
        return image, target

    def __len__(self):
        return len(self.img_labels)

    def transform(self, img):
        pass


def get_wbc2_train_and_test_dataset_for_anomaly_detection():
    df = pd.read_csv('./wbc/segmentation_WBC/Class Labels of Dataset 2.csv')
    df = df[df['class'] != 5]
    train_data = df[df['class'] == 1].sample(n=20, random_state=2)

    df = df.drop(train_data.index)

    test_data = pd.DataFrame()
    for label in [1, 2, 3, 4]:
        class_samples = df[df['class'] == label]
        test_data = pd.concat([test_data, class_samples])

    train_data.to_csv('wbc2_train_dataset.csv', index=False)
    test_data.to_csv('wbc2_test_dataset.csv', index=False)

    train_dataset = WBC_dataset2(csv_path='wbc2_train_dataset.csv', images_path='wbc/segmentation_WBC/Dataset 2')
    test_dataset = WBC_dataset2(csv_path='wbc2_test_dataset.csv', images_path='wbc/segmentation_WBC/Dataset 2')

    count_unique_labels_of_dataset(train_dataset, "wbc2_train_dataset")
    count_unique_labels_of_dataset(test_dataset, "wbc2_test_dataset")

    return train_dataset, test_dataset

def get_wbc2_id_test_dataset(id_label=1):
    df = pd.read_csv('wbc2_test_dataset.csv')
    df = df[df['class'] != 5]

    test_data = pd.DataFrame()
    for label in [id_label]:
        class_samples = df[df['class'] == label]
        test_data = pd.concat([test_data, class_samples])

    test_data.to_csv('wbc2_id_test_dataset.csv', index=False)

    test_dataset = WBC_dataset2(csv_path='wbc2_id_test_dataset.csv', images_path='wbc/segmentation_WBC/Dataset 2')

    count_unique_labels_of_dataset(test_dataset, "wbc2_id_test_dataset")

    return test_dataset

def get_just_wbc2_test_dataset_for_anomaly_detection():
    df = pd.read_csv('./wbc/segmentation_WBC/Class Labels of Dataset 2.csv')
    df = df[df['class'] != 5]

    test_data = pd.DataFrame()
    for label in [1, 2, 3, 4]:
        class_samples = df[df['class'] == label]
        test_data = pd.concat([test_data, class_samples])

    test_data.to_csv('wbc2_just_test_dataset.csv', index=False)

    test_dataset = WBC_dataset2(csv_path='wbc2_just_test_dataset.csv', images_path='wbc/segmentation_WBC/Dataset 2')

    count_unique_labels_of_dataset(test_dataset, "wbc2_test_dataset")

    return test_dataset
