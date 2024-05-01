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
import pydicom


class Chest(Dataset):
    def __init__(self, image_path, labels, count=-1, dicom=True):
        use_imagenet = True
        self.dicom = dicom
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
        if self.dicom:
            dicom = pydicom.dcmread(self.image_files[index])
            image = dicom.pixel_array

            # Convert to a PIL Image
            image = Image.fromarray(image).convert('RGB')

            # Apply the transform if it's provided
            if self.transform is not None:
                image = self.transform(image)
            return image, self.labels[index]

        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[index]

    def __len__(self):
        return len(self.image_files)


def get_rsna_trainset():
    test_normal_path = glob('/kaggle/working/train/normal/*')

    test_path = test_normal_path
    test_label = [0] * len(test_normal_path)

    print(f"len(train_path): {len(test_path)}")

    rsna_train = Chest(image_path=test_path, labels=test_label)
    print("train_set shapes: ", rsna_train[0][0].shape)

    count_unique_labels_of_dataset(rsna_train, "rsna_trainset_id")
    visualize_random_samples_from_clean_dataset(rsna_train, "rsna_train")
    return rsna_train


def get_rsna_test_set():
    test_normal_path = glob('/kaggle/working/test/normal/*')
    test_anomaly_path = glob('/kaggle/working/test/anomaly/*')

    test_path = test_normal_path + test_anomaly_path
    test_label = [0] * len(test_normal_path) + [1] * len(test_anomaly_path)

    print(f"len(test_path): {len(test_path)}")

    rsna_test = Chest(image_path=test_path, labels=test_label)
    print("test_set shapes: ", rsna_test[0][0].shape)

    count_unique_labels_of_dataset(rsna_test, "rsna_test")
    visualize_random_samples_from_clean_dataset(rsna_test, "rsna_test")

    return rsna_test


def get_rsna_test_set_id():
    test_normal_path = glob('/kaggle/working/test/normal/*')

    test_path = test_normal_path
    test_label = [0] * len(test_normal_path)
    print(f"len(test_label): {len(test_label)}")
    print(f"len(test_path): {len(test_path)}")

    rsna_test_id = Chest(image_path=test_path, labels=test_label)
    print("test_set shapes: ", rsna_test_id[0][0].shape)

    count_unique_labels_of_dataset(rsna_test_id, "rsna_test_id")
    visualize_random_samples_from_clean_dataset(rsna_test_id, "rsna_test_id")

    return rsna_test_id


def get_rsna_test_set_ood():
    test_anomaly_path = glob('/kaggle/working/test/anomaly/*')

    test_path = test_anomaly_path
    test_label = [1] * len(test_anomaly_path)
    print(f"len(test_label): {len(test_label)}")
    print(f"len(test_path): {len(test_path)}")

    rsna_test_ood = Chest(image_path=test_path, labels=test_label)
    print("test_set shapes: ", rsna_test_ood[0][0].shape)

    count_unique_labels_of_dataset(rsna_test_ood, "rsna_test_ood")
    visualize_random_samples_from_clean_dataset(rsna_test_ood, "rsna_test_ood")

    return rsna_test_ood


def get_rsna_just_test_shifted():
    test_normal_path = glob('/kaggle/working/chest_xray/test/NORMAL/*')
    test_anomaly_path = glob('/kaggle/working/chest_xray/test/PNEUMONIA/*')

    image_paths = test_normal_path + test_anomaly_path
    test_label = [0] * len(test_normal_path) + [1] * len(test_anomaly_path)

    shifted_test = Chest(image_paths, test_label, dicom=False)
    visualize_random_samples_from_clean_dataset(shifted_test, 'shifted test set')
    return shifted_test
