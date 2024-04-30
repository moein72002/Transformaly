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

class Aptos(Dataset):
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

def get_aptos_trainset():
    test_normal_path = glob('/kaggle/working/APTOS/train/NORMAL/*')

    test_path = test_normal_path
    test_label = [0] * len(test_normal_path)
    
    print(f"len(test_path): {len(test_path)}")

    aptos_train = Aptos(image_path=test_path, labels=test_label)
    print("test_set shapes: ", aptos_train[0][0].shape)

    count_unique_labels_of_dataset(aptos_train, "aptos_testset_id")
    visualize_random_samples_from_clean_dataset(aptos_train, "aptos_testset_id")
    return aptos_train


def get_aptos_test_set():
    test_normal_path = glob('/kaggle/working/APTOS/test/NORMAL/*')
    test_anomaly_path = glob('/kaggle/working/APTOS/test/ABNORMAL/*')


    test_path = test_normal_path + test_anomaly_path
    test_label = [0] * len(test_normal_path) + [1] * len(test_anomaly_path)

    print(f"len(test_path): {len(test_path)}")

    aptos_test = Aptos(image_path=test_path, labels=test_label)
    print("test_set shapes: ", aptos_test[0][0].shape)

    count_unique_labels_of_dataset(aptos_test, "aptos_test")
    visualize_random_samples_from_clean_dataset(aptos_test, "aptos_test")

    return aptos_test
def get_aptos_test_set_id():
    test_normal_path = glob('/kaggle/working/APTOS/test/NORMAL/*')

    test_path = test_normal_path
    test_label = [0] * len(test_normal_path)
    print(f"len(test_label): {len(test_label)}")
    print(f"len(test_path): {len(test_path)}")

    aptos_test_id = Aptos(image_path=test_path, labels=test_label)
    print("test_set shapes: ", aptos_test_id[0][0].shape)

    count_unique_labels_of_dataset(aptos_test_id, "aptos_test_id")
    visualize_random_samples_from_clean_dataset(aptos_test_id, "aptos_test_id")

    return aptos_test_id

def get_aptos_test_set_ood():
    test_anomaly_path = glob('/kaggle/working/APTOS/test/ABNORMAL/*')

    test_path = test_anomaly_path
    test_label = [1] * len(test_anomaly_path)
    print(f"len(test_label): {len(test_label)}")
    print(f"len(test_path): {len(test_path)}")

    aptos_test_ood = Aptos(image_path=test_path, labels=test_label)
    print("test_set shapes: ", aptos_test_ood[0][0].shape)

    count_unique_labels_of_dataset(aptos_test_ood, "aptos_test_ood")
    visualize_random_samples_from_clean_dataset(aptos_test_ood, "aptos_test_ood")

    return aptos_test_ood
def get_aptos_just_test_shifted():
    df = pd.read_csv('/kaggle/input/ddrdataset/DR_grading.csv')
    label = df["diagnosis"].to_numpy()
    path = df["id_code"].to_numpy()

    normal_path = path[label == 0]
    anomaly_path = path[label != 0]

    shifted_test_path = list(normal_path) + list(anomaly_path)
    shifted_test_label = [0] * len(normal_path) + [1] * len(anomaly_path)

    shifted_test_path = ["/kaggle/input/ddrdataset/DR_grading/DR_grading/" + s for s in shifted_test_path]

    test_path = shifted_test_path
    test_label = shifted_test_label
    shifted_test = Aptos(test_path, test_label)
    visualize_random_samples_from_clean_dataset(shifted_test, 'shifted test set')
    return shifted_test

