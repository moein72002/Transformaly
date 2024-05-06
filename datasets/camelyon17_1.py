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
import pickle
class Mnist(Dataset):
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
        image = self.image_files[index]
        image = Image.fromarray(image.transpose(1, 2, 0))

        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[index]

    def __len__(self):
        return len(self.image_files)

def get_camelyon17_trainset():
    with open('./content/mnist_shifted_dataset/train_normal.pkl', 'rb') as f:
        normal_train = pickle.load(f)
    images = normal_train['images']
    labels = [0] * len(images)

    camelyon17_trainset = Mnist(image_path=images, labels=labels)
    print("train_set shapes: ", camelyon17_trainset[0][0].shape)

    count_unique_labels_of_dataset(camelyon17_trainset, "mnist train")
    visualize_random_samples_from_clean_dataset(camelyon17_trainset, "mnist train")

    return camelyon17_trainset

def get_camelyon17_test_set():
    with open('./content/mnist_shifted_dataset/test_normal_main.pkl', 'rb') as f:
        normal_test = pickle.load(f)
    with open('./content/mnist_shifted_dataset/test_abnormal_main.pkl', 'rb') as f:
        abnormal_test = pickle.load(f)
    images = normal_test['images'] + abnormal_test['images']
    labels = [0] * len(normal_test['images']) + [1] * len(abnormal_test['images'])

    camelyon17_trainset = Mnist(image_path=images, labels=labels)
    print("test_set shapes: ", camelyon17_trainset[0][0].shape)

    count_unique_labels_of_dataset(camelyon17_trainset, "mnist_trainset")
    visualize_random_samples_from_clean_dataset(camelyon17_trainset, "mnist_trainset")

    return camelyon17_trainset

def get_camelyon_test_set_id():
    with open('./content/mnist_shifted_dataset/test_normal_main.pkl', 'rb') as f:
        normal_test = pickle.load(f)
    test_path = normal_test['images']
    test_label = [0] * len(test_path)

    camelyon17_trainset_id = Mnist(image_path=test_path, labels=test_label)
    print("test_set shapes: ", camelyon17_trainset_id[0][0].shape)

    count_unique_labels_of_dataset(camelyon17_trainset_id, "mnist_trainset_id")
    visualize_random_samples_from_clean_dataset(camelyon17_trainset_id, "mnist_trainset_id")

    return camelyon17_trainset_id

def get_camelyon_test_set_ood():
    with open('./content/mnist_shifted_dataset/test_abnormal_main.pkl', 'rb') as f:
        normal_test = pickle.load(f)
    test_path = normal_test['images']
    test_label = [0] * len(test_path)

    camelyon17_trainset_ood = Mnist(image_path=test_path, labels=test_label)
    print("test_set shapes: ", camelyon17_trainset_ood[0][0].shape)

    count_unique_labels_of_dataset(camelyon17_trainset_ood, "mnist_trainset_ood")
    visualize_random_samples_from_clean_dataset(camelyon17_trainset_ood, "mnist_trainset_ood")

    return camelyon17_trainset_ood

def get_camelyon_just_test_shifted():
    with open('./content/mnist_shifted_dataset/test_normal_shifted.pkl', 'rb') as f:
        normal_test = pickle.load(f)
    with open('./content/mnist_shifted_dataset/test_abnormal_shifted.pkl', 'rb') as f:
        abnormal_test = pickle.load(f)
    images = normal_test['images'] + abnormal_test['images']
    labels = [0] * len(normal_test['images']) + [1] * len(abnormal_test['images'])

    camelyon17_just_testset = Mnist(image_path=images, labels=labels)
    print("test_set shapes: ", camelyon17_just_testset[0][0].shape)

    count_unique_labels_of_dataset(camelyon17_just_testset, "mnist_testset_shifted")
    visualize_random_samples_from_clean_dataset(camelyon17_just_testset, "mnist_testset_shifted")

    return camelyon17_just_testset