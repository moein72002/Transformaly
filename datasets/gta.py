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


def get_cityscape_globs():
    from glob import glob
    import random
    normal_path = glob('/kaggle/input/cityscapes-5-10-threshold/cityscapes/ID/*')
    anomaly_path = glob('/kaggle/input/cityscapes-5-10-threshold/cityscapes/OOD/*')

    random.seed(42)
    random.shuffle(normal_path)
    train_ratio = 0.7
    separator = int(train_ratio * len(normal_path))
    normal_path_train = normal_path[:separator]
    normal_path_test = normal_path[separator:]

    return normal_path_train, normal_path_test, anomaly_path

def get_gta_globs():
    from glob import glob
    nums = [f'0{i}' for i in range(1, 10)] + ['10']
    folder_paths = []
    globs_id = []
    globs_ood = []
    for i in range(10):
        id_path = f'/kaggle/input/gta5-15-5-{nums[i]}/gta5_{i}/gta5_{i}/ID/*'
        ood_path = f'/kaggle/input/gta5-15-5-{nums[i]}/gta5_{i}/gta5_{i}/OOD/*'
        globs_id.append(glob(id_path))
        globs_ood.append(glob(ood_path))
        print(i, len(globs_id[-1]), len(globs_ood[-1]))

    glob_id = []
    glob_ood = []
    for i in range(len(globs_id)):
        glob_id += globs_id[i]
        glob_ood += globs_ood[i]

    random.seed(42)
    random.shuffle(glob_id)
    train_ratio = 0.7
    separator = int(train_ratio * len(glob_id))
    glob_train_id = glob_id[:separator]
    glob_test_id = glob_id[separator:]

    return glob_train_id, glob_test_id, glob_ood


class GTA(Dataset):
    def __init__(self, image_path, labels, count=-1):
        use_imagenet = True
        val_transforms_list = [
            transforms.Resize((384, 384)) if use_imagenet else transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
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


def get_gta_trainset():
    normal_path_train, normal_path_test, anomaly_path = get_cityscape_globs()
    train_label = [0] * len(normal_path_train)
    
    gta_train = GTA(normal_path_train, train_label)


    visualize_random_samples_from_clean_dataset(gta_train, "gta_testset_id")
    return gta_train


def get_gta_test_set():
    normal_path_train, normal_path_test, anomaly_path = get_cityscape_globs()
    test_path = normal_path_test + anomaly_path
    test_label = [0] * len(normal_path_test) + [1] * len(anomaly_path)
    train_label = [0] * len(normal_path_train)

    gta_test = GTA(test_path, test_label)
    visualize_random_samples_from_clean_dataset(gta_test, "gta_test")

    return gta_test


def get_gta_test_set_id():
    normal_path_train, normal_path_test, anomaly_path = get_cityscape_globs()


    gta_test_id = GTA(normal_path_test, [0]*len(normal_path_test))


    visualize_random_samples_from_clean_dataset(gta_test_id, "gta_test_id")

    return gta_test_id


def get_gta_test_set_ood():
    normal_path_train, normal_path_test, anomaly_path = get_cityscape_globs()

    gta_test_ood = GTA(anomaly_path, [1] * len(anomaly_path))

    visualize_random_samples_from_clean_dataset(gta_test_ood, "gta_test_ood")

    return gta_test_ood


def get_gta_just_test_shifted():
    glob_train_id, glob_test_id, glob_ood = get_gta_globs()
    shifted_test = GTA(image_path=glob_test_id+glob_ood, labels=[0]*len(glob_test_id)+[1]*len(glob_ood))

    visualize_random_samples_from_clean_dataset(shifted_test, 'shifted test set')
    return shifted_test

