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


class Waterbird(Dataset):
    def __init__(self, train=True, count_train_landbg=-1, count_train_waterbg=-1, mode='bg_all',
                 count=-1,
                 copy=False):
        use_imagenet = True
        val_transforms_list = [
            transforms.Resize((384, 384)) if use_imagenet else transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]
        val_transforms = Compose(val_transforms_list)
        self.transform = val_transforms

        root = '/kaggle/input/waterbird/waterbird'
        df = pd.read_csv(os.path.join(root, 'metadata.csv'))

        print(len(df))

        self.train = train
        self.df = df
        lb_on_l = df[(df['y'] == 0) & (df['place'] == 0)]
        lb_on_w = df[(df['y'] == 0) & (df['place'] == 1)]
        self.normal_paths = []
        self.labels = []

        normal_df = lb_on_l.iloc[:count_train_landbg]
        normal_df_np = normal_df['img_filename'].to_numpy()
        self.normal_paths.extend([os.path.join(root, x) for x in normal_df_np][:count_train_landbg])
        normal_df = lb_on_w.iloc[:count_train_waterbg]
        normal_df_np = normal_df['img_filename'].to_numpy()
        copy_count = 1
        if copy:
            copy_count = count_train_landbg // count_train_waterbg
        for _ in range(copy_count):
            self.normal_paths.extend([os.path.join(root, x) for x in normal_df_np][:count_train_waterbg])

        if train:
            self.image_paths = self.normal_paths
        else:
            self.image_paths = []
            if mode == 'bg_all':
                dff = df
            elif mode == 'bg_water':
                dff = df[(df['place'] == 1)]
            elif mode == 'bg_land':
                dff = df[(df['place'] == 0)]
            elif mode == 'ood':
                dff = df[(df['place'] == 0 & df['y'] == 1)]
            else:
                print('Wrong mode!')
                raise ValueError('Wrong bg mode!')
            all_paths = dff[['img_filename', 'y']].to_numpy()
            for i in range(len(all_paths)):
                full_path = os.path.join(root, all_paths[i][0])
                if full_path not in self.normal_paths:
                    self.image_paths.append(full_path)
                    self.labels.append(all_paths[i][1])

        if count != -1:
            if count < len(self.image_paths):
                self.image_paths = self.image_paths[:count]
                self.labels = self.labels[:count]
            else:
                t = len(self.image_paths)
                for i in range(count - t):
                    self.image_paths.append(random.choice(self.image_paths[:t]))
                    self.labels.append(random.choice(self.labels[:t]))

    def __getitem__(self, index):
        image_file = self.image_paths[index]
        image = Image.open(image_file)
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[index]

    def __len__(self):
        return len(self.image_paths)


def get_waterbird_trainset():
    train_set = Waterbird(train=True, count_train_landbg=3500, count_train_waterbg=100, mode='bg_all')
    visualize_random_samples_from_clean_dataset(train_set, 'waterbird train set')

    return train_set


def get_waterbird_test_set():
    test_set = Waterbird(train=False, count_train_landbg=3500, count_train_waterbg=100, mode='bg_land')
    visualize_random_samples_from_clean_dataset(test_set, 'test set1 visualize')
    return test_set


def get_waterbird_test_set_id():

    return None


def get_waterbird_test_set_ood():
    test_set = Waterbird(train=False, count_train_landbg=3500, count_train_waterbg=100, mode='ood')
    return test_set


def get_waterbird_just_test_shifted():
    test_set = Waterbird(train=False, count_train_landbg=3500, count_train_waterbg=100, mode='bg_water')
    visualize_random_samples_from_clean_dataset(test_set, 'shifted test set visualization')
    return test_set

