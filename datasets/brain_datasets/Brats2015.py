import pandas as pd
from glob import glob
import shutil
from pathlib import Path
import os
import random

from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import Compose
from visualize.visualize_dataset import visualize_random_samples_from_clean_dataset
from visualize.count_labels import count_unique_labels_of_dataset

def prepare_brats2015_dataset_files():
    labels = pd.read_csv('brats/Brain Tumor.csv')
    labels = labels[['Image', 'Class']]
    labels.tail() # 0: no tumor, 1: tumor

    labels.head()

    brats_path = 'brats/Brain Tumor/Brain Tumor'
    lbl = dict(zip(labels.Image, labels.Class))
    len(lbl), len(labels)

    keys = lbl.keys()
    normalbrats = [x for x in keys if lbl[x] == 0]
    anomalybrats = [x for x in keys if lbl[x] == 1]
    len(normalbrats), len(anomalybrats)

    labels['Class'].value_counts()

    Path('brats/dataset/test/anomaly').mkdir(parents=True, exist_ok=True)
    Path('brats/dataset/test/normal').mkdir(parents=True, exist_ok=True)
    Path('brats/dataset/train/normal').mkdir(parents=True, exist_ok=True)

    flist = [f for f in os.listdir('brats/dataset/test/anomaly')]
    for f in flist:
        os.remove(os.path.join('brats/dataset/test/anomaly', f))

    flist = [f for f in os.listdir('brats/dataset/test/normal')]
    for f in flist:
        os.remove(os.path.join('brats/dataset/test/normal', f))

    flist = [f for f in os.listdir('brats/dataset/train/normal')]
    for f in flist:
        os.remove(os.path.join('brats/dataset/train/normal', f))

    ratio = 0.7
    random.shuffle(normalbrats)
    bratsep = round(len(normalbrats) * ratio)

    for f in anomalybrats:
        ext = f'{f}.jpg'
        shutil.copy2(os.path.join(brats_path, ext), 'brats/dataset/test/anomaly')
    for f in normalbrats[:bratsep]:
        ext = f'{f}.jpg'
        shutil.copy2(os.path.join(brats_path, ext), 'brats/dataset/train/normal')
    for f in normalbrats[bratsep:]:
        ext = f'{f}.jpg'
        shutil.copy2(os.path.join(brats_path, ext), 'brats/dataset/test/normal')

class Brats2015_Dataset(Dataset):
    def __init__(self, image_path, labels, count=-1):
        prepare_brats2015_dataset_files()

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

def get_brats_trainset():
    normal_train_brats = glob('./brats/dataset/train/normal/*')
    normal_train_label_brats = [0] * len(normal_train_brats)
    brats_normal_train = Brats2015_Dataset(image_path=normal_train_brats, labels=normal_train_label_brats)
    print("brats_normal_train shapes: ", brats_normal_train[0][0].shape)

    count_unique_labels_of_dataset(brats_normal_train, "brats_normal_train")
    visualize_random_samples_from_clean_dataset(brats_normal_train, "brats_normal_train")

    return brats_normal_train

def get_brats_testset():
    normal_test_brats = glob('./brats/dataset/test/normal/*')
    anomaly_brats = glob('./brats/dataset/test/anomaly/*')
    test_path_brats = normal_test_brats + anomaly_brats
    test_label_brats = [0]*len(normal_test_brats) + [1]*len(anomaly_brats)
    brats_test = Brats2015_Dataset(image_path=test_path_brats, labels=test_label_brats)
    print("brats_test shapes: ", brats_test[0][0].shape)

    count_unique_labels_of_dataset(brats_test, "brats_test")
    visualize_random_samples_from_clean_dataset(brats_test, "brats_test")

    return brats_test


def get_brats_testset_id():
    normal_test_brats = glob('./brats/dataset/test/normal/*')
    test_path_brats = normal_test_brats
    test_label_brats = [0] * len(normal_test_brats)
    brats_test_id = Brats2015_Dataset(image_path=test_path_brats, labels=test_label_brats)
    print("brats_test shapes: ", brats_test_id[0][0].shape)

    count_unique_labels_of_dataset(brats_test_id, "brats_test_id")
    visualize_random_samples_from_clean_dataset(brats_test_id, "brats_test_id")

    return brats_test_id


def get_brats_testset_ood():
    anomaly_brats = glob('./brats/dataset/test/anomaly/*')
    test_path_brats = anomaly_brats
    test_label_brats = [1] * len(anomaly_brats)
    brats_test_ood = Brats2015_Dataset(image_path=test_path_brats, labels=test_label_brats)
    print("brats_test shapes: ", brats_test_ood[0][0].shape)

    count_unique_labels_of_dataset(brats_test_ood, "brats_test_ood")
    visualize_random_samples_from_clean_dataset(brats_test_ood, "brats_test_ood")

    return brats_test_ood

def get_brats_just_test():
    normal_train_brats = glob('./brats/dataset/train/normal/*')
    normal_train_label_brats = [0] * len(normal_train_brats)

    normal_test_brats = glob('./brats/dataset/test/normal/*')
    anomaly_brats = glob('./brats/dataset/test/anomaly/*')
    test_path_brats = normal_test_brats + anomaly_brats
    test_label_brats = [0] * len(normal_test_brats) + [1] * len(anomaly_brats)

    just_test_path = normal_train_brats+test_path_brats
    just_test_label = normal_train_label_brats + test_label_brats
    len(just_test_path), len(just_test_label)

    just_test_set_brats = Brats2015_Dataset(image_path=just_test_path, labels=just_test_label)
    print("just_test_set_brats shapes: ", just_test_set_brats[0][0].shape)

    count_unique_labels_of_dataset(just_test_set_brats, "just_test_set_brats")
    visualize_random_samples_from_clean_dataset(just_test_set_brats, "just_test_set_brats")

    return just_test_set_brats




# print("len(just_test_set), len(brats_train), len(brats_test):", len(just_test_set), len(brats_normal_train), len(brats_test))

# brats_train_loader = torch.utils.data.DataLoader(brats_normal_train, shuffle=True, batch_size=batch_size)
# x, y = next(iter(brats_train_loader))
# x.shape, y.shape, len(brats_train_loader)
#
# brats_test_loader = torch.utils.data.DataLoader(brats_test, shuffle=True, batch_size=batch_size)
# x, y = next(iter(brats_test_loader))
# x.shape, y.shape, len(brats_test_loader)
#
# just_test_loader = torch.utils.data.DataLoader(just_test_set, shuffle=False, batch_size=batch_size)
# x, y = next(iter(just_test_loader))
# x.shape, y.shape, len(just_test_loader)
#
# train_iter = iter(just_test_loader)
# x, y = next(train_iter)
# a=0
# display([x[i] for i in range(a, a+10)], [int(y[i]) for i in range(a, a+10)])
# a=10
# display([x[i] for i in range(a, a+10)], [int(y[i]) for i in range(a, a+10)])
# a=20
# display([x[i] for i in range(a, a+10)], [int(y[i]) for i in range(a, a+10)])
# a=30
# display([x[i] for i in range(a, a+10)], [int(y[i]) for i in range(a, a+10)])
#
# test_iter = iter(just_test_loader)
#
# for _ in range(20):
#     x, y = next(test_iter)
#
# a=0
# display([x[i] for i in range(a, a+10)], [int(y[i]) for i in range(a, a+10)])
# a=10
# display([x[i] for i in range(a, a+10)], [int(y[i]) for i in range(a, a+10)])
# a=20
# display([x[i] for i in range(a, a+10)], [int(y[i]) for i in range(a, a+10)])
# a=30
# display([x[i] for i in range(a, a+10)], [int(y[i]) for i in range(a, a+10)])