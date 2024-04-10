import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torchvision.models as models
import torch.nn.functional as F
from PIL import ImageFilter
import random
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC
from PIL import ImageFilter, Image, ImageOps
from torchvision.datasets.folder import default_loader
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import random
from torchvision.transforms import Compose


def center_paste(large_img, small_img):
    # Calculate the center position
    large_width, large_height = large_img.size
    small_width, small_height = small_img.size

    # Calculate the top-left position
    left = (large_width - small_width) // 2
    top = (large_height - small_height) // 2

    # Create a copy of the large image to keep the original unchanged
    result_img = large_img.copy()

    # Paste the small image onto the large one at the calculated position
    result_img.paste(small_img, (left, top))

    return result_img

class IMAGENET30_TEST_DATASET(Dataset):
    def __init__(self, root_dir="/kaggle/input/imagenet30-dataset/one_class_test/one_class_test/", transform=None):
        """
        Args:
            root_dir (string): Directory with all the classes.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.img_path_list = []
        self.targets = []

        # Map each class to an index
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(sorted(os.listdir(root_dir)))}
        # print(f"self.class_to_idx in ImageNet30_Test_Dataset:\n{self.class_to_idx}")

        # Walk through the directory and collect information about the images and their labels
        for i, class_name in enumerate(os.listdir(root_dir)):
            class_path = os.path.join(root_dir, class_name)
            for instance_folder in os.listdir(class_path):
                instance_path = os.path.join(class_path, instance_folder)
                if instance_path != "/kaggle/input/imagenet30-dataset/one_class_test/one_class_test/airliner/._1.JPEG":
                    for img_name in os.listdir(instance_path):
                        if img_name.endswith('.JPEG'):
                            img_path = os.path.join(instance_path, img_name)
                            # image = Image.open(img_path).convert('RGB')
                            self.img_path_list.append(img_path)
                            self.targets.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, idx):
        img_path = self.img_path_list[idx]
        image = default_loader(img_path)
        label = self.targets[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

class MVTEC(Dataset):
    """`MVTEC <https://www.mvtec.com/company/research/datasets/mvtec-ad/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directories
            ``bottle``, ``cable``, etc., exists.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        resize (int, optional): Desired output image size.
        interpolation (int, optional): Interpolation method for downsizing image.
        category: bottle, cable, capsule, etc.
    """

    def __init__(self, root, train=True, target_transform=None,
                 category='carpet', resize=None, select_random_image_from_imagenet=False, shrink_factor=1.0):
        self.root = os.path.expanduser(root)
        use_imagenet = True
        val_transforms_list = [
            transforms.Resize((384, 384)) if use_imagenet else transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]
        val_transforms = Compose(val_transforms_list)
        self.transform = val_transforms
        self.target_transform = target_transform
        self.train = train
        self.resize = resize
        self.resize = int(resize * shrink_factor)
        self.select_random_image_from_imagenet = select_random_image_from_imagenet

        self.imagenet30_testset = IMAGENET30_TEST_DATASET()

        # load images for training
        if self.train:
            self.train_data = []
            self.train_labels = []
            cwd = os.getcwd()
            trainFolder = self.root + '/' + category + '/train/good/'
            os.chdir(trainFolder)
            filenames = [f.name for f in os.scandir()]
            for file in filenames:
                img = mpimg.imread(file)
                img = img * 255
                img = img.astype(np.uint8)
                self.train_data.append(img)
                self.train_labels.append(0)
            os.chdir(cwd)

            self.train_data = np.array(self.train_data)
        else:
            # load images for testing
            self.test_data = []
            self.test_labels = []

            cwd = os.getcwd()
            testFolder = self.root + '/' + category + '/test/'
            os.chdir(testFolder)
            subfolders = [sf.name for sf in os.scandir() if sf.is_dir()]
            #             print(subfolders)
            cwsd = os.getcwd()

            # for every subfolder in test folder
            for subfolder in subfolders:
                label = 0
                if subfolder == 'good':
                    label = 1
                testSubfolder = testFolder + subfolder + '/'
                #                 print(testSubfolder)
                os.chdir(testSubfolder)
                filenames = [f.name for f in os.scandir()]
                for file in filenames:
                    img = mpimg.imread(file)
                    img = img * 255
                    img = img.astype(np.uint8)
                    self.test_data.append(img)
                    self.test_labels.append(label)
                os.chdir(cwsd)
            os.chdir(cwd)

            self.test_data = np.array(self.test_data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img).convert('RGB')
        # print(f"img.size: {img.size}")

        if self.select_random_image_from_imagenet:
            imagenet30_img = self.imagenet30_testset[int(random.random() * len(self.imagenet30_testset))][0].resize((224, 224))
        else:
            imagenet30_img = self.imagenet30_testset[100][0].resize((224, 224))

        # if resizing image
        if self.resize is not None:
            resizeTransf = transforms.Resize(self.resize)
            img = resizeTransf(img)

        #         print(f"imagenet30_img.size: {imagenet30_img.size}")
        #         print(f"img.size: {img.size}")
        if not self.train:
            img = center_paste(imagenet30_img, img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        """
        Args:
            None
        Returns:
            int: length of array.
        """
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)



def get_mvtec_trainset(category):
    im_shape = 224
    trainset = MVTEC(root='/kaggle/input/mvtec-ad/', train=True, resize=im_shape,
                     category=category, select_random_image_from_imagenet=True)
    return trainset

def get_mvtec_testset_with_padding(category, shrink_factor=1.0):
    im_shape = 224
    testset = MVTEC(root='/kaggle/input/mvtec-ad/', train=False, resize=im_shape,
                     category=category, select_random_image_from_imagenet=True, shrink_factor=shrink_factor)
    return testset

def get_mvtec_testset_id(category, normal_label=0):
    im_shape = 224
    testset = MVTEC(root='/kaggle/input/mvtec-ad/', train=False, resize=im_shape,
                     category=category, select_random_image_from_imagenet=True)

    test_data_id = []
    test_labels_id = []

    for data, label in zip(testset.test_data, testset.test_labels):
        if label == normal_label:
            test_data_id.append(data)
            test_labels_id.append(label)

    return testset


def get_mvtec_testset_ood(category, anomaly_label=1):
    im_shape = 224
    testset = MVTEC(root='/kaggle/input/mvtec-ad/', train=False, resize=im_shape,
                    category=category, select_random_image_from_imagenet=True)

    test_data_ood = []
    test_labels_ood = []

    # Loop through the original dataset
    for data, label in zip(testset.test_data, testset.test_labels):
        if label == anomaly_label:
            # If the label is 0, add the sample and its label to the filtered lists
            test_data_ood.append(data)
            test_labels_ood.append(label)

    return testset