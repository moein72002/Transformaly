"""
Transformaly Utils File
"""
# from PIL import Image
import logging
import math
import sys
import os
import gc
import time
from enum import Enum
from os import listdir
from os.path import isfile, join
import pickle
import numpy as np
from tqdm import tqdm
from PIL import Image
import faiss
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pytorch_pretrained_vit.model import ViT
from torchvision.datasets import CIFAR10, CIFAR100, FashionMNIST, ImageFolder
from torchvision.transforms import Compose
from datasets.wbc1 import get_wbc1_train_and_test_dataset_for_anomaly_detection, get_wbc1_id_test_dataset, get_just_wbc1_test_dataset_for_anomaly_detection
from datasets.wbc2 import get_wbc2_train_and_test_dataset_for_anomaly_detection, get_wbc2_id_test_dataset, get_just_wbc2_test_dataset_for_anomaly_detection
from torchvision.transforms import Compose

import logging
from os.path import join
import pandas as pd
import numpy as np
from numpy.linalg import eig
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn import mixture
import torch.nn
from pytorch_pretrained_vit.model import AnomalyViT
from datasets.wbc1 import get_wbc1_train_and_test_dataset_for_anomaly_detection, get_just_wbc1_test_dataset_for_anomaly_detection
from datasets.wbc2 import get_wbc2_train_and_test_dataset_for_anomaly_detection, get_just_wbc2_test_dataset_for_anomaly_detection
from datasets.brain_datasets.Br35H import get_br35h_trainset, get_br35h_test_set, get_br35h_just_test
from datasets.brain_datasets.Brats2015 import get_brats_trainset, get_brats_testset, get_brats_just_test
from datasets.mvtec import get_mvtec_trainset, get_mvtec_testset_with_padding, get_mvtec_testset_id, get_mvtec_testset_ood



class DiorDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self,
                 image_path,
                 labels_dict_path,
                 transform=None):
        """
        Args:
            image_path (string): Path to the images.
            labels_dict_path (string): Path to the dict with annotations.
        """
        self.image_path = image_path
        self.labels_dict_path = labels_dict_path
        self.transform = transform

        with open(self.labels_dict_path, 'rb') as handle:
            self.labels_dict = pickle.load(handle)
        self.images = [f for f in listdir(image_path) if isfile(join(image_path, f))]
        self.targets = [self.labels_dict[img]['label_index'] for img in self.images]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(join(self.image_path, self.images[idx]))
        if self.transform:
            img = self.transform(img)

        label = self.targets[idx]
        return img, label


def knn_score(train_set, test_set, n_neighbours=2):
    """
    Calculates the KNN distance
    """
    index = faiss.IndexFlatL2(train_set.shape[1])
    index.add(train_set)
    dist, _ = index.search(test_set, n_neighbours)
    return np.sum(dist, axis=1)


def get_features(model, data_loader, early_break=-1):
    pretrained_features = []
    for i, (data, _) in enumerate(tqdm(data_loader)):
        if early_break > 0 and early_break < i:
            break

        encoded_outputs = model(data.to('cuda'))
        pretrained_features.append(encoded_outputs.detach().cpu().numpy())

    pretrained_features = np.concatenate(pretrained_features)
    return pretrained_features


def freeze_pretrained_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def extract_fetures(base_path,
                    data_path,
                    datasets,
                    model,
                    logging,
                    calculate_features=False,
                    manual_class_num_range=None,
                    unimodal_vals=None,
                    output_train_features=True,
                    output_test_features=True,
                    use_imagenet=False,
                    mvtec_category=None):
    if unimodal_vals is None:
        unimodal_vals = [True, False]

    BATCH_SIZE = 18
    exp_num = -1
    for dataset in datasets:
        print_and_add_to_log("=======================================", logging)
        print_and_add_to_log(f"Dataset: {dataset}", logging)
        print_and_add_to_log(f"Path: {base_path}", logging)
        print_and_add_to_log("=======================================", logging)
        exp_num += 1

        number_of_classes = get_number_of_classes(dataset)

        if manual_class_num_range is not None:
            _classes = range(*manual_class_num_range)
        else:
            if dataset in ['mvtec']:
                _classes = [0]
            elif dataset in ['br35h', 'brats2015']:
                _classes = [0]
            elif dataset in ['wbc1', 'wbc2']:
                _classes = [1]
            else:
                _classes = range(number_of_classes)

        print(f"_classes: {_classes}")
        for _class in _classes:

            # config
            for unimodal in unimodal_vals:

                print_and_add_to_log("=================================================",
                                     logging)
                print_and_add_to_log(f"Experiment number: {exp_num}", logging)
                print_and_add_to_log(f"Dataset: {dataset}", logging)
                print_and_add_to_log(f"Class: {_class}", logging)
                print_and_add_to_log(f"Unimodal setting: {unimodal}", logging)

                assert dataset in ['mvtec', 'br35h', 'brats2015', 'wbc1', 'wbc2', 'cifar10', 'cifar100', 'fmnist', 'cats_vs_dogs',
                                   'dior'], f"{dataset} not supported yet!"
                if unimodal:
                    base_feature_path = join(base_path, f'unimodal/{dataset}/class_{str(_class)}')
                else:
                    base_feature_path = join(base_path, f'multimodal/{dataset}/class_{str(_class)}')

                if not os.path.exists((base_feature_path)):
                    os.makedirs(base_feature_path, )
                else:
                    print_and_add_to_log(f"Experiment of class {_class} already exists", logging)

                if unimodal:
                    anomaly_classes = [i for i in range(number_of_classes) if i != _class]
                else:
                    anomaly_classes = [_class]

                if dataset == 'fmnist':
                    if use_imagenet:
                        val_transforms = Compose(
                            [
                                transforms.Resize((384, 384)),
                                transforms.Grayscale(num_output_channels=3),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                            ]
                        )
                    else:
                        val_transforms = Compose(
                            [
                                transforms.Resize((224, 224)),
                                transforms.Grayscale(num_output_channels=3),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                            ]
                        )
                else:
                    if use_imagenet:
                        val_transforms = Compose(
                            [
                                transforms.Resize((384, 384)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                            ]
                        )
                    else:
                        val_transforms = Compose(
                            [
                                transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                            ]
                        )

                model.eval()
                freeze_pretrained_model(model)
                model.to('cuda')

                # get dataset
                if dataset == 'mvtec':
                    trainset = get_mvtec_trainset(mvtec_category)
                    testset = get_mvtec_testset_with_padding(mvtec_category)
                    anomaly_targets = testset.test_labels
                    testset_98 = get_mvtec_testset_with_padding(mvtec_category, shrink_factor=0.98)
                    anomaly_targets_98 = testset_98.test_labels
                    testset_95 = get_mvtec_testset_with_padding(mvtec_category, shrink_factor=0.95)
                    anomaly_targets_95 = testset_95.test_labels
                    testset_90 = get_mvtec_testset_with_padding(mvtec_category, shrink_factor=0.90)
                    anomaly_targets_90 = testset_90.test_labels
                    testset_85 = get_mvtec_testset_with_padding(mvtec_category, shrink_factor=0.85)
                    anomaly_targets_85 = testset_85.test_labels
                    testset_80 = get_mvtec_testset_with_padding(mvtec_category, shrink_factor=0.80)
                    anomaly_targets_80 = testset_80.test_labels
                elif dataset == 'br35h':
                    trainset = get_br35h_trainset()
                    testset = get_br35h_test_set()
                    anomaly_targets = testset.labels
                    just_testset = get_brats_just_test()
                    just_test_anomaly_targets = just_testset.labels
                elif dataset == 'brats2015':
                    trainset = get_brats_trainset()
                    testset = get_brats_testset()
                    anomaly_targets = testset.labels
                    just_testset = get_br35h_just_test()
                    just_test_anomaly_targets = just_testset.labels
                elif dataset == 'wbc1':
                    trainset, testset = get_wbc1_train_and_test_dataset_for_anomaly_detection()
                    anomaly_targets = [0 if label == testset.normal_class_label else 1 for label in testset.targets]
                    just_testset = get_just_wbc2_test_dataset_for_anomaly_detection()
                    just_test_anomaly_targets = [0 if label == just_testset.normal_class_label else 1 for label in just_testset.targets]
                elif dataset == 'wbc2':
                    trainset, testset = get_wbc2_train_and_test_dataset_for_anomaly_detection()
                    anomaly_targets = [0 if label == testset.normal_class_label else 1 for label in testset.targets]
                    just_testset = get_just_wbc1_test_dataset_for_anomaly_detection()
                    just_test_anomaly_targets = [0 if label == just_testset.normal_class_label else 1 for label in just_testset.targets]
                else:
                    trainset_origin, testset = get_datasets(dataset, data_path, val_transforms)
                    indices = [i for i, val in enumerate(trainset_origin.targets)
                               if val not in anomaly_classes]
                    trainset = torch.utils.data.Subset(trainset_origin, indices)
                    anomaly_targets = [1 if i in anomaly_classes else 0 for i in testset.targets]

                print_and_add_to_log(f"Train dataset len: {len(trainset)}", logging)
                print_and_add_to_log(f"Test dataset len: {len(testset)}", logging)

                # Create datasetLoaders from trainset and testset
                trainsetLoader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=False)
                testsetLoader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)
                if dataset == 'mvtec':
                    testset_98_Loader = DataLoader(testset_98, batch_size=BATCH_SIZE, shuffle=False)
                    testset_95_Loader = DataLoader(testset_95, batch_size=BATCH_SIZE, shuffle=False)
                    testset_90_Loader = DataLoader(testset_90, batch_size=BATCH_SIZE, shuffle=False)
                    testset_85_Loader = DataLoader(testset_85, batch_size=BATCH_SIZE, shuffle=False)
                    testset_80_Loader = DataLoader(testset_80, batch_size=BATCH_SIZE, shuffle=False)
                if dataset in ['wbc1', 'wbc2', 'br35h', 'brats2015']:
                    just_testsetLoader = DataLoader(just_testset, batch_size=BATCH_SIZE, shuffle=False)


                extracted_features_path = join(base_feature_path, 'extracted_features')
                if not os.path.exists(extracted_features_path):
                    os.makedirs(extracted_features_path)

                print_and_add_to_log("Extracted features", logging)
                if not os.path.exists(extracted_features_path):
                    os.mkdir(extracted_features_path)

                if calculate_features or not os.path.exists(
                        join(extracted_features_path, 'train_pretrained_ViT_features.npy')):
                    if output_train_features:
                        train_features = get_features(model=model, data_loader=trainsetLoader)
                        with open(join(extracted_features_path,
                                       'train_pretrained_ViT_features.npy'), 'wb') as f:
                            np.save(f, train_features)

                    if output_test_features:
                        test_features = get_features(model=model, data_loader=testsetLoader)
                        with open(join(extracted_features_path,
                                       'test_pretrained_ViT_features.npy'), 'wb') as f:
                            np.save(f, test_features)

                        if dataset == 'mvtec':
                            test_98_features = get_features(model=model, data_loader=testset_98_Loader)
                            with open(join(extracted_features_path, f'{mvtec_category}_test_98_pretrained_ViT_features.npy'), 'wb') as f:
                                np.save(f, test_98_features)
                            test_95_features = get_features(model=model, data_loader=testset_95_Loader)
                            with open(join(extracted_features_path, f'{mvtec_category}_test_95_pretrained_ViT_features.npy'), 'wb') as f:
                                np.save(f, test_95_features)
                            test_90_features = get_features(model=model, data_loader=testset_90_Loader)
                            with open(join(extracted_features_path, f'{mvtec_category}_test_90_pretrained_ViT_features.npy'), 'wb') as f:
                                np.save(f, test_90_features)
                            test_85_features = get_features(model=model, data_loader=testset_85_Loader)
                            with open(join(extracted_features_path, f'{mvtec_category}_test_85_pretrained_ViT_features.npy'), 'wb') as f:
                                np.save(f, test_85_features)
                            test_80_features = get_features(model=model, data_loader=testset_80_Loader)
                            with open(join(extracted_features_path, f'{mvtec_category}_test_80_pretrained_ViT_features.npy'), 'wb') as f:
                                np.save(f, test_80_features)
                        if dataset in ['wbc1', 'wbc2', 'br35h', 'brats2015']:
                            just_test_features = get_features(model=model, data_loader=just_testsetLoader)
                            with open(join(extracted_features_path, 'just_test_pretrained_ViT_features.npy'), 'wb') as f:
                                np.save(f, just_test_features)

                else:
                    if output_train_features:
                        print_and_add_to_log(f"loading feature from {extracted_features_path}",
                                             logging)
                        with open(join(extracted_features_path,
                                       'train_pretrained_ViT_features.npy'), 'rb') as f:
                            train_features = np.load(f)
                    if output_test_features:
                        with open(join(extracted_features_path,
                                       f'test_pretrained_ViT_features.npy'), 'rb') as f:
                            test_features = np.load(f)

                        if dataset == "mvtec":
                            with open(join(extracted_features_path, f'{mvtec_category}_test_98_pretrained_ViT_features.npy'), 'rb') as f:
                                test_98_features = np.load(f)
                            with open(join(extracted_features_path, f'{mvtec_category}_test_95_pretrained_ViT_features.npy'), 'rb') as f:
                                test_95_features = np.load(f)
                            with open(join(extracted_features_path, f'{mvtec_category}_test_90_pretrained_ViT_features.npy'), 'rb') as f:
                                test_90_features = np.load(f)
                            with open(join(extracted_features_path, f'{mvtec_category}_test_85_pretrained_ViT_features.npy'), 'rb') as f:
                                test_85_features = np.load(f)
                            with open(join(extracted_features_path, f'{mvtec_category}_test_80_pretrained_ViT_features.npy'), 'rb') as f:
                                test_80_features = np.load(f)
                        if dataset in ['wbc1', 'wbc2', 'br35h', 'brats2015']:
                            with open(join(extracted_features_path,
                                           f'just_test_pretrained_ViT_features.npy'), 'rb') as f:
                                just_test_features = np.load(f)

                if output_train_features and output_test_features:
                    print_and_add_to_log("Calculate KNN score", logging)
                    distances = knn_score(train_features, test_features, n_neighbours=2)
                    if dataset == 'mvtec':
                        test_98_distances = knn_score(train_features, test_98_features, n_neighbours=2)
                        test_95_distances = knn_score(train_features, test_95_features, n_neighbours=2)
                        test_90_distances = knn_score(train_features, test_90_features, n_neighbours=2)
                        test_85_distances = knn_score(train_features, test_85_features, n_neighbours=2)
                        test_80_distances = knn_score(train_features, test_80_features, n_neighbours=2)
                    if dataset in ['wbc1', 'wbc2', 'br35h', 'brats2015']:
                        just_test_distances = knn_score(train_features, just_test_features, n_neighbours=2)

                    # Convert list to set to remove duplicates and count unique elements
                    unique_values = set(anomaly_targets)
                    print(f"unique_values: {unique_values}")
                    print(f'Number of unique values in anomaly_targets: {len(unique_values)}')

                    auc = roc_auc_score(anomaly_targets, distances)
                    print(f"train {dataset}, test {dataset} -> AUC: {auc}")
                    if dataset == 'mvtec':
                        test_98_auc = roc_auc_score(anomaly_targets_98, test_98_distances)
                        print(f"train mvtec_{mvtec_category}, test mvtec_{mvtec_category}_98 -> AUC: {test_98_auc}")
                        test_95_auc = roc_auc_score(anomaly_targets_95, test_95_distances)
                        print(f"train mvtec_{mvtec_category}, test mvtec_{mvtec_category}_95 -> AUC: {test_95_auc}")
                        test_90_auc = roc_auc_score(anomaly_targets_90, test_90_distances)
                        print(f"train mvtec_{mvtec_category}, test mvtec_{mvtec_category}_90 -> AUC: {test_90_auc}")
                        test_85_auc = roc_auc_score(anomaly_targets_85, test_85_distances)
                        print(f"train mvtec_{mvtec_category}, test mvtec_{mvtec_category}_85 -> AUC: {test_85_auc}")
                        test_80_auc = roc_auc_score(anomaly_targets_80, test_80_distances)
                        print(f"train mvtec_{mvtec_category}, test mvtec_{mvtec_category}_80 -> AUC: {test_80_auc}")
                    if dataset in ['wbc1', 'wbc2', 'br35h', 'brats2015']:
                        just_test_auc = roc_auc_score(just_test_anomaly_targets, just_test_distances)
                        just_test_dataset_name = "wbc2" if dataset == "wbc1" else "wbc1"
                        print(f"train {dataset}, test {just_test_dataset_name} -> AUC: {just_test_auc}")
                    print_and_add_to_log(auc, logging)


def freeze_finetuned_model(model):
    non_freezed_layer = []
    for name, param in model.named_parameters():
        if not (name.startswith('transformer.cloned_block') or name.startswith('cloned_')):
            param.requires_grad = False
        else:
            non_freezed_layer.append(name)
    print("=========================================")
    print("Clone block didn't freezed")
    print(f"layers name: {non_freezed_layer}")
    print("=========================================")
    return


def sparse2coarse(targets):
    """Convert Pytorch CIFAR100 sparse targets to coarse targets.
    Usage:
        trainset = torchvision.datasets.CIFAR100(path)
        trainset.targets = sparse2coarse(trainset.targets)
    """
    coarse_labels = np.array([4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
                              3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
                              6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
                              0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
                              5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
                              16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
                              10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
                              2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
                              16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
                              18, 1, 2, 15, 6, 0, 17, 8, 14, 13])
    return coarse_labels[targets]


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def forward_one_epoch(loader,
                      optimizer,
                      criterion,
                      net,
                      mode,
                      progress_bar_str,
                      num_of_epochs,
                      device='cuda'
                      ):
    losses = []

    for batch_idx, (inputs, targets) in enumerate(tqdm(loader)):

        if mode == Mode.training:
            optimizer.zero_grad()

        inputs = inputs.to(device)

        print(f"inputs.size: {inputs.size}")
        origin_block_outputs, cloned_block_outputs = net(inputs)
        loss = criterion(cloned_block_outputs, origin_block_outputs)
        losses.append(loss.item())

        if mode == Mode.training:
            # do a step
            loss.backward()
            optimizer.step()

        if batch_idx % 20 == 0:
            progress_bar(batch_idx, len(loader), progress_bar_str
                         % (num_of_epochs, np.mean(losses), losses[-1]))
        del inputs, origin_block_outputs, cloned_block_outputs, loss
        torch.cuda.empty_cache()

        # if batch_idx > 10:
        #     break
    return losses


def train(model, best_model, args, dataloaders,
          model_checkpoint_path,
          output_path, device='cuda',
          seed=42, anomaly_classes=None, dataset=None, _class=None, BASE_PATH=None, eval_classes=None, all_results_dict=None, mvtec_category=None):
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = model.to(device)
    best_model = best_model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
    criterion = nn.MSELoss()

    training_losses, val_losses = [], []

    training_loader = dataloaders['training']
    val_loader = dataloaders['val']
    test_loader = dataloaders['test']

    best_val_loss = np.inf

    # start training
    for epoch in range(1, args['epochs'] + 1):

        # training
        model = model.train()
        progress_bar_str = 'Teain: repeat %d -- Mean Loss: %.3f | Last Loss: %.3f'

        losses = forward_one_epoch(loader=training_loader,
                                   optimizer=optimizer,
                                   criterion=criterion,
                                   net=model,
                                   mode=Mode.training,
                                   progress_bar_str=progress_bar_str,
                                   num_of_epochs=epoch)

        # save first batch loss for normalization
        train_epoch_loss = np.mean(losses)
        sys.stdout.flush()
        print()
        print(f'Train epoch {epoch}: loss {train_epoch_loss}', flush=True)
        training_losses.append(train_epoch_loss)

        torch.cuda.empty_cache()
        torch.save(model.state_dict(), model_checkpoint_path)

        if epoch == 1 or epoch == 5:
            init_model_checkpoint_path = join(output_path,
                                              f'{epoch}_full_recon_model_state_dict.pkl')
            torch.save(model.state_dict(), init_model_checkpoint_path)

        del losses
        gc.collect()

        if (epoch - 1) % args['eval_every'] == 0:
            # validation
            model.eval()
            progress_bar_str = 'Validation: repeat %d -- Mean Loss: %.3f | Last Loss: %.3f'

            losses = forward_one_epoch(loader=val_loader,
                                       optimizer=optimizer,
                                       criterion=criterion,
                                       net=model,
                                       mode=Mode.validation,
                                       progress_bar_str=progress_bar_str,
                                       num_of_epochs=epoch
                                             )

            val_epoch_loss = np.mean(losses)
            sys.stdout.flush()

            print()
            print(f'Validation epoch {epoch // args["eval_every"]}: loss {val_epoch_loss}',
                  flush=True)
            val_losses.append(val_epoch_loss)

            #
            cur_acc_loss = {
                'training_losses': training_losses,
                'val_losses': val_losses
            }

            if best_val_loss - 0.001 > val_epoch_loss:
                best_val_loss = val_epoch_loss
                best_acc_epoch = epoch

                print(f'========== new best model! epoch {best_acc_epoch}, loss {best_val_loss}  ==========')

                best_model.load_state_dict(model.state_dict())
                # best_model = copy.deepcopy(model)
                # no_imporvement_epochs = 0
            # else:
            #     no_imporvement_epochs += 1

            del losses
            gc.collect()

            progress_bar_str = 'Test: repeat %d -- Mean Loss: %.3f | Last Loss: %.3f'
            model.eval()
            test_losses = forward_one_epoch(loader=test_loader,
                                            optimizer=None,
                                            criterion=criterion,
                                            net=model,
                                            mode=Mode.test,
                                            progress_bar_str=progress_bar_str,
                                            num_of_epochs=0)

            test_epoch_loss = np.mean(test_losses)
            print("===================== OOD val Results =====================")
            print(f'OOD val Loss : {test_epoch_loss}')
            del test_losses
            gc.collect()
            # if no_imporvement_epochs > args['early_stopping_n_epochs']:
            #     print(f"Stop due to early stopping after {no_imporvement_epochs} epochs without improvment")
            #     print(f"epoch number {epoch}")
            #     break

            if args['plot_every_layer_summarization']:
                if dataset == 'mvtec':
                    testset = get_mvtec_testset_with_padding(mvtec_category)
                    anomaly_targets = testset.test_labels
                elif dataset == 'br35h':
                    testset = get_br35h_test_set()
                    anomaly_targets = testset.labels
                elif dataset == 'brats2015':
                    testset = get_brats_testset()
                    anomaly_targets = testset.labels
                elif dataset == 'wbc1':
                    _, testset = get_wbc1_train_and_test_dataset_for_anomaly_detection()
                    anomaly_targets = [0 if label == testset.normal_class_label else 1 for label in testset.targets]
                elif dataset == 'wbc2':
                    _, testset = get_wbc2_train_and_test_dataset_for_anomaly_detection()
                    anomaly_targets = [0 if label == testset.normal_class_label else 1 for label in testset.targets]
                else:
                    _, testset = get_datasets_for_ViT(dataset=args['dataset'],
                                                      data_path = args['data_path'],
                                                      one_vs_rest=args['unimodal'],
                                                      _class=args['_class'],
                                                      normal_test_sample_only=False,
                                                      use_imagenet=args['use_imagenet']
                                                      )
                    anomaly_targets = [0 if i in anomaly_classes else 1 for i in testset.targets]

                eval_test_loader = torch.utils.data.DataLoader(testset,
                                                               batch_size=args['batch_size'],
                                                               shuffle=False)

                model = model.eval()
                outputs_recon_scores = get_finetuned_features(model,
                                                              eval_test_loader)
                outputs_recon_scores = outputs_recon_scores[0]

                print("========================================================")
                for j in range(len(args['use_layer_outputs'])):
                    layer_ind = args['use_layer_outputs'][j]
                    print(f"Layer number: {layer_ind}")
                    print(
                        f"Test Max layer outputs score: {np.max(np.abs(outputs_recon_scores[:, layer_ind]))}")
                    rot_auc = roc_auc_score(anomaly_targets,
                                            outputs_recon_scores[:, layer_ind])
                    print(f'layer AUROC score: {rot_auc}')
                    print("--------------------------------------------------------")
            model = model.train()

        if args['test_every_epoch']:
            # save models
            torch.save(best_model.state_dict(), join(output_path,
                                                     'best_full_finetuned_model_state_dict.pkl'))
            torch.save(model.state_dict(), join(output_path,
                                                'last_full_finetuned_model_state_dict.pkl'))

            if args['use_imagenet']:
                MODEL_NAME = 'B_16_imagenet1k'
            else:
                MODEL_NAME = 'B_16'

            pretrained_model_for_test = ViT(MODEL_NAME, pretrained=True)
            pretrained_model_for_test.fc = Identity()
            pretrained_model_for_test.eval()

            if args['dataset'] in ['wbc1', 'wbc2', 'br35h', 'brats2015', 'mvtec']:
                manual_class_num_range = None
            else:
                manual_class_num_range = [_class]

            extract_fetures(base_path=BASE_PATH,
                            data_path=args['data_path'],
                            datasets=[args['dataset']],
                            model=pretrained_model_for_test,
                            logging=logging,
                            calculate_features=True,
                            unimodal_vals=[args['unimodal']],
                            manual_class_num_range=manual_class_num_range,
                            output_train_features=True,
                            output_test_features=True,
                            use_imagenet=args['use_imagenet'],
                            mvtec_category=mvtec_category)

            eval_args = {
                "dataset": args["dataset"],
                "data_path": args["data_path"],
                "whitening_threshold": args["whitening_threshold_for_eval"],
                "unimodal": args["unimodal"],
                "batch_size": args["batch_size"],
                "test_every_epoch": args["test_every_epoch"],
                "use_imagenet": args['use_imagenet'],
                "use_layer_outputs": list(range(2, 12))
            }
            eval_BASE_PATH = 'experiments'
            results = evaluate_method(args=eval_args, BASE_PATH=eval_BASE_PATH, _classes=eval_classes, mvtec_category=mvtec_category)
            result_name = f"{epoch}_{dataset}_{mvtec_category}" if dataset == "mvtec" else f"{epoch}_{dataset}"
            all_results_dict[result_name] = results

    progress_bar_str = 'Test: repeat %d -- Mean Loss: %.3f | Last Loss: %.3f'

    model = model.eval()
    test_losses = forward_one_epoch(loader=test_loader,
                                    optimizer=None,
                                    criterion=criterion,
                                    net=model,
                                    mode=Mode.test,
                                    progress_bar_str=progress_bar_str,
                                    num_of_epochs=0)

    best_model = best_model.to('cpu')
    model = model.to('cpu')
    test_epoch_loss = np.mean(test_losses)
    print("===================== OOD val Results =====================")
    print(f'OOD val Loss : {test_epoch_loss}')
    return model, best_model, cur_acc_loss, all_results_dict


def get_finetuned_features(model,
                           loader,
                           seed = 42
                           ):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = model.to('cuda')
    criterion = nn.MSELoss(reduce=False)

    # start eval
    model = model.eval()
    progress_bar_str = 'Test: repeat %d -- Mean Loss: %.3f'

    all_outputs_recon_scores = []

    with torch.no_grad():
        outputs_recon_scores = []
        for batch_idx, (inputs, targets) in enumerate(tqdm(loader)):

            inputs = inputs.to('cuda')

            origin_block_outputs, cloned_block_outputs = model(inputs)
            loss = criterion(cloned_block_outputs, origin_block_outputs)
            loss = torch.mean(loss, [2, 3])
            loss = loss.permute(1, 0)
            outputs_recon_scores.extend(-1 * loss.detach().cpu().data.numpy())

            if batch_idx % 20 == 0:
                progress_bar(batch_idx, len(loader), progress_bar_str
                             % (1, np.mean(outputs_recon_scores)))

            del inputs, origin_block_outputs, cloned_block_outputs, loss
            torch.cuda.empty_cache()
        all_outputs_recon_scores.append(outputs_recon_scores)

    return np.array(all_outputs_recon_scores)


def get_transforms(dataset, use_imagenet):
    # 0.5 normalization
    if dataset == 'fmnist':
        val_transforms_list = [
            transforms.Resize((384, 384)) if use_imagenet else transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]


    else:
        val_transforms_list = [
            transforms.Resize((384, 384)) if use_imagenet else transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]

    val_transforms = Compose(val_transforms_list)
    return val_transforms


def get_number_of_classes(dataset):
    if dataset == 'mvtec':
        number_of_classes = 2

    elif dataset in ['br35h', 'brats2015']:
        number_of_classes = 2

    elif dataset in ['wbc1', 'wbc2']:
        number_of_classes = 4

    elif dataset == 'cifar10':
        number_of_classes = 10

    elif dataset == 'cifar100':
        number_of_classes = 20

    elif dataset == 'fmnist':
        number_of_classes = 10

    elif dataset == 'cats_vs_dogs':
        number_of_classes = 2

    elif dataset == 'dior':
        number_of_classes = 19

    else:
        raise ValueError(f"{dataset} not supported yet!")
    return number_of_classes


def get_datasets_for_ViT(dataset, data_path, one_vs_rest, _class,
                         normal_test_sample_only=True,
                         use_imagenet=False):
    number_of_classes = get_number_of_classes(dataset)
    if one_vs_rest:
        anomaly_classes = [i for i in range(number_of_classes) if i != _class]
    else:
        anomaly_classes = [_class]

    val_transforms = get_transforms(dataset=dataset,
                                    use_imagenet=use_imagenet)

    # get dataset
    trainset_origin, testset = get_datasets(dataset, data_path, val_transforms)

    train_indices = [i for i, val in enumerate(trainset_origin.targets)
                     if val not in anomaly_classes]
    logging.info(f"len of train dataset {len(train_indices)}")
    trainset = torch.utils.data.Subset(trainset_origin, train_indices)

    if normal_test_sample_only:
        test_indices = [i for i, val in enumerate(testset.targets)
                        if val not in anomaly_classes]
        testset = torch.utils.data.Subset(testset, test_indices)

    logging.info(f"len of test dataset {len(testset)}")
    return trainset, testset


def print_and_add_to_log(msg, logging):
    print(msg)
    logging.info(msg)


def get_datasets(dataset, data_path, val_transforms):
    if dataset == 'cifar100':
        testset = CIFAR100(root=data_path,
                           train=False, download=True,
                           transform=val_transforms)

        trainset = CIFAR100(root=data_path,
                            train=True, download=True,
                            transform=val_transforms)

        trainset.targets = sparse2coarse(trainset.targets)
        testset.targets = sparse2coarse(testset.targets)

    elif dataset == 'cifar10':
        testset = CIFAR10(root=data_path,
                          train=False, download=True,
                          transform=val_transforms)

        trainset = CIFAR10(root=data_path,
                           train=True, download=True,
                           transform=val_transforms)


    elif dataset == 'fmnist':
        trainset = FashionMNIST(root=data_path,
                                train=True, download=True,
                                transform=val_transforms)

        testset = FashionMNIST(root=data_path,
                               train=False, download=True,
                               transform=val_transforms)

    elif dataset == 'cats_vs_dogs':
        trainset = ImageFolder(root=data_path,
                               transform=val_transforms)
        testset = ImageFolder(root=data_path,
                              transform=val_transforms)

    else:
        raise ValueError(f"{dataset} not supported yet!")

    return trainset, testset


class Mode(Enum):
    training = 1
    validation = 2
    test = 3


try:
    _, term_width = os.popen('stty size', 'r').read().split()
except ValueError:
    term_width = 0

term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current + 1, total))

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def plot_graphs(train_accuracies, val_accuracies, train_losses,
                val_losses, path_to_save=''):
    plot_accuracy(train_accuracies, val_accuracies, path_to_save=path_to_save)
    plot_loss(train_losses, val_losses, path_to_save=path_to_save)
    return max(val_accuracies)


def plot_accuracy(train_accuracies, val_accuracies, to_show=True,
                  label='accuracy', path_to_save=''):
    print(f'Best val accuracy was {max(val_accuracies)}, at epoch {np.argmax(val_accuracies)}')
    train_len = len(np.array(train_accuracies))
    val_len = len(np.array(val_accuracies))

    xs_train = list(range(0, train_len))

    if train_len != val_len:
        xs_val = list(range(0, train_len, math.ceil(train_len / val_len)))
    else:
        xs_val = list(range(0, train_len))

    plt.plot(xs_val, np.array(val_accuracies), label='val ' + label)
    plt.plot(xs_train, np.array(train_accuracies), label='train ' + label)
    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    if len(path_to_save) > 0:
        plt.savefig(f'{path_to_save}/accuracy_graph.png')

    if to_show:
        plt.show()


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def plot_loss(train_losses, val_losses, to_show=True,
              val_label='val loss', train_label='train loss',
              path_to_save=''):
    train_len = len(np.array(train_losses))
    val_len = len(np.array(val_losses))

    xs_train = list(range(0, train_len))
    if train_len != val_len:
        xs_val = list(range(0, train_len, int(train_len / val_len) + 1))
    else:
        xs_val = list(range(0, train_len))

    plt.plot(xs_val, np.array(val_losses), label=val_label)
    plt.plot(xs_train, np.array(train_losses), label=train_label)

    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("loss")
    if len(path_to_save) > 0:
        plt.savefig(f'{path_to_save}/loss_graph.png')
    if to_show:
        plt.show()

def evaluate_method(args=None, BASE_PATH=None, _classes=None, mvtec_category=None):
    logging.basicConfig(
        filename=join(BASE_PATH,
                      f'{"unimodal" if args["unimodal"] else "multimodal"}/{args["dataset"]}',
                      f'Eval_{args["dataset"]}_Transformaly_outputs.log'), level=logging.DEBUG)
    print_and_add_to_log("========================================================",
                         logging)
    print_and_add_to_log("Args are:", logging)
    print_and_add_to_log(args, logging)
    print_and_add_to_log("========================================================",
                         logging)

    if args['dataset']:
        results = {'class': [],
                   'pretrained_AUROC_scores': [],
                   'test_98_pretrained_AUROC_scores': [],
                   'test_95_pretrained_AUROC_scores': [],
                   'test_90_pretrained_AUROC_scores': [],
                   'test_85_pretrained_AUROC_scores': [],
                   'test_80_pretrained_AUROC_scores': [],
                   'eval_using_knn_pretrained_AUROC_scores': [],
                   'all_layers_finetuned_AUROC_scores': [],
                   'test_98_all_layers_finetuned_AUROC_scores': [],
                   'test_95_all_layers_finetuned_AUROC_scores': [],
                   'test_90_all_layers_finetuned_AUROC_scores': [],
                   'test_85_all_layers_finetuned_AUROC_scores': [],
                   'test_80_all_layers_finetuned_AUROC_scores': [],
                   'pretrained_and_finetuned_AUROC_scores': [],
                   'test_98_pretrained_and_finetuned_AUROC_scores': [],
                   'test_95_pretrained_and_finetuned_AUROC_scores': [],
                   'test_90_pretrained_and_finetuned_AUROC_scores': [],
                   'test_85_pretrained_and_finetuned_AUROC_scores': [],
                   'test_80_pretrained_and_finetuned_AUROC_scores': []}
    else:
        results = {'class': [],
                   'pretrained_AUROC_scores': [],
                   'just_test_pretrained_AUROC_scores': [],
                   'eval_using_knn_pretrained_AUROC_scores': [],
                   'just_test_using_knn_pretrained_AUROC_scores': [],
                   'all_layers_finetuned_AUROC_scores': [],
                   'just_test_all_layers_finetuned_AUROC_scores': [],
                   'pretrained_and_finetuned_AUROC_scores': [],
                   'just_test_pretrained_and_finetuned_AUROC_scores': []}

    for _class in _classes:
        print_and_add_to_log("===================================", logging)
        print_and_add_to_log(f"Class is : {_class}", logging)
        print_and_add_to_log("===================================", logging)
        args['_class'] = _class
        base_feature_path = join(
            BASE_PATH,
            f'{"unimodal" if args["unimodal"] else "multimodal"}/{args["dataset"]}/class_{_class}')
        model_path = join(base_feature_path, 'model')

        args['base_feature_path'] = base_feature_path
        args['model_path'] = model_path

        # create the relevant directories
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        if args['unimodal']:
            anomaly_classes = [i for i in _classes if i != args['_class']]
        else:
            anomaly_classes = [args['_class']]

        results['class'].append(args['_class'])

        if args['dataset'] == 'mvtec':
            trainset = get_mvtec_trainset(mvtec_category)
            testset = get_mvtec_testset_with_padding(mvtec_category)
            anomaly_targets = testset.test_labels
            testset_98 = get_mvtec_testset_with_padding(mvtec_category, shrink_factor=0.98)
            anomaly_targets_98 = testset_98.test_labels
            testset_95 = get_mvtec_testset_with_padding(mvtec_category, shrink_factor=0.95)
            anomaly_targets_95 = testset_95.test_labels
            testset_90 = get_mvtec_testset_with_padding(mvtec_category, shrink_factor=0.90)
            anomaly_targets_90 = testset_90.test_labels
            testset_85 = get_mvtec_testset_with_padding(mvtec_category, shrink_factor=0.85)
            anomaly_targets_85 = testset_85.test_labels
            testset_80 = get_mvtec_testset_with_padding(mvtec_category, shrink_factor=0.80)
            anomaly_targets_80 = testset_80.test_labels
        elif args['dataset'] == 'br35h':
            trainset = get_br35h_trainset()
            testset = get_br35h_test_set()
            anomaly_targets = testset.labels
            just_testset = get_brats_just_test()
            just_test_anomaly_targets = just_testset.labels
        elif args['dataset'] == 'brats2015':
            trainset = get_brats_trainset()
            testset = get_brats_testset()
            anomaly_targets = testset.labels
            just_testset = get_br35h_just_test()
            just_test_anomaly_targets = just_testset.labels
        elif args['dataset'] == 'wbc1':
            trainset, testset = get_wbc1_train_and_test_dataset_for_anomaly_detection()
            anomaly_targets = [0 if label == testset.normal_class_label else 1 for label in testset.targets]
            just_testset = get_just_wbc2_test_dataset_for_anomaly_detection()
            just_test_anomaly_targets = [0 if label == just_testset.normal_class_label else 1 for label in just_testset.targets]
        elif args['dataset'] == 'wbc2':
            trainset, testset = get_wbc2_train_and_test_dataset_for_anomaly_detection()
            anomaly_targets = [0 if label == testset.normal_class_label else 1 for label in testset.targets]
            just_testset = get_just_wbc1_test_dataset_for_anomaly_detection()
            just_test_anomaly_targets = [0 if label == just_testset.normal_class_label else 1 for label in just_testset.targets]
        else:
            trainset, testset = get_datasets_for_ViT(dataset=args['dataset'],
                                                     data_path=args['data_path'],
                                                     one_vs_rest=args['unimodal'],
                                                     _class=args['_class'],
                                                     normal_test_sample_only=False,
                                                     use_imagenet=args['use_imagenet'],
                                                     )
            anomaly_targets = [1 if i in anomaly_classes else 0 for i in testset.targets]

        test_loader = torch.utils.data.DataLoader(testset,
                                                  batch_size=args['batch_size'],
                                                  shuffle=False)
        if args['dataset'] == 'mvtec':
            test_98_loader = torch.utils.data.DataLoader(testset_98, batch_size=args['batch_size'], shuffle=False)
            test_95_loader = torch.utils.data.DataLoader(testset_95, batch_size=args['batch_size'], shuffle=False)
            test_90_loader = torch.utils.data.DataLoader(testset_90, batch_size=args['batch_size'], shuffle=False)
            test_85_loader = torch.utils.data.DataLoader(testset_85, batch_size=args['batch_size'], shuffle=False)
            test_80_loader = torch.utils.data.DataLoader(testset_80, batch_size=args['batch_size'], shuffle=False)
        if args['dataset'] in ['wbc1', 'wbc2', 'br35h', 'brats2015']:
            just_test_loader = torch.utils.data.DataLoader(just_testset,
                                                           batch_size=args['batch_size'],
                                                           shuffle=False)
        train_loader = torch.utils.data.DataLoader(trainset,
                                                   batch_size=args['batch_size'],
                                                   shuffle=False)

        print_and_add_to_log("=====================================================",
                             logging)

        # get ViT features
        with open(join(BASE_PATH, f'{"unimodal" if args["unimodal"] else "multimodal"}',
                       f'{args["dataset"]}/class_{args["_class"]}/extracted_features',
                       'train_pretrained_ViT_features.npy'), 'rb') as f:
            train_features = np.load(f)

        with open(join(BASE_PATH, f'{"unimodal" if args["unimodal"] else "multimodal"}',
                       f'{args["dataset"]}/class_{args["_class"]}/extracted_features',
                       'test_pretrained_ViT_features.npy'), 'rb') as f:
            test_features = np.load(f)
        if args['dataset'] == 'mvtec':
            with open(join(BASE_PATH, f'{"unimodal" if args["unimodal"] else "multimodal"}',
                           f'{args["dataset"]}/class_{args["_class"]}/extracted_features',
                           f'{mvtec_category}_test_98_pretrained_ViT_features.npy'), 'rb') as f:
                test_98_features = np.load(f)
            with open(join(BASE_PATH, f'{"unimodal" if args["unimodal"] else "multimodal"}',
                           f'{args["dataset"]}/class_{args["_class"]}/extracted_features',
                           f'{mvtec_category}_test_95_pretrained_ViT_features.npy'), 'rb') as f:
                test_95_features = np.load(f)
            with open(join(BASE_PATH, f'{"unimodal" if args["unimodal"] else "multimodal"}',
                           f'{args["dataset"]}/class_{args["_class"]}/extracted_features',
                           f'{mvtec_category}_test_90_pretrained_ViT_features.npy'), 'rb') as f:
                test_90_features = np.load(f)
            with open(join(BASE_PATH, f'{"unimodal" if args["unimodal"] else "multimodal"}',
                           f'{args["dataset"]}/class_{args["_class"]}/extracted_features',
                           f'{mvtec_category}_test_85_pretrained_ViT_features.npy'), 'rb') as f:
                test_85_features = np.load(f)
            with open(join(BASE_PATH, f'{"unimodal" if args["unimodal"] else "multimodal"}',
                           f'{args["dataset"]}/class_{args["_class"]}/extracted_features',
                           f'{mvtec_category}_test_80_pretrained_ViT_features.npy'), 'rb') as f:
                test_80_features = np.load(f)
        if args['dataset'] in ['wbc1', 'wbc2', 'br35h', 'brats2015']:
            with open(join(BASE_PATH, f'{"unimodal" if args["unimodal"] else "multimodal"}',
                           f'{args["dataset"]}/class_{args["_class"]}/extracted_features',
                           'just_test_pretrained_ViT_features.npy'), 'rb') as f:
                just_test_features = np.load(f)

        # estimate the number of components
        cov_train_features = np.cov(train_features.T)
        values, vectors = eig(cov_train_features)
        sorted_vals = sorted(values, reverse=True)
        cumsum_vals = np.cumsum(sorted_vals)
        explained_vars = cumsum_vals / cumsum_vals[-1]

        for i, explained_var in enumerate(explained_vars):
            n_components = i
            if explained_var > args['whitening_threshold']:
                break

        print_and_add_to_log("=======================", logging)
        print_and_add_to_log(f"number of components are: {n_components}", logging)
        print_and_add_to_log("=======================", logging)

        pca = PCA(n_components=n_components, svd_solver='full', whiten=True)
        train_features = np.ascontiguousarray(pca.fit_transform(train_features))
        test_features = np.ascontiguousarray(pca.transform(test_features))
        if args['dataset'] == 'mvtec':
            test_98_features = np.ascontiguousarray(pca.transform(test_98_features))
            test_95_features = np.ascontiguousarray(pca.transform(test_95_features))
            test_90_features = np.ascontiguousarray(pca.transform(test_90_features))
            test_85_features = np.ascontiguousarray(pca.transform(test_85_features))
            test_80_features = np.ascontiguousarray(pca.transform(test_80_features))
        if args['dataset'] in ['wbc1', 'wbc2', 'br35h', 'brats2015']:
            just_test_features = np.ascontiguousarray(pca.transform(just_test_features))
        print_and_add_to_log("Whitening ended", logging)

        # build GMM
        dens_model = mixture.GaussianMixture(n_components=1,
                                             max_iter=1000,
                                             verbose=1,
                                             n_init=1)
        dens_model.fit(train_features)
        test_pretrained_samples_likelihood = dens_model.score_samples(test_features)
        if args['dataset'] == 'mvtec':
            test_98_pretrained_samples_likelihood = dens_model.score_samples(test_98_features)
            test_95_pretrained_samples_likelihood = dens_model.score_samples(test_95_features)
            test_90_pretrained_samples_likelihood = dens_model.score_samples(test_90_features)
            test_85_pretrained_samples_likelihood = dens_model.score_samples(test_85_features)
            test_80_pretrained_samples_likelihood = dens_model.score_samples(test_80_features)
        if args['dataset'] in ['wbc1', 'wbc2', 'br35h', 'brats2015']:
            just_test_pretrained_samples_likelihood = dens_model.score_samples(just_test_features)
        print_and_add_to_log("----------------------", logging)

        pretrained_auc = roc_auc_score(anomaly_targets, -test_pretrained_samples_likelihood)

        eval_using_knn_distances = knn_score(train_features, test_features, n_neighbours=2)
        eval_using_knn_auc = roc_auc_score(anomaly_targets, eval_using_knn_distances)
        if args['dataset'] == 'mvtec':
            test_98_pretrained_auc = roc_auc_score(anomaly_targets_98, -test_98_pretrained_samples_likelihood)
            test_95_pretrained_auc = roc_auc_score(anomaly_targets_95, -test_95_pretrained_samples_likelihood)
            test_90_pretrained_auc = roc_auc_score(anomaly_targets_90, -test_90_pretrained_samples_likelihood)
            test_85_pretrained_auc = roc_auc_score(anomaly_targets_85, -test_85_pretrained_samples_likelihood)
            test_80_pretrained_auc = roc_auc_score(anomaly_targets_80, -test_80_pretrained_samples_likelihood)
        if args['dataset'] in ['wbc1', 'wbc2', 'br35h', 'brats2015']:
            just_test_pretrained_auc = roc_auc_score(just_test_anomaly_targets, -just_test_pretrained_samples_likelihood)
            just_test_eval_using_knn_distances = knn_score(train_features, just_test_features, n_neighbours=2)
            just_test_eval_using_knn_auc = roc_auc_score(just_test_anomaly_targets, just_test_eval_using_knn_distances)

        print_and_add_to_log(f"Pretrained AUROC score is: {pretrained_auc}", logging)
        print_and_add_to_log("----------------------", logging)
        results['pretrained_AUROC_scores'].append(pretrained_auc)
        results['eval_using_knn_pretrained_AUROC_scores'].append(eval_using_knn_auc)
        if args['dataset'] == 'mvtec':
            results['test_98_pretrained_AUROC_scores'].append(test_98_pretrained_auc)
            results['test_95_pretrained_AUROC_scores'].append(test_95_pretrained_auc)
            results['test_90_pretrained_AUROC_scores'].append(test_90_pretrained_auc)
            results['test_85_pretrained_AUROC_scores'].append(test_85_pretrained_auc)
            results['test_80_pretrained_AUROC_scores'].append(test_80_pretrained_auc)
        if args['dataset'] in ['wbc1', 'wbc2', 'br35h', 'brats2015']:
            results['just_test_pretrained_AUROC_scores'].append(just_test_pretrained_auc)
            results['just_test_using_knn_pretrained_AUROC_scores'].append(just_test_eval_using_knn_auc)

        # get finetuned prediction head scores
        FINETUNED_PREDICTION_FILE_NAME = 'full_test_finetuned_scores.npy'
        if args['dataset'] == 'mvtec':
            TEST_98_FINETUNED_PREDICTION_FILE_NAME = 'full_test_98_finetuned_scores.npy'
            TEST_95_FINETUNED_PREDICTION_FILE_NAME = 'full_test_95_finetuned_scores.npy'
            TEST_90_FINETUNED_PREDICTION_FILE_NAME = 'full_test_90_finetuned_scores.npy'
            TEST_85_FINETUNED_PREDICTION_FILE_NAME = 'full_test_85_finetuned_scores.npy'
            TEST_80_FINETUNED_PREDICTION_FILE_NAME = 'full_test_80_finetuned_scores.npy'
        if args['dataset'] in ['wbc1', 'wbc2', 'br35h', 'brats2015']:
            JUST_TEST_FINETUNED_PREDICTION_FILE_NAME = 'just_full_test_finetuned_scores.npy'

        if args['use_imagenet']:
            VIT_MODEL_NAME = 'B_16_imagenet1k'
        else:
            VIT_MODEL_NAME = 'B_16'

        # load best instance
        # Build model
        if args["test_every_epoch"] or not os.path.exists(join(base_feature_path, 'features_distances',
                                   FINETUNED_PREDICTION_FILE_NAME)):

            print_and_add_to_log("Load Model", logging)
            model_checkpoint_path = join(model_path, 'best_full_finetuned_model_state_dict.pkl')
            model = AnomalyViT(VIT_MODEL_NAME, pretrained=True)
            model.fc = Identity()
            model_state_dict = torch.load(model_checkpoint_path)
            ret = model.load_state_dict(model_state_dict)
            print_and_add_to_log(
                'Missing keys when loading pretrained weights: {}'.format(ret.missing_keys),
                logging)
            print_and_add_to_log(
                'Unexpected keys when loading pretrained weights: {}'.format(ret.unexpected_keys),
                logging)
            print_and_add_to_log("model loadded from checkpoint here:", logging)
            print_and_add_to_log(model_checkpoint_path, logging)
            model = model.to('cuda')
            model.eval()

            test_finetuned_features = get_finetuned_features(model=model,
                                                             loader=test_loader)
            if args['dataset'] == 'mvtec':
                test_98_finetuned_features = get_finetuned_features(model=model, loader=test_98_loader)
                test_95_finetuned_features = get_finetuned_features(model=model, loader=test_95_loader)
                test_90_finetuned_features = get_finetuned_features(model=model, loader=test_90_loader)
                test_85_finetuned_features = get_finetuned_features(model=model, loader=test_85_loader)
                test_80_finetuned_features = get_finetuned_features(model=model, loader=test_80_loader)
            if args['dataset'] in ['wbc1', 'wbc2', 'br35h', 'brats2015']:
                just_test_finetuned_features = get_finetuned_features(model=model, loader=just_test_loader)

            if not os.path.exists(join(base_feature_path, 'features_distances')):
                os.makedirs(join(base_feature_path, 'features_distances'))
            np.save(join(base_feature_path, 'features_distances', FINETUNED_PREDICTION_FILE_NAME),
                    test_finetuned_features)
            if args['dataset'] == 'mvtec':
                np.save(join(base_feature_path, 'features_distances', TEST_98_FINETUNED_PREDICTION_FILE_NAME), test_98_finetuned_features)
                np.save(join(base_feature_path, 'features_distances', TEST_95_FINETUNED_PREDICTION_FILE_NAME), test_95_finetuned_features)
                np.save(join(base_feature_path, 'features_distances', TEST_90_FINETUNED_PREDICTION_FILE_NAME), test_90_finetuned_features)
                np.save(join(base_feature_path, 'features_distances', TEST_85_FINETUNED_PREDICTION_FILE_NAME), test_85_finetuned_features)
                np.save(join(base_feature_path, 'features_distances', TEST_80_FINETUNED_PREDICTION_FILE_NAME), test_80_finetuned_features)
            if args['dataset'] in ['wbc1', 'wbc2', 'br35h', 'brats2015']:
                np.save(join(base_feature_path, 'features_distances', JUST_TEST_FINETUNED_PREDICTION_FILE_NAME),
                        just_test_finetuned_features)

        else:
            test_finetuned_features = np.load(
                join(base_feature_path, 'features_distances', FINETUNED_PREDICTION_FILE_NAME))
            if args['dataset'] == 'mvtec':
                test_98_finetuned_features = np.load(join(base_feature_path, 'features_distances', TEST_98_FINETUNED_PREDICTION_FILE_NAME))
                test_95_finetuned_features = np.load(join(base_feature_path, 'features_distances', TEST_95_FINETUNED_PREDICTION_FILE_NAME))
                test_90_finetuned_features = np.load(join(base_feature_path, 'features_distances', TEST_90_FINETUNED_PREDICTION_FILE_NAME))
                test_85_finetuned_features = np.load(join(base_feature_path, 'features_distances', TEST_85_FINETUNED_PREDICTION_FILE_NAME))
                test_80_finetuned_features = np.load(join(base_feature_path, 'features_distances', TEST_80_FINETUNED_PREDICTION_FILE_NAME))
            if args['dataset'] in ['wbc1', 'wbc2', 'br35h', 'brats2015']:
                just_test_finetuned_features = np.load(join(base_feature_path, 'features_distances', JUST_TEST_FINETUNED_PREDICTION_FILE_NAME))

        if test_finetuned_features.shape[0] == 1:
            test_finetuned_features = test_finetuned_features[0]

        if args['dataset'] == 'mvtec':
            if test_98_finetuned_features.shape[0] == 1:
                test_98_finetuned_features = test_98_finetuned_features[0]
            if test_95_finetuned_features.shape[0] == 1:
                test_95_finetuned_features = test_95_finetuned_features[0]
            if test_90_finetuned_features.shape[0] == 1:
                test_90_finetuned_features = test_90_finetuned_features[0]
            if test_85_finetuned_features.shape[0] == 1:
                test_85_finetuned_features = test_85_finetuned_features[0]
            if test_80_finetuned_features.shape[0] == 1:
                test_80_finetuned_features = test_80_finetuned_features[0]
        if args['dataset'] in ['wbc1', 'wbc2', 'br35h', 'brats2015']:
            if just_test_finetuned_features.shape[0] == 1:
                just_test_finetuned_features = just_test_finetuned_features[0]

        # TODO: check below
        if args["use_layer_outputs"] is None:
            assert test_finetuned_features.shape[1] == just_test_finetuned_features.shape[1]
            args["use_layer_outputs"] = list(range(test_finetuned_features.shape[1]))

        # TODO: delete file if you want to call eval multiple times, I can add boolean as method parameter
        if args["test_every_epoch"] or not os.path.exists(join(base_feature_path,
                                   'features_distances', 'train_finetuned_features.npy')):
            print_and_add_to_log("Load Model", logging)
            model_checkpoint_path = join(model_path, 'best_full_finetuned_model_state_dict.pkl')
            model = AnomalyViT(VIT_MODEL_NAME, pretrained=True)
            model.fc = Identity()
            model_state_dict = torch.load(model_checkpoint_path)
            ret = model.load_state_dict(model_state_dict)
            print_and_add_to_log(
                'Missing keys when loading pretrained weights: {}'.format(ret.missing_keys),
                logging)
            print_and_add_to_log(
                'Unexpected keys when loading pretrained weights: {}'.format(ret.unexpected_keys),
                logging)
            print_and_add_to_log("model loadded from checkpoint here:", logging)
            print_and_add_to_log(model_checkpoint_path, logging)
            model = model.to('cuda')

            train_finetuned_features = get_finetuned_features(model=model,
                                                              loader=train_loader)
            np.save(join(base_feature_path, 'features_distances', 'train_finetuned_features.npy'),
                    train_finetuned_features)
        else:
            train_finetuned_features = np.load(join(base_feature_path, 'features_distances',
                                                    'train_finetuned_features.npy'))

        if train_finetuned_features.shape[0] == 1:
            train_finetuned_features = train_finetuned_features[0]
            print_and_add_to_log("squeeze training features", logging)

        train_finetuned_features = train_finetuned_features[:, args['use_layer_outputs']]
        test_finetuned_features = test_finetuned_features[:, args['use_layer_outputs']]

        if args['dataset'] == 'mvtec':
            test_98_finetuned_features = test_98_finetuned_features[:, args['use_layer_outputs']]
            test_95_finetuned_features = test_95_finetuned_features[:, args['use_layer_outputs']]
            test_90_finetuned_features = test_90_finetuned_features[:, args['use_layer_outputs']]
            test_85_finetuned_features = test_85_finetuned_features[:, args['use_layer_outputs']]
            test_80_finetuned_features = test_80_finetuned_features[:, args['use_layer_outputs']]
        if args['dataset'] in ['wbc1', 'wbc2', 'br35h', 'brats2015']:
            just_test_finetuned_features = just_test_finetuned_features[:, args['use_layer_outputs']]

        gmm_scores = []
        train_gmm_scores = []
        gmm = mixture.GaussianMixture(n_components=1,
                                      max_iter=500,
                                      verbose=1,
                                      n_init=1)
        gmm.fit(train_finetuned_features)
        test_finetuned_samples_likelihood = gmm.score_samples(test_finetuned_features)
        if args['dataset'] == 'mvtec':
            test_98_finetuned_samples_likelihood = gmm.score_samples(test_98_finetuned_features)
            test_95_finetuned_samples_likelihood = gmm.score_samples(test_95_finetuned_features)
            test_90_finetuned_samples_likelihood = gmm.score_samples(test_90_finetuned_features)
            test_85_finetuned_samples_likelihood = gmm.score_samples(test_85_finetuned_features)
            test_80_finetuned_samples_likelihood = gmm.score_samples(test_80_finetuned_features)
        if args['dataset'] in ['wbc1', 'wbc2', 'br35h', 'brats2015']:
            just_test_finetuned_samples_likelihood = gmm.score_samples(just_test_finetuned_features)
        # train_finetuned_samples_likelihood = gmm.score_samples(train_finetuned_features)
        # max_train_finetuned_features = np.max(np.abs(train_finetuned_samples_likelihood), axis=0)

        test_finetuned_auc = roc_auc_score(anomaly_targets, -test_finetuned_samples_likelihood)
        print_and_add_to_log(f"All Block outputs prediciton AUROC score is: {test_finetuned_auc}",
                             logging)

        if args['dataset'] == 'mvtec':
            test_98_finetuned_auc = roc_auc_score(anomaly_targets_98, -test_98_finetuned_samples_likelihood)
            print_and_add_to_log(f"test_98 all Block outputs prediciton AUROC score is: {test_98_finetuned_auc}", logging)
            test_95_finetuned_auc = roc_auc_score(anomaly_targets_95, -test_95_finetuned_samples_likelihood)
            print_and_add_to_log(f"test_95 all Block outputs prediciton AUROC score is: {test_95_finetuned_auc}", logging)
            test_90_finetuned_auc = roc_auc_score(anomaly_targets_90, -test_90_finetuned_samples_likelihood)
            print_and_add_to_log(f"test_90 all Block outputs prediciton AUROC score is: {test_90_finetuned_auc}", logging)
            test_85_finetuned_auc = roc_auc_score(anomaly_targets_85, -test_85_finetuned_samples_likelihood)
            print_and_add_to_log(f"test_85 all Block outputs prediciton AUROC score is: {test_85_finetuned_auc}", logging)
            test_80_finetuned_auc = roc_auc_score(anomaly_targets_80, -test_80_finetuned_samples_likelihood)
            print_and_add_to_log(f"test_80 all Block outputs prediciton AUROC score is: {test_80_finetuned_auc}", logging)
        if args['dataset'] in ['wbc1', 'wbc2', 'br35h', 'brats2015']:
            just_test_finetuned_auc = roc_auc_score(just_test_anomaly_targets, -just_test_finetuned_samples_likelihood)
            print_and_add_to_log(f"Just test all Block outputs prediciton AUROC score is: {just_test_finetuned_auc}", logging)
        results['all_layers_finetuned_AUROC_scores'].append(test_finetuned_auc)
        if args['dataset'] == 'mvtec':
            results['test_98_all_layers_finetuned_AUROC_scores'].append(test_98_finetuned_auc)
            results['test_95_all_layers_finetuned_AUROC_scores'].append(test_95_finetuned_auc)
            results['test_90_all_layers_finetuned_AUROC_scores'].append(test_90_finetuned_auc)
            results['test_85_all_layers_finetuned_AUROC_scores'].append(test_85_finetuned_auc)
            results['test_80_all_layers_finetuned_AUROC_scores'].append(test_80_finetuned_auc)
        if args['dataset'] in ['wbc1', 'wbc2', 'br35h', 'brats2015']:
            results['just_test_all_layers_finetuned_AUROC_scores'].append(just_test_finetuned_auc)

        print_and_add_to_log("----------------------", logging)

        finetuned_and_pretrained_samples_likelihood = np.array([
            test_finetuned_samples_likelihood[i] + test_pretrained_samples_likelihood[i] for i in
            range(len(test_pretrained_samples_likelihood))])
        if args['dataset'] == 'mvtec':
            test_98_finetuned_and_pretrained_samples_likelihood = np.array([
                test_98_finetuned_samples_likelihood[i] + test_98_pretrained_samples_likelihood[i] for i in
                range(len(test_98_pretrained_samples_likelihood))])
            test_95_finetuned_and_pretrained_samples_likelihood = np.array([
                test_95_finetuned_samples_likelihood[i] + test_95_pretrained_samples_likelihood[i] for i in
                range(len(test_95_pretrained_samples_likelihood))])
            test_90_finetuned_and_pretrained_samples_likelihood = np.array([
                test_90_finetuned_samples_likelihood[i] + test_90_pretrained_samples_likelihood[i] for i in
                range(len(test_90_pretrained_samples_likelihood))])
            test_85_finetuned_and_pretrained_samples_likelihood = np.array([
                test_85_finetuned_samples_likelihood[i] + test_85_pretrained_samples_likelihood[i] for i in
                range(len(test_85_pretrained_samples_likelihood))])
            test_80_finetuned_and_pretrained_samples_likelihood = np.array([
                test_80_finetuned_samples_likelihood[i] + test_80_pretrained_samples_likelihood[i] for i in
                range(len(test_80_pretrained_samples_likelihood))])
        if args['dataset'] in ['wbc1', 'wbc2', 'br35h', 'brats2015']:
            just_test_finetuned_and_pretrained_samples_likelihood = np.array([
                just_test_finetuned_samples_likelihood[i] + just_test_pretrained_samples_likelihood[i] for i in
                range(len(just_test_pretrained_samples_likelihood))])

        finetuned_and_pretrained_auc = roc_auc_score(anomaly_targets, -finetuned_and_pretrained_samples_likelihood)
        if args['dataset'] == 'mvtec':
            test_98_finetuned_and_pretrained_auc = roc_auc_score(anomaly_targets_98, -test_98_finetuned_and_pretrained_samples_likelihood)
            test_95_finetuned_and_pretrained_auc = roc_auc_score(anomaly_targets_95, -test_95_finetuned_and_pretrained_samples_likelihood)
            test_90_finetuned_and_pretrained_auc = roc_auc_score(anomaly_targets_90, -test_90_finetuned_and_pretrained_samples_likelihood)
            test_85_finetuned_and_pretrained_auc = roc_auc_score(anomaly_targets_85, -test_85_finetuned_and_pretrained_samples_likelihood)
            test_80_finetuned_and_pretrained_auc = roc_auc_score(anomaly_targets_80, -test_80_finetuned_and_pretrained_samples_likelihood)
        if args['dataset'] in ['wbc1', 'wbc2', 'br35h', 'brats2015']:
            just_test_finetuned_and_pretrained_auc = roc_auc_score(just_test_anomaly_targets, -just_test_finetuned_and_pretrained_samples_likelihood)
        print_and_add_to_log(
            f"The bgm and output prediction prediciton AUROC is: {finetuned_and_pretrained_auc}",
            logging)
        if args['dataset'] == 'mvtec':
            print_and_add_to_log(f"test_98: the bgm and output prediction prediciton AUROC is: {test_98_finetuned_and_pretrained_auc}", logging)
            print_and_add_to_log(f"test_95: the bgm and output prediction prediciton AUROC is: {test_95_finetuned_and_pretrained_auc}", logging)
            print_and_add_to_log(f"test_90: the bgm and output prediction prediciton AUROC is: {test_90_finetuned_and_pretrained_auc}", logging)
            print_and_add_to_log(f"test_85: the bgm and output prediction prediciton AUROC is: {test_85_finetuned_and_pretrained_auc}", logging)
            print_and_add_to_log(f"test_80: the bgm and output prediction prediciton AUROC is: {test_80_finetuned_and_pretrained_auc}", logging)
        if args['dataset'] in ['wbc1', 'wbc2', 'br35h', 'brats2015']:
            print_and_add_to_log(
                f"Just test: the bgm and output prediction prediciton AUROC is: {just_test_finetuned_and_pretrained_auc}",
                logging)

        results['pretrained_and_finetuned_AUROC_scores'].append(finetuned_and_pretrained_auc)
        if args['dataset'] == 'mvtec':
            results['test_98_pretrained_and_finetuned_AUROC_scores'].append(test_98_finetuned_and_pretrained_auc)
            results['test_95_pretrained_and_finetuned_AUROC_scores'].append(test_95_finetuned_and_pretrained_auc)
            results['test_90_pretrained_and_finetuned_AUROC_scores'].append(test_90_finetuned_and_pretrained_auc)
            results['test_85_pretrained_and_finetuned_AUROC_scores'].append(test_85_finetuned_and_pretrained_auc)
            results['test_80_pretrained_and_finetuned_AUROC_scores'].append(test_80_finetuned_and_pretrained_auc)
        if args['dataset'] in ['wbc1', 'wbc2', 'br35h', 'brats2015']:
            results['just_test_pretrained_and_finetuned_AUROC_scores'].append(just_test_finetuned_and_pretrained_auc)
    results_pd = pd.DataFrame.from_dict(results)
    print(f"all_results: {results}")
    results_dict_path = join(BASE_PATH,
                             f'summarize_results/{args["dataset"]}/{args["dataset"]}_results.csv')
    if not os.path.exists(join(BASE_PATH, f'summarize_results/{args["dataset"]}')):
        os.makedirs(join(BASE_PATH, f'summarize_results/{args["dataset"]}'))
    results_pd.to_csv(results_dict_path)

    return results
