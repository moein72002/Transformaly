"""
Transformaly Evaluation Script
"""
import os
import argparse
import logging
from os.path import join
import pandas as pd
import numpy as np
from numpy.linalg import eig
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn import mixture
import torch.nn
from utils import print_and_add_to_log, get_datasets_for_ViT, \
    Identity, get_finetuned_features, knn_score, evaluate_method
from pytorch_pretrained_vit.model import AnomalyViT
from datasets.wbc1 import get_wbc1_train_and_test_dataset_for_anomaly_detection, get_just_wbc1_test_dataset_for_anomaly_detection
from datasets.wbc2 import get_wbc2_train_and_test_dataset_for_anomaly_detection, get_just_wbc2_test_dataset_for_anomaly_detection



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--data_path', default='./data/', help='Path to the dataset')
    parser.add_argument('--whitening_threshold', default=0.9, type=float,
                        help='Explained variance of the whitening process')
    parser.add_argument('--unimodal', default=False, action='store_true',
                        help='Use the unimodal settings')
    parser.add_argument('--batch_size', type=int, default=6, help='Training batch size')
    parser_args = parser.parse_args()
    args = vars(parser_args)

    args['use_layer_outputs'] = list(range(2, 12))
    args['use_imagenet'] = True
    BASE_PATH = 'experiments'

    if args['dataset'] in ['wbc1', 'wbc2']:
        _classes = [1]
    elif args['dataset'] == 'cifar10':
        _classes = range(10)
    elif args['dataset'] == 'fmnist':
        _classes = range(10)
    elif args['dataset'] == 'cifar100':
        _classes = range(20)
    elif args['dataset'] == 'cats_vs_dogs':
        _classes = range(2)
    elif args['dataset'] == 'dior':
        _classes = range(19)
    else:
        raise ValueError(f"Does not support the {args['dataset']} dataset")

    # create the relevant directories
    if not os.path.exists(
            join(BASE_PATH,
                 f'{"unimodal" if args["unimodal"] else "multimodal"}/{args["dataset"]}')):
        os.makedirs(join(BASE_PATH,
                         f'{"unimodal" if args["unimodal"] else "multimodal"}/{args["dataset"]}'))

    evaluate_method(args=args, BASE_PATH=BASE_PATH, _classes=_classes)
