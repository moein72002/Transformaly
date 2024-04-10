"""
Transformaly Training Script
"""
import argparse
import logging
import pickle
import os

import torch.nn
from utils import print_and_add_to_log, get_datasets_for_ViT, \
    Identity, freeze_finetuned_model, train, plot_graphs, \
    extract_fetures
from os.path import join
from pytorch_pretrained_vit.model import AnomalyViT, ViT
from collections import defaultdict
import re
from datasets.wbc1 import get_wbc1_train_and_test_dataset_for_anomaly_detection, get_wbc1_id_test_dataset, get_wbc1_ood_test_dataset, get_just_wbc1_test_dataset_for_anomaly_detection
from datasets.wbc2 import get_wbc2_train_and_test_dataset_for_anomaly_detection, get_wbc2_id_test_dataset, get_wbc2_ood_test_dataset, get_just_wbc2_test_dataset_for_anomaly_detection
from datasets.brain_datasets.Br35H import prepare_br35h_dataset_files, get_br35h_trainset, get_br35h_test_set_id, get_br35h_test_set_ood, get_br35h_just_test
from datasets.brain_datasets.Brats2015 import prepare_brats2015_dataset_files, get_brats_trainset, get_brats_testset_id, get_brats_testset_ood, get_brats_just_test
from datasets.mvtec import get_mvtec_trainset, get_mvtec_testset_id, get_mvtec_testset_ood

def train_model(args, all_results_dict, mvtec_category=None):
    args['use_layer_outputs'] = list(range(2, 12))
    args['use_imagenet'] = True
    BASE_PATH = 'experiments'

    if args['dataset'] in ['br35h', 'brats2015']:
        _classes = [0]
        prepare_br35h_dataset_files()
        prepare_brats2015_dataset_files()
    elif args['dataset'] in ['wbc1', 'wbc2']:
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

    logging.basicConfig(
        filename=join(BASE_PATH,
                      f'{"unimodal" if args["unimodal"] else "multimodal"}/{args["dataset"]}',
                      f'Train_{args["dataset"]}_Transformaly_outputs.log'), level=logging.DEBUG)

    print_and_add_to_log("========================================================", logging)
    print_and_add_to_log("Args are:", logging)
    print_and_add_to_log(args, logging)
    print_and_add_to_log("========================================================", logging)
    results = {'class': [],
               'pretrained_AUROC_scores': [],
               'all_layers_finetuned_AUROC_scores': [],
               'pretrained_and_finetuned_AUROC_scores': []}



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

        print_and_add_to_log(
            "====================================================================",
            logging)
        print_and_add_to_log(
            "Start Training", logging)
        print_and_add_to_log(
            "====================================================================",
            logging)
        if args['dataset'] == 'mvtec':
            trainset = get_mvtec_trainset(mvtec_category)
            testset = get_mvtec_testset_id(mvtec_category)
            ood_test_set = get_mvtec_testset_ood(mvtec_category)
        elif args['dataset'] == 'br35h':
            trainset = get_br35h_trainset()
            testset = get_br35h_test_set_id()
            ood_test_set = get_br35h_test_set_ood()
        elif args['dataset'] == 'brats2015':
            trainset = get_brats_trainset()
            testset = get_brats_testset_id()
            ood_test_set = get_brats_testset_ood()
        elif args['dataset'] == 'wbc1':
            trainset, _ = get_wbc1_train_and_test_dataset_for_anomaly_detection()
            testset = get_wbc1_id_test_dataset()
            ood_test_set = get_wbc1_ood_test_dataset()
        elif args['dataset'] == 'wbc2':
            trainset, _ = get_wbc2_train_and_test_dataset_for_anomaly_detection()
            testset = get_wbc2_id_test_dataset()
            ood_test_set = get_wbc2_ood_test_dataset()
        else:
            trainset, testset = get_datasets_for_ViT(dataset=args['dataset'],
                                                     data_path=args['data_path'],
                                                     one_vs_rest=args['unimodal'],
                                                     _class=args['_class'],
                                                     normal_test_sample_only=True,
                                                     use_imagenet=args['use_imagenet']
                                                     )
        if not args['dataset'] in ['wbc1', 'wbc2', 'br35h', 'brats2015']:
            _, ood_test_set = get_datasets_for_ViT(dataset=args['dataset'],
                                                   data_path=args['data_path'],
                                                   one_vs_rest=not args['unimodal'],
                                                   _class=args['_class'],
                                                   normal_test_sample_only=True,
                                                   use_imagenet=args['use_imagenet']
                                                   )

        print_and_add_to_log("---------------", logging)
        print_and_add_to_log(f'Class size: {args["_class"]}', logging)
        print_and_add_to_log(f'Trainset size: {len(trainset)}', logging)
        print_and_add_to_log(f'Testset size: {len(testset)}', logging)
        print_and_add_to_log(f'OOD testset size: {len(ood_test_set)}', logging)
        print_and_add_to_log("---------------", logging)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args['batch_size'],
                                                   shuffle=True)
        val_loader = torch.utils.data.DataLoader(testset, batch_size=args['batch_size'],
                                                 shuffle=False)
        ood_val_loader = torch.utils.data.DataLoader(ood_test_set, batch_size=args['batch_size'],
                                                     shuffle=False)

        dataloaders = {'training': train_loader,
                       'val': val_loader,
                       'test': ood_val_loader
                       }

        # Build model
        if args['use_imagenet']:
            VIT_MODEL_NAME = 'B_16_imagenet1k'
        else:
            VIT_MODEL_NAME = 'B_16'

        # Build model
        model = AnomalyViT(VIT_MODEL_NAME, pretrained=True)
        model.fc = Identity()
        # Build model for best instance
        best_model = AnomalyViT(VIT_MODEL_NAME, pretrained=True)
        best_model.fc = Identity()

        model.to('cuda')
        best_model.to('cuda')

        model_checkpoint_path = join(model_path, 'last_full_finetuned_model_state_dict.pkl')
        if os.path.exists(model_checkpoint_path):
            model_state_dict = torch.load(model_checkpoint_path)
            model.load_state_dict(model_state_dict)
            print_and_add_to_log("model loadded from checkpoint here:", logging)
            print_and_add_to_log(model_checkpoint_path, logging)

        # freeze the model
        freeze_finetuned_model(model)
        model, best_model, cur_acc_loss, all_results_dict = train(model=model,
                                                best_model=best_model,
                                                args=args,
                                                dataloaders=dataloaders,
                                                output_path=model_path,
                                                device='cuda',
                                                seed=42,
                                                model_checkpoint_path=model_checkpoint_path,
                                                anomaly_classes=anomaly_classes,
                                                dataset=args['dataset'],
                                                _class=_class,
                                                BASE_PATH=BASE_PATH,
                                                eval_classes=_classes,
                                                all_results_dict=all_results_dict,
                                                mvtec_category=mvtec_category
                                                )

        training_losses = cur_acc_loss['training_losses']
        val_losses = cur_acc_loss['val_losses']
        try:
            plot_graphs(training_losses, val_losses, training_losses, val_losses)

        except Exception as e:
            print_and_add_to_log('raise error:', logging)
            print_and_add_to_log(e, logging)

        # save models
        torch.save(best_model.state_dict(), join(model_path,
                                                 'best_full_finetuned_model_state_dict.pkl'))
        torch.save(model.state_dict(), join(model_path,
                                            'last_full_finetuned_model_state_dict.pkl'))

        # save losses
        with open(join(model_path, 'full_finetuned_training_losses.pkl'), 'wb') as f:
            pickle.dump(training_losses, f)
        with open(join(model_path, 'full_finetuned_val_losses.pkl'), 'wb') as f:
            pickle.dump(val_losses, f)

        if args['use_imagenet']:
            MODEL_NAME = 'B_16_imagenet1k'
        else:
            MODEL_NAME = 'B_16'

        model = ViT(MODEL_NAME, pretrained=True)
        model.fc = Identity()
        model.eval()

        if args['dataset'] in ['wbc1', 'wbc2', 'br35h', 'brats2015']:
            manual_class_num_range = None
        else:
            manual_class_num_range = [_class]

        extract_fetures(base_path=BASE_PATH,
                        data_path=args['data_path'],
                        datasets=[args['dataset']],
                        model=model,
                        logging=logging,
                        calculate_features=True,
                        unimodal_vals=[args['unimodal']],
                        manual_class_num_range=manual_class_num_range,
                        output_train_features=True,
                        output_test_features=True,
                        use_imagenet=args['use_imagenet'],
                        mvtec_category=mvtec_category)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset',default='cifar10')
    parser.add_argument('--data_path', default='./data/', help='Path to the dataset')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=6, help='Training batch size')
    parser.add_argument('--lr', default=0.0001,
                        help='Learning rate value')
    parser.add_argument('--eval_every', type=int, default=2,
                        help='Will evaluate the model ever <eval_every> epochs')
    parser.add_argument('--unimodal', default=False, action='store_true',
                        help='Use the unimodal settings')
    parser.add_argument('--test_every_epoch', default=False, action='store_true',
                        help='Test every epoch or not')
    parser.add_argument('--plot_every_layer_summarization', default=False, action='store_true',
                        help='plot the per layer AUROC')
    parser.add_argument('--whitening_threshold_for_eval', default=0.9, type=float,
                        help='Explained variance of the whitening process for evaluation')
    parser.add_argument('--shrink_factor', default=1.0, type=float, help='shrink factor for mvtec')
    parser_args = parser.parse_args()
    args = vars(parser_args)

    all_results_dict = {}

    if args['dataset'] == "mvtec":
        all_entries = sorted(os.listdir("/kaggle/input/mvtec-ad/"))
        mvtec_categories = [entry for entry in all_entries if os.path.isdir(os.path.join("/kaggle/input/mvtec-ad/", entry))]
        # all_categories = sorted(os.listdir("/kaggle/input/mvtec-ad/"))
        auc_sum = 0.0
        for category in mvtec_categories:
            train_model(args, all_results_dict, mvtec_category=category)

    else:
        train_model(args, all_results_dict)

    print(f"all_results_dict: {all_results_dict}")

    if args['dataset'] == "mvtec":
        # Step 1: Group the dictionaries by the extracted number
        grouped_dicts = defaultdict(list)
        for key, value in all_results_dict.items():
            match = re.match(r'(\d+)_', key)
            if match:
                number = int(match.group(1))
                grouped_dicts[number].append(value)

        # Step 2: Calculate the average for each group
        average_dict = {}
        for number, dicts in grouped_dicts.items():
            summed_dict = defaultdict(int)
            for d in dicts:
                for k, v in d.items():
                    summed_dict[k] += v
            # Calculating the average
            averaged_values = {k: v / len(dicts) for k, v in summed_dict.items()}
            average_dict[number] = averaged_values

        # average_dict now contains the averaged values for each group
        print(f"average_dict: {average_dict}")

