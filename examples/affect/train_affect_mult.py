import torch
import sys
import os
import argparse
import pickle
import numpy as np

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

from training_structures.Supervised_Learning import train, test  # noqa
from fusions.mult import MULTModel  # noqa
from unimodals.common_models import Identity, MLP  # noqa
from datasets.affect.get_data import get_dataloader  # noqa
from fusions.common_fusions import Concat  # noqa
from utils.feature_extract import get_features


class HParams():
    num_heads = 8
    layers = 4
    attn_dropout = 0.1
    attn_dropout_modalities = [0, 0, 0.1]
    relu_dropout = 0.1
    res_dropout = 0.1
    out_dropout = 0.1
    embed_dropout = 0.2
    embed_dim = 24  # original: 40, %num_heads==0
    attn_mask = True
    output_dim = 1
    all_steps = False


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Dataset path and training modality')

    parser.add_argument('--data_path', type=str, help='Path to input dataset .pkl')
    parser.add_argument('--save_path', type=str, help='Path to output model file')
    parser.add_argument('--train_modal', type=int, nargs='+', help='Modal used for train model')
    # parser.add_argument('--modal_dim', type=int, nargs='+', help='Modality dimension')
    parser.add_argument('--num_epochs', type=int, help='Number of training epoch')
    parser.add_argument('--extract_features', action='store_true', help='Extract feature after training or not')

    args = parser.parse_args()
    data_path = args.data_path
    save_path = args.save_path
    train_modal = args.train_modal
    # modal_dim = args.modal_dim
    num_epochs = args.num_epochs
    extract_features = args.extract_features

    modalities = ['vision', 'audio', 'text']
    traindata, validdata, test_robust = get_dataloader(data_path, robust_test=False, max_pad=True)
    use_modal = train_modal
    post_fix = '_'.join([modalities[x] for x in use_modal])
    save_path = f"{save_path}/{post_fix}"
    os.makedirs(save_path)

    n_dims = None
    for d in test_robust:
        n_dims = [d[i].shape[-1] for i in range(3)]
        break
    modal_dim = [n_dims[i] for i in train_modal]
    encoders = [Identity().cuda(), Identity().cuda(), Identity().cuda()]
    with open(data_path, 'rb') as file:
        data = pickle.load(file)  # for multibench format

    print(f'Training modalities: {train_modal} -> {[modalities[i] for i in train_modal]} {num_epochs} epochs')
    fusion = MULTModel(len(train_modal), modal_dim, hyp_params=HParams, use_modal=train_modal, sample=True).cuda()
    head = Identity().cuda()

    dataset_name = data_path.split('/')[-1].split('.')[0]
    model_save_path = f'{save_path}/{dataset_name}_best_{post_fix}.pt'
    train(encoders, fusion, head, traindata, validdata, num_epochs, task="regression",
          optimtype=torch.optim.AdamW, early_stop=False, is_packed=False, lr=1e-4,
          clip_val=1.0, save=model_save_path,
          weight_decay=0.01, objective=torch.nn.L1Loss())

    print("Testing: ")
    model = torch.load(model_save_path).cuda()

    # Extract features
    if extract_features:
        feature_save_path = f'{save_path}/{dataset_name}_features'
        print('Extracting features for PID')
        features = get_features(model, test_robust, use_modal)
        features = [[x[0].cpu().numpy(), x[1].cpu().numpy()] for x in features]

        np.save(feature_save_path, features)
    test(model=model, test_dataloaders_all=test_robust, dataset='mosei', is_packed=False,
         criterion=torch.nn.L1Loss(), task='posneg-classification', no_robust=True, output_folder=save_path)  # posneg-classification
