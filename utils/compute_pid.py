import numpy as np
from pid import get_measure_from_data
from .feature_extract import read_pickle
dataset_name = 'mosi'


def compute_pid(output_folder, dataset_name=dataset_name):
    features = np.load(f'{output_folder}/{dataset_name}_data_features.npy', allow_pickle=True)
    size = features[0][0].shape[1] * features[0][0].shape[2]

    features1 = np.concatenate([x[0].reshape(-1, size) for x in features])
    features2 = np.concatenate([x[1].reshape(-1, size) for x in features])

    predictions = read_pickle(f'{output_folder}/prediction.pickle')

    measure = get_measure_from_data(features1, features2, predictions.numpy(), num_cluster=20)
    return measure
