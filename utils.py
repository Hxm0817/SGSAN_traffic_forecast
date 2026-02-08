import os
import numpy as np
import pandas as pd
import torch


def load_data(dataset_name, normalized=True):
    if dataset_name == 'PEMS-Bay':
        file_path = 'data/PEMS_BAY/'
        A = np.load(os.path.join(file_path, 'pems_adj_mat.npy')).astype(np.float32)
        X = np.array(pd.read_csv(os.path.join(file_path, 'pems_bay.csv'), index_col=0))[:, :-1]
    elif 'PEMS' in dataset_name:
        file_path = f'data/{dataset_name.lower()}/'
        adj_info = np.load(os.path.join(file_path, 'adj.npz'))
        node_num = adj_info['shape'][0]
        edge_num = adj_info['row'].shape[0]
        A = np.zeros((node_num, node_num)).astype(np.float32)

        for i in range(edge_num):
            A[adj_info['row'][i]][adj_info['col'][i]] = adj_info['data'][i]

        X = np.array(pd.read_hdf(os.path.join(file_path, f'{dataset_name}.h5')))
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported.")

    keep_idx = np.where(~np.isnan(X).any(axis=0))[0]
    X = X[:, keep_idx]
    A = A[np.ix_(keep_idx, keep_idx)]

    X = X.astype(np.float32).transpose(1, 0)
    X = X.reshape(X.shape[0], 1, X.shape[1])
    # X: (num_nodes, num_features, num_timesteps)

    means = np.mean(X, axis=(0, 2))
    stds = np.std(X, axis=(0, 2))

    if normalized:
        X = (X - means.reshape(1, -1, 1)) / (stds.reshape(1, -1, 1) + 1e-6)

    return A, X, means, stds


def get_normalized_adj(A):
    # Add self-loops
    A = A + np.eye(A.shape[0], dtype=np.float32)
    D = np.sum(A, axis=1)
    D[D <= 1e-6] = 1e-6
    D_inv_sqrt = np.power(D, -0.5)
    D_mat = np.diag(D_inv_sqrt)
    return D_mat @ A @ D_mat


def generate_dataset(X, num_timesteps_input, num_timesteps_output):
    # X shape: (num_nodes, num_features, total_timesteps)
    # Return: (num_samples, num_nodes, timesteps, features)
    num_samples = X.shape[2] - (num_timesteps_input + num_timesteps_output) + 1
    features, target = [], []

    for i in range(num_samples):
        # Input: (num_nodes, input_len, features)
        input_seq = X[:, :, i: i + num_timesteps_input].transpose(0, 2, 1)
        features.append(input_seq)

        # Target: (num_nodes, output_len) - currently taking 1 step based on original code
        end_idx = i + num_timesteps_input + num_timesteps_output
        target_seq = X[:, 0, (end_idx - 1): end_idx]
        target.append(target_seq)

    return torch.tensor(np.array(features),
                        dtype=torch.float32), torch.tensor(np.array(target), dtype=torch.float32)