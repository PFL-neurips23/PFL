import numpy as np
import math
from config import *
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader, TensorDataset


def create_data_loaders(train_data, test_data, batch_size_train=10, batch_size_test=10):
    train_loader_list = []
    worker_train_loader_list = []
    data_iter_list = []
    data_iter_list_w = []
    all_train_data = []
    all_train_labels = []

    for user in train_data['users']:
        user_data = train_data['user_data'][user]
        x_train = torch.tensor(user_data['x'], dtype=torch.float32)
        y_train = torch.tensor(user_data['y'], dtype=torch.long)
        
        
        
        train_dataset = TensorDataset(x_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
        train_loader_list.append(train_loader)
        if 'f_' in user:  # user data
            train_loader_list.append(train_loader)
            data_iter_list.append(iter(train_loader))
            all_train_data.append(x_train)
            all_train_labels.append(y_train)
        elif 'w_' in user:  # worker data
            worker_train_loader_list.append(train_loader)
            data_iter_list_w.append(iter(train_loader))
    
    all_train_data = torch.cat(all_train_data, dim=0)
    all_train_labels = torch.cat(all_train_labels, dim=0)
    
    train_dataset = TensorDataset(all_train_data, all_train_labels)
    data_train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
    
    all_test_data = []
    all_test_labels = []

    for user in test_data['users']:
        user_data = test_data['user_data'][user]
        x_test = torch.tensor(user_data['x'], dtype=torch.float32)
        y_test = torch.tensor(user_data['y'], dtype=torch.long)

        all_test_data.append(x_test)
        all_test_labels.append(y_test)
    
#     all_test_data = torch.tensor(test_data['x'], dtype=torch.float32)
#     all_test_labels = torch.tensor(test_data['y'], dtype=torch.long)
    all_test_data = torch.cat(all_test_data, dim=0)
    all_test_labels = torch.cat(all_test_labels, dim=0)

    test_dataset = TensorDataset(all_test_data, all_test_labels)
    data_test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)

    return train_loader_list, data_iter_list, data_train_loader, data_test_loader, worker_train_loader_list, data_iter_list_w


class NodeSampler:
    def __init__(self, n_nodes, permutation=True):
        self.n_nodes = n_nodes
        self.permutation = permutation
        self.remaining_permutation = []

    def sample(self, node_sample_set, size):
        if self.permutation:
            sampled_set = []
            while len(sampled_set) < size:
                if len(self.remaining_permutation) == 0:
                    self.remaining_permutation = list(np.random.permutation(self.n_nodes))

                i = self.remaining_permutation.pop()

                if i in node_sample_set:
                    sampled_set.append(i)
        else:
            sampled_set = np.random.choice(node_sample_set, size, replace=False)

        return sampled_set


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


def mnist_partition(dataset, n_nodes,mixing_ratio):
    # TODO Check whether MNIST and CIFAR can be combined into the same function

    dict_users = {i: np.array([], dtype='int64') for i in range(n_nodes)}
    if isinstance(dataset, torch.utils.data.Subset):
        indices = dataset.indices
        labels = dataset.dataset.targets[indices].numpy()
    else:
        labels = dataset.targets.numpy()


    min_label = min(labels)
    max_label = max(labels)
    num_labels = max_label - min_label + 1

    if n_nodes > num_labels:
        label_for_nodes = []
        for n in range(0, n_nodes):
            for i in range(0, num_labels):
                if int(np.round(i * n_nodes / num_labels)) <= n < int(np.round((i + 1) * n_nodes / num_labels)):
                    label_for_nodes.append(i + min_label)
                    
    for i in range(0, len(labels)):
        if np.random.rand() <= mixing_ratio:
            tmp_target_node = np.random.randint(n_nodes)
        else:
            tmp_target_node = int((labels[i] - min_label) % n_nodes)
            if n_nodes > num_labels:
                tmp_min_index = 0
                tmp_min_val = math.inf
                for n in range(0, n_nodes):
                    if label_for_nodes[n] == labels[i] and len(dict_users[n]) < tmp_min_val:
                        tmp_min_val = len(dict_users[n])
                        tmp_min_index = n
                tmp_target_node = tmp_min_index
        dict_users[tmp_target_node] = np.concatenate((dict_users[tmp_target_node], [i]), axis=0)

    return dict_users


def cifar_partition(dataset, n_nodes,mixing_ratio):

    dict_users = {i: np.array([], dtype='int64') for i in range(n_nodes)}
    if isinstance(dataset, torch.utils.data.Subset):
        indices = dataset.indices
        labels = dataset.dataset.targets[indices].numpy()
    else:
        labels = dataset.targets.numpy()


    min_label = min(labels)
    max_label = max(labels)
    num_labels = max_label - min_label + 1

    if n_nodes > num_labels:
        label_for_nodes = []
        for n in range(0, n_nodes):
            for i in range(0, num_labels):
                if int(np.round(i * n_nodes / num_labels)) <= n < int(np.round((i + 1) * n_nodes / num_labels)):
                    label_for_nodes.append(i + min_label)

    for i in range(0, len(labels)):
        if np.random.rand() <= mixing_ratio:
            tmp_target_node = np.random.randint(n_nodes)
        else:
            tmp_target_node = int((labels[i] - min_label) % n_nodes)
            if n_nodes > num_labels:
                tmp_min_index = 0
                tmp_min_val = math.inf
                for n in range(0, n_nodes):
                    if label_for_nodes[n] == labels[i] and len(dict_users[n]) < tmp_min_val:
                        tmp_min_val = len(dict_users[n])
                        tmp_min_index = n
                tmp_target_node = tmp_min_index
        dict_users[tmp_target_node] = np.concatenate((dict_users[tmp_target_node], [i]), axis=0)

    return dict_users

# def mnist_sample(dataset, n_nodes, mixing_ratio,size):
#     # TODO Check whether MNIST and CIFAR can be combined into the same function

#     dict_users = {i: np.array([], dtype='int64') for i in range(n_nodes)}
#     if isinstance(dataset, torch.utils.data.Subset):
#         indices = dataset.indices
#         labels = dataset.dataset.targets[indices].numpy()
#     else:
#         labels = dataset.targets.numpy()


#     min_label = min(labels)
#     max_label = max(labels)
#     num_labels = max_label - min_label + 1

#     if n_nodes > num_labels:
#         label_for_nodes = []
#         for n in range(0, n_nodes):
#             for i in range(0, num_labels):
#                 if int(np.round(i * n_nodes / num_labels)) <= n < int(np.round((i + 1) * n_nodes / num_labels)):
#                     label_for_nodes.append(i + min_label)
#     else:
#         label_for_nodes = np.random.choice(num_labels, n_nodes, replace=True)              
#     for node in range(n_nodes):
#         node_majority_label = label_for_nodes[node]
#         majority_indices = np.where(labels == node_majority_label)[0]
#         other_indices = np.where(labels != node_majority_label)[0]

#         majority_size = int(size * (1 - mixing_ratio))
#         random_size = size - majority_size

#         chosen_majority_indices = np.random.choice(majority_indices, majority_size, replace=False)
#         chosen_random_indices = np.random.choice(other_indices, random_size, replace=False)

#         dict_users[node] = np.concatenate((chosen_majority_indices, chosen_random_indices))
   
#     return dict_users

def mnist_sample(dataset, n_nodes, mixing_ratio, size, all_type=True):
    dict_users = {i: np.array([], dtype='int64') for i in range(n_nodes)}
    
    if isinstance(dataset, torch.utils.data.Subset):
        indices = dataset.indices
        labels = dataset.dataset.targets[indices].numpy()
    else:
        labels = dataset.targets.numpy()

    min_label = min(labels)
    max_label = max(labels)
    num_labels = max_label - min_label + 1

    if all_type:
        majority_label = np.random.choice(num_labels) + min_label
        majority_indices = np.where(labels == majority_label)[0]
        for node in range(n_nodes):
            dict_users[node] = majority_indices
    else:
        if n_nodes > num_labels:
            label_for_nodes = []
            for n in range(0, n_nodes):
                for i in range(0, num_labels):
                    if int(np.round(i * n_nodes / num_labels)) <= n < int(np.round((i + 1) * n_nodes / num_labels)):
                        label_for_nodes.append(i + min_label)
        else:
            label_for_nodes = np.random.choice(num_labels, n_nodes, replace=True)

        for node in range(n_nodes):
            node_majority_label = label_for_nodes[node]
            majority_indices = np.where(labels == node_majority_label)[0]
            other_indices = np.where(labels != node_majority_label)[0]

            majority_size = int(size * (1 - mixing_ratio))
            random_size = size - majority_size

            chosen_majority_indices = np.random.choice(majority_indices, majority_size, replace=False)
            chosen_random_indices = np.random.choice(other_indices, random_size, replace=False)

            dict_users[node] = np.concatenate((chosen_majority_indices, chosen_random_indices))

    return dict_users
def split_data(dataset, data_train, n_nodes,typ = 'client'):
    if n_nodes > 0:
        if dataset == 'FashionMNIST' and typ == 'client' :
            dict_users = mnist_partition(data_train, n_nodes,mixing_ratio)
        elif dataset == 'CIFAR10' and typ == 'client':
            dict_users = cifar_partition(data_train, n_nodes,mixing_ratio)
        elif dataset == 'FashionMNIST' and typ == 'worker' :
            dict_users = mnist_sample(data_train, n_nodes,mixing_ratio_w,temp_worker_size)
        elif dataset == 'CIFAR10' and typ == 'worker':
            dict_users = mnist_sample(data_train, n_nodes,mixing_ratio_w,temp_worker_size)
        else:
            raise Exception('Unknown dataset name.')
    elif n_nodes == 0:
        return None
    return dict_users


