import torch
import numpy as np
import random
def distribute_batches_equally(train_data_loader, num_workers):
    """

    Gives each worker the same number of batches of training data.

    :param train_data_loader: Training data loader
    :type train_data_loader: torch.utils.data.DataLoader
    :param num_workers: number of workers
    :type num_workers: int
    """

    distributed_dataset = [[] for i in range(num_workers)]

    for batch_idx, (data, target) in enumerate(train_data_loader):
        worker_idx = batch_idx % num_workers

        distributed_dataset[worker_idx].append((data, target))

    return distributed_dataset

def mixture_distribution_split_noniid(train_data_loader, n_classes, num_workers, n_clusters, alpha):
    def avg_divide(l, g):
        num_elems = len(l)
        group_size = int(len(l) / g)
        num_big_groups = num_elems - g * group_size
        num_small_groups = g - num_big_groups
        glist = []
        for i in range(num_small_groups):
            glist.append(l[group_size * i: group_size * (i + 1)])
        bi = group_size * num_small_groups
        group_size += 1
        for i in range(num_big_groups):
            glist.append(l[bi + group_size * i:bi + group_size * (i + 1)])
        return glist

    def split_list_by_idcs(l, idcs):
        res = []
        current_index = 0
        for index in idcs: 
            res.append(l[current_index: index])
            current_index = index

        return res


    if n_clusters == -1:
        n_clusters = n_classes


    all_labels = list(range(n_classes))
    np.random.shuffle(all_labels)
    
    clusters_labels = avg_divide(all_labels, n_clusters)


    label2cluster = dict()  # maps label to its cluster
    for group_idx, labels in enumerate(clusters_labels):
        for label in labels:
            label2cluster[label] = group_idx


    #data_idcs = list(range(len(dataset)))

    data60000=[[] for i in range(60000)]

    clusters_sizes = np.zeros(n_clusters, dtype=int)

    clusters = {k: [] for k in range(n_clusters)}
    for batch_idx, (data, target) in enumerate(train_data_loader):

        target_n=int(target)
        group_id = label2cluster[target_n]

        clusters_sizes[group_id] += 1

        clusters[group_id].append(batch_idx)
        data60000[batch_idx].append((data, target))


    for _, cluster in clusters.items():
        random.shuffle(cluster)



    clients_counts = np.zeros((n_clusters, num_workers), dtype=np.int64) 


    for cluster_id in range(n_clusters):

        weights = np.random.dirichlet(alpha=alpha * np.ones(num_workers))

        clients_counts[cluster_id] = np.random.multinomial(clusters_sizes[cluster_id], weights)


    clients_counts = np.cumsum(clients_counts, axis=1)



    singel_data=()
    clients_idcs = [[] for i in range(num_workers)]
    for cluster_id in range(n_clusters):

        cluster_split = split_list_by_idcs(clusters[cluster_id], clients_counts[cluster_id])

        for client_id, idcs in enumerate(cluster_split):
            #clients_idcs[client_id] += idcs
            for i in idcs:
                clients_idcs[client_id].extend(data60000[i])
            
    return clients_idcs























def distribute_dataset(train_data_loader):
    distributed_dataset = []
    
    for batch_idx, (data, target) in enumerate(train_data_loader):
        data_idcs = list(range(len((data, target))))
        for idx in data_idcs:
            distributed_dataset.append((data[idx], target[idx]))
        #distributed_dataset.append((data, target))

    print(distributed_dataset[0][1])
    return distributed_dataset
