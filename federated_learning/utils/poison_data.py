from .label_replacement import apply_class_label_replacement
from .client_utils import log_client_data_statistics
from .main import main
from .tensor_converter_gan import convert_distributed_gandata_into_numpy
from .tensor_converter import convert_distributed_data_into_numpy
from torch.utils.data import TensorDataset

import time
import numpy 

####################################
#############VagueGAN###############
####################################


def poison_data(logger, distributed_dataset, num_workers, poisoned_worker_ids, replacement_method,idx):

    poisoned_dataset = []
    distributed_dataset_gan = [[] for i in range(60)]

    class_labels = list(set(distributed_dataset[0][1]))


    logger.info("Poisoning data for workers: {}".format(str(poisoned_worker_ids)))
    
    for worker_idx in range(num_workers):
        if worker_idx in poisoned_worker_ids:
            gan_dataset=[]
            
            gan_dataset.append(distributed_dataset[worker_idx])
            
            distributed_dataset_gan[worker_idx].append(main(gan_dataset,worker_idx,idx))
            
            X_ = numpy.array([tensor.cpu().numpy() for batch in distributed_dataset_gan[worker_idx] for tensor in batch[0]])
            Y_ = numpy.array([tensor.cpu().numpy() for batch in distributed_dataset_gan[worker_idx] for tensor in batch[1]])

            poisoned_dataset.append((X_, Y_))

            poisoned_dataset.append(apply_class_label_replacement(distributed_dataset[worker_idx][0], distributed_dataset[worker_idx][1], replacement_method))
        else:
            poisoned_dataset.append(distributed_dataset[worker_idx])

    log_client_data_statistics(logger, class_labels, poisoned_dataset)

    return poisoned_dataset
    
    


    
