import os
from loguru import logger
from federated_learning.arguments import Arguments
from federated_learning.dimensionality_reduction import calculate_pca_of_gradients
from federated_learning.parameters import get_layer_parameters
from federated_learning.parameters import calculate_parameter_gradients
from federated_learning.utils import get_model_files_for_epoch
from federated_learning.utils import get_model_files_for_suffix
from federated_learning.utils import apply_standard_scaler
from federated_learning.utils import get_worker_num_from_model_file_name
from client import Client
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import copy

# Paths you need to put in.
MODELS_PATH = "/absolute/path/to/models/folder/0_models"

# The epochs over which you are calculating gradients.
EPOCHS = list(range(1, 101))

# The layer of the NNs that you want to investigate.
#   If you are using the provided Fashion MNIST CNN, this should be "fc.weight"
#   If you are using the provided Cifar 10 CNN, this should be "fc2.weight"
LAYER_NAME = "fc.weight"

# The source class.
CLASS_NUM = 6

# The IDs for the poisoned workers. This needs to be manually filled out.
# You can find this information at the beginning of an experiment's log file.
POISONED_WORKER_IDS = [9, 16, 14, 18]
NUM_WORKERS= 20
#Important parameter that directly affects MCD accuracy
fbc=1#first baseline client id

# The resulting graph is saved to a file
#SAVE_NAME = "defense_results.jpg"
#SAVE_SIZE = (18, 14)


def load_models(args, model_filenames):
    clients = []
    for model_filename in model_filenames:
        client = Client(args, 0, None, None)
        client.set_net(client.load_model_from_file(model_filename))

        clients.append(client)

    return clients


def calculate_MCD_metrics(gradients):
    Omega1=4
    Omega2=2
    lambda1=1
    lambda2=4
    base_workers=0
    idx=np.zeros(NUM_WORKERS)#The number of times the client has been selected
    x1=np.zeros(NUM_WORKERS)#theta dimension 1
    x2=np.zeros(NUM_WORKERS)#theta dimension 2
    x1mean=np.zeros(NUM_WORKERS)
    x2mean=np.zeros(NUM_WORKERS)
    d=np.zeros(NUM_WORKERS)
    h1=np.zeros(NUM_WORKERS)
    h2=np.zeros(NUM_WORKERS)
    x1base=0
    x2base=0
    dbase=0
    oral = copy.deepcopy(gradients)
    for (worker_id, gradient) in gradients:
        x1[worker_id]+=gradient[0]
        print(gradient[0])
        x2[worker_id]+=gradient[1]
        idx[worker_id]+=1
        
        
    for n in range(NUM_WORKERS):
        x1[n]=x1[n]/idx[n]
        x2[n]=x2[n]/idx[n]
    print("Theta:")
    print(x1)
    print("\n")
    print(x2)
    for (worker, grad) in oral:
        d[worker]+= pow ( pow(( np.float64(x1[worker]) - np.float64(grad[0]) ),2) + pow(( np.float64(x2[worker]) - np.float64(grad[1]) ),2) ,0.5)
    
    for n in range(NUM_WORKERS):
        d[n]=d[n]/idx[n]
    print("D:")
    print(d)
    print("I:")
    print(idx)
    
    for n in range(NUM_WORKERS):
        if ((pow( pow(( np.float64(x1[fbc]) - np.float64(x1[n]) ),2) + pow(( np.float64(x2[fbc]) - np.float64(x2[n]) ),2) ,0.5))/d[fbc] < Omega1) and (abs(d[n]-d[fbc])/d[fbc] <Omega2):
            x1base+=x1[n]
            x2base+=x2[n]
            dbase+=d[n]
            base_workers+=1
            
    x1base = x1base/base_workers
    x2base = x2base/base_workers
    dbase= dbase/base_workers
    print("Base Workers Number")
    print(base_workers)
    print("Theta_base:")
    print(x1base)
    print(",")
    print(x2base)
    print("\n")
    print("D_base:")
    print(dbase)
    print("\n")
    
    for n in range(NUM_WORKERS):
        h1[n]=lambda1*((pow( pow(( np.float64(x1base) - np.float64(x1[n]) ),2) + pow(( np.float64(x2base) - np.float64(x2[n]) ),2) ,0.5))/dbase)
        h2[n]=lambda2*(abs(d[n]-dbase))
    print("Client Abnormality:")    
    print(h1)
    print("\n")
    print(h2)

if __name__ == '__main__':
    args = Arguments(logger)
    args.log()

    model_files = sorted(os.listdir(MODELS_PATH))
    logger.debug("Number of models: {}", str(len(model_files)))

    param_diff = []
    worker_ids = []

    for epoch in EPOCHS:
        start_model_files = get_model_files_for_epoch(model_files, epoch)
        start_model_file = get_model_files_for_suffix(start_model_files, args.get_epoch_save_start_suffix())[0]
        start_model_file = os.path.join(MODELS_PATH, start_model_file)
        start_model = load_models(args, [start_model_file])[0]

        start_model_layer_param = list(get_layer_parameters(start_model.get_nn_parameters(), LAYER_NAME)[CLASS_NUM])

        end_model_files = get_model_files_for_epoch(model_files, epoch)
        end_model_files = get_model_files_for_suffix(end_model_files, args.get_epoch_save_end_suffix())

        for end_model_file in end_model_files:
            worker_id = get_worker_num_from_model_file_name(end_model_file)
            end_model_file = os.path.join(MODELS_PATH, end_model_file)
            end_model = load_models(args, [end_model_file])[0]

            end_model_layer_param = list(get_layer_parameters(end_model.get_nn_parameters(), LAYER_NAME)[CLASS_NUM])

            gradient = calculate_parameter_gradients(logger, start_model_layer_param, end_model_layer_param)
            gradient = gradient.flatten()

            param_diff.append(gradient)
            worker_ids.append(worker_id)

    logger.info("Gradients shape: ({}, {})".format(len(param_diff), param_diff[0].shape[0]))

    logger.info("Prescaled gradients: {}".format(str(param_diff)))
    scaled_param_diff = apply_standard_scaler(param_diff)
    logger.info("Postscaled gradients: {}".format(str(scaled_param_diff)))
    dim_reduced_gradients = calculate_pca_of_gradients(logger, scaled_param_diff, 2)
    logger.info("PCA reduced gradients: {}".format(str(dim_reduced_gradients)))

    logger.info("Dimensionally-reduced gradients shape: ({}, {})".format(len(dim_reduced_gradients), dim_reduced_gradients[0].shape[0]))

    calculate_MCD_metrics(zip(worker_ids, dim_reduced_gradients))
