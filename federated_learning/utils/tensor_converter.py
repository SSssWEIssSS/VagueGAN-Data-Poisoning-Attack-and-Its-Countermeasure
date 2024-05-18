import numpy

def convert_distributed_data_into_numpy(distributed_dataset):
    """
    Converts a distributed dataset (returned by a data distribution method) from Tensors into numpy arrays.

    :param distributed_dataset: Distributed dataset
    :type distributed_dataset: list(tuple)
    """
    converted_distributed_dataset = []

    for worker_idx in range(len(distributed_dataset)):
        worker_training_data = distributed_dataset[worker_idx]

        X_ = numpy.array([tensor.numpy() for batch in worker_training_data for tensor in batch[0]])#,dtype=float
        Y_ = numpy.array([tensor.numpy() for batch in worker_training_data for tensor in batch[1]])#.astype(numpy.float)

        converted_distributed_dataset.append((X_, Y_))

    return converted_distributed_dataset

def convert_numpy(distributed_dataset):
    converted_distributed_dataset = []

    X_ = numpy.array([tensor.numpy() for batch in distributed_dataset for tensor in batch[0]])#,dtype=float
    Y_ = numpy.array([tensor.numpy() for batch in distributed_dataset for tensor in batch[1]])#.astype(numpy.float)

    converted_distributed_dataset.append((X_, Y_))

    return converted_distributed_dataset