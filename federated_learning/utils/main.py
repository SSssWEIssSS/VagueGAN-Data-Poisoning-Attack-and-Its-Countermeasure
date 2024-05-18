import argparse
import os
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms

import errno
import os
import shutil
import sys

from .build_gan import Model
from .data_loader_utils import generate_data_loaders_from_distributed_dataset

FLAGS = None

def main(data,k,idx):

    parser = argparse.ArgumentParser(description='VaguegAN')
    parser.add_argument('--model', type=str, default='usVGAN', help='one of `VGAN` and `usVGAN`.')
    parser.add_argument('--cuda', type=boolean_string, default=True, help='enable CUDA.')
    parser.add_argument('--train', type=boolean_string, default=True, help='train mode or eval mode.')
    parser.add_argument('--out_dir', type=str, default='', help='Directory for output.')
    parser.add_argument('--epochs', type=int, default=2, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=3000, help='size of batches')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=50, help='latent space dimension')
    parser.add_argument('--classes', type=int, default=10, help='number of classes')
    parser.add_argument('--img_size', type=int, default=28, help='size of images')
    parser.add_argument('--channels', type=int, default=1, help='number of image channels')
    parser.add_argument('--log_interval', type=int, default=1, help='interval between logging and image sampling')
    parser.add_argument('--seed', type=int, default=1, help='random seed')

    FLAGS = parser.parse_args()
    FLAGS.cuda = FLAGS.cuda and torch.cuda.is_available()
    FLAGS.out_dir='client_idx{}//output{}'.format(idx,k)

    if FLAGS.seed is not None:
        torch.manual_seed(FLAGS.seed)
        if FLAGS.cuda:
            torch.cuda.manual_seed(FLAGS.seed)
        np.random.seed(FLAGS.seed)

    cudnn.benchmark = True

    if FLAGS.train:
        clear_folder(FLAGS.out_dir)

    log_file = os.path.join(FLAGS.out_dir, 'VagueGAN_log.txt')
    print("VagueGAN-Logging to {}\n".format(log_file))
    #sys.stdout = utils.StdOut(log_file)
    sys.stdout = StdOut(log_file)

    print("VagueGAN-PyTorch version: {}".format(torch.__version__))
    print("VagueGAN-CUDA version: {}\n".format(torch.version.cuda))

    print(" " * 9 + "Args" + " " * 9 + "|    " + "Type" + \
          "    |    " + "Value")
    print("-" * 50)
    for arg in vars(FLAGS):
        arg_str = str(arg)
        var_str = str(getattr(FLAGS, arg))
        type_str = str(type(getattr(FLAGS, arg)).__name__)
        print("  " + arg_str + " " * (20-len(arg_str)) + "|" + \
              "  " + type_str + " " * (10-len(type_str)) + "|" + \
              "  " + var_str)

    device = torch.device("cuda:0" if FLAGS.cuda else "cpu")
    poision_dataset=[]
    if FLAGS.train:
        print('VagueGAN-Loading data...\n')

        gan_client=[]
        dataset=data

        assert dataset
        dataloader =generate_data_loaders_from_distributed_dataset(dataset, FLAGS.batch_size)
        dataloader=dataloader[0] 
        print('VagueGAN-Creating model...\n')
        model = Model(FLAGS.batch_size,FLAGS.model, device, dataloader, FLAGS.classes, FLAGS.channels, FLAGS.img_size, FLAGS.latent_dim)
        model.create_optim(FLAGS.lr)

        # Train
        poision_dataset=model.train(FLAGS.epochs, FLAGS.log_interval,FLAGS.out_dir, True)

        model.save_to('')
        return poision_dataset

#####   gan_utils
def to_np(var):
    """Exports torch.Tensor to Numpy array.
    """
    return var.detach().cpu().numpy()


def create_folder(folder_path):
    """Create a folder if it does not exist.
    """
    try:
        os.makedirs(folder_path)
    except OSError as _e:
        if _e.errno != errno.EEXIST:
            raise


def clear_folder(folder_path):
    """Clear all contents recursively if the folder exists.
    Create the folder if it has been accidently deleted.
    """
    create_folder(folder_path)
    for the_file in os.listdir(folder_path):
        _file_path = os.path.join(folder_path, the_file)
        try:
            if os.path.isfile(_file_path):
                os.unlink(_file_path)
            elif os.path.isdir(_file_path):
                shutil.rmtree(_file_path)
        except OSError as _e:
            print(_e)


class StdOut(object):
    """Redirect stdout to file, and print to console as well.
    """
    def __init__(self, output_file):
        self.terminal = sys.stdout
        self.log = open(output_file, "a")

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'