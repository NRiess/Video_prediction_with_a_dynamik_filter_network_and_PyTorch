__all__ = ["DynamicFilterLayer"]
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import subprocess as sp

writer = SummaryWriter()

def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    print(memory_free_values)


def DynamicFilterLayer(incomings, filter_size, pad=0, stride=1, flip_filters=False, grouping=False, **kwargs): # Layer

    image = incomings[0] # output
    filters = incomings[1] # filters



    filter_localexpand_np = np.reshape(np.eye(np.prod(filter_size), np.prod(filter_size)),
                                       (np.prod(filter_size), filter_size[2], filter_size[0],
                                        filter_size[1]))

    filter_localexpand = torch.tensor(filter_localexpand_np.astype((float)))
    image = image.type(torch.DoubleTensor).to('cuda')

    filter_localexpand = torch.tensor(filter_localexpand).to('cuda')
    input_localexpanded = F.conv2d(image, filter_localexpand, padding="same")


    output = input_localexpanded * filters



    output = torch.sum(output, dim=1, keepdim=True)
    return output











