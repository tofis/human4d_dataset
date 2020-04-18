import numpy
import torch

def load_timestamp(filename, data_type=torch.long):
    data = numpy.loadtxt(filename)
    timestamp = torch.tensor(data, dtype=data_type)
    return timestamp