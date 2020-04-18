import cv2
import torch
import numpy
import csv
from numpy import genfromtxt

def load_image(filename, data_type=torch.float32):
    color_img = numpy.array(cv2.imread(filename, cv2.IMREAD_ANYCOLOR))
    h, w, c = color_img.shape
    color_data = color_img.astype(numpy.float32).transpose(2, 0, 1)
    return torch.from_numpy(
        color_data.reshape(1, c, h, w)        
    ).type(data_type) / 255.0

def load_depth(filename, data_type=torch.float32, scale=0.001):
    depth_img = numpy.array(cv2.imread(filename, cv2.IMREAD_ANYDEPTH))
    h, w = depth_img.shape
    depth_data = depth_img.astype(numpy.float32) * scale
    return torch.from_numpy(
        depth_data.reshape(1, 1, h, w)        
    ).type(data_type)

def load_depth_csv(filename, data_type=torch.float32, scale=1):
    # with open(filename, newline='') as csvfile:
    #     spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    #     for row in spamreader:
    #         print(', '.join(row))

    depth_img = genfromtxt(filename, delimiter=',')
    h, w = depth_img.shape
    depth_data = depth_img.astype(numpy.float32) * scale
    return torch.from_numpy(
        depth_data.reshape(1, 1, h, w)        
    ).type(data_type)

def readpgm(name):
    with open(name) as f:
        lines = f.readlines()

    # Ignores commented lines
    for l in list(lines):
        if l[0] == '#':
            lines.remove(l)

    # Makes sure it is ASCII format (P2)
    assert lines[0].strip() == 'P2' 

    # Converts data to a list of integers
    data = []
    for line in lines[1:]:
        data.extend([int(c) for c in line.split()])

    buffer = numpy.array(data[3:])
    # return (numpy.array(data[3:]), (data[1],data[0]),data[2])
    return buffer.reshape(data[1], data[0])

def load_depth_pgm(filename, data_type=torch.float32):
    depth_img = readpgm(filename)
    h, w = depth_img.shape

    # cv2.imshow("before", depth_img)
    # depth_img = cv2.warpAffine(depth_img, M, (h, w))
    # cv2.imshow("after", depth_img)
    # cv2.waitKey()

    depth_data = depth_img.astype(numpy.float32)
    return torch.from_numpy(
        depth_data.reshape(1, 1, h, w)        
    ).type(data_type)
