# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import argparse
import cv2
import numpy
import skimage.draw

from scipy import ndimage
from scipy.spatial.transform import Rotation as R
import math

import re
import time
import os

from importers import *
from vision import *
from exporters import *
from optimization import *
from visualization import *
from structs import *

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

import networkx as nx
from networkx.algorithms import bipartite
from networkx.algorithms import flow

from sparseba import SBA
from spacepy import pycdf

from sklearn.cluster import AgglomerativeClustering

import json
import open3d

ply_colors = [ 'red', 'blue', 'orange', 'green', 'brown' ]
sequence_paths = [
    ### SUBJECT 1 ###
    # "E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/_ird_recordings/S1/19-07-12-08-12-28",
    "E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/_ird_recordings/S1/19-07-12-08-13-23",
    "E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/_ird_recordings/S1/19-07-12-08-14-17",
    "E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/_ird_recordings/S1/19-07-12-08-15-13",
    # "E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/_ird_recordings/S1/19-07-12-08-17-29",
    "E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/_ird_recordings/S1/19-07-12-08-20-39",
    # "E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/_ird_recordings/S1/19-07-12-08-21-48",
    ### SUBJECT 2 ###
    "E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/_ird_recordings/S2/19-07-12-09-16-58",
    "E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/_ird_recordings/S2/19-07-12-09-17-52",
    "E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/_ird_recordings/S2/19-07-12-09-18-54",
    "E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/_ird_recordings/S2/19-07-12-09-26-19",
    ### SUBJECT 3 ###
    "E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/_ird_recordings/S3/19-07-12-10-12-49",
    # "E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/_ird_recordings/S3/19-07-12-10-14-07",
    # "E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/_ird_recordings/S3/19-07-12-10-15-30",
    # "E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/_ird_recordings/S3/19-07-12-10-16-22",
    "E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/_ird_recordings/S3/19-07-12-10-17-54",
    # "E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/_ird_recordings/S3/19-07-12-10-18-57",
    "E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/_ird_recordings/S3/19-07-12-10-20-28",
    # "E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/_ird_recordings/S3/19-07-12-10-21-36",
    "E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/_ird_recordings/S3/19-07-12-10-22-37",
    "E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/_ird_recordings/S3/19-07-12-10-23-34",
    "E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/_ird_recordings/S3/19-07-12-10-25-04",
    "E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/_ird_recordings/S3/19-07-12-10-26-42",
    "E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/_ird_recordings/S3/19-07-12-10-27-51",
    "E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/_ird_recordings/S3/19-07-12-10-29-28",
    ### SUBJECT 4 ###
    "E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/_ird_recordings/S4/19-07-12-12-10-24",
    # "E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/_ird_recordings/S4/19-07-12-12-11-24",
    # "E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/_ird_recordings/S4/19-07-12-12-12-33",
    # "E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/_ird_recordings/S4/19-07-12-12-13-39",
    "E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/_ird_recordings/S4/19-07-12-12-14-40",
    # "E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/_ird_recordings/S4/19-07-12-12-15-39",
    "E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/_ird_recordings/S4/19-07-12-12-17-02",
    # "E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/_ird_recordings/S4/19-07-12-12-18-12",
    "E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/_ird_recordings/S4/19-07-12-12-19-22",
    "E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/_ird_recordings/S4/19-07-12-12-20-30",
    "E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/_ird_recordings/S4/19-07-12-12-21-22",
    "E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/_ird_recordings/S4/19-07-12-12-22-34",
    "E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/_ird_recordings/S4/19-07-12-12-23-52",
    "E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/_ird_recordings/S4/19-07-12-12-26-59",
]

sequence_files = [
    ### SUBJECT 1 ###
    # "INF_Running_S1_01",             
    "INF_JumpingJack_S1_01",         
    "INF_Bending_S1_01",             
    "INF_PunchingKicking_S1_01",     
    # "INF_LayingDown_S1_01",          
    "INF_SittingStanding_S1_01",     
    # "INF_Talking_S1_01",             
    ### SUBJECT 2 ###
    "INF_JumpingJack_S2_01",         
    "INF_Bending_S2_01",             
    "INF_PunchingKicking_S2_01",     
    "INF_SittingStanding_S2_01",
    ### SUBJECT 3 ###
    "INF_Running_S3_01_eval",
    # "INF_JumpingJack_S3_01",          
    # "INF_Bending_S3_01",             
    # "INF_PunchingKicking_S3_01",     
    "INF_Basketball_S3_01",          
    # "INF_LayingDown_S3_01",          
    "INF_SittingFloor_S3_01",        
    # "INF_SittingStanding_S3_01",     
    "INF_Talking_S3_01",             
    "INF_PickingDroppingObj_S3_01",  
    "INF_StretchingTalking_S3_01",   
    "INF_TalkingWalking_S3_01",      
    "INF_WatchingScaryMovie_S3_01",  
    "INF_InflightSafety_S3_01",      
    ### SUBJECT 4 ###
    "INF_Running_S4_01",             
    # "INF_JumpingJack_S4_01",         
    # "INF_Bending_S4_01",             
    # "INF_PunchingKicking_S4_01",     
    "INF_Basketball_S4_01",          
    # "INF_LayingDown_S4_01",          
    "INF_SittingFloor_S4_01",        
    # "INF_SittingStanding_S4_01",     
    "INF_Talking_S4_01",             
    "INF_PickingDroppingObj_S4_01",  
    "INF_StretchingTalking_S4_01",   
    "INF_TalkingWalking_S4_01",      
    "INF_WatchingScaryMovie_S4_01",  
    "INF_InflightSafety_S4_02",      
]

def main():
    parser = argparse.ArgumentParser(description="PyTorch Show Pose and Pointcloud")
    parser.add_argument(
        "--show_markers",
        default=False,      
        help="render marker positions",
    )
    parser.add_argument(
        "--show_joints",
        default=False,      
        help="render joint positions",
    )
    parser.add_argument(
        "--show_rect",
        default=False,      
        help="render joint positions",
    )
    parser.add_argument(
        "--save_img",
        default=False,      
        help="render joint positions",
    )
    parser.add_argument(
        "--show_img",
        default=False,      
        help="render joint positions",
    )
    parser.add_argument(
        "--do_sba",
        default=False,      
        help="Conduct Sparse Bundle Adjustment and save new extrinsics",
    )
    parser.add_argument(
        "--vis_3d",
        default=False,      
        help="Visualize 3d points",
    )
    parser.add_argument(
        "--test_mode",
        default=False
    )
    # parser.add_argument(
    #     "--sequence_path",
    #     ### SUBJECT 1 ###
    #     # default="E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/_ird_recordings/S1/19-07-12-08-12-28",    +
    #     # default="E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/_ird_recordings/S1/19-07-12-08-13-23",    +
    #     # default="E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/_ird_recordings/S1/19-07-12-08-14-17",    +
    #     # default="E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/_ird_recordings/S1/19-07-12-08-15-13",    +
    #     # default="E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/_ird_recordings/S1/19-07-12-08-16-29",    -
    #     # default="E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/_ird_recordings/S1/19-07-12-08-17-29",    +
    #     # default="E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/_ird_recordings/S1/19-07-12-08-19-14",    -
    #     # default="E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/_ird_recordings/S1/19-07-12-08-20-39",    +
    #     # default="E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/_ird_recordings/S1/19-07-12-08-21-48",    +
    #     ### SUBJECT 2 ###
    #     # default="E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/_ird_recordings/S2/19-07-12-09-16-58",    +
    #     # default="E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/_ird_recordings/S2/19-07-12-09-17-52",    +
    #     # default="E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/_ird_recordings/S2/19-07-12-09-18-54",    +
    #     default="E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/_ird_recordings/S2/19-07-12-09-26-19",
    #     ### SUBJECT 3 ###
    #     # default="E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/_ird_recordings/S3/19-07-12-10-12-49",    +
    #     # default="E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/_ird_recordings/S3/19-07-12-10-14-07",    +
    #     # default="E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/_ird_recordings/S3/19-07-12-10-15-30",    +
    #     # default="E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/_ird_recordings/S3/19-07-12-10-16-22",    +
    #     # default="E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/_ird_recordings/S3/19-07-12-10-17-54",    +
    #     # default="E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/_ird_recordings/S3/19-07-12-10-18-57",    +
    #     # default="E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/_ird_recordings/S3/19-07-12-10-20-28",    +
    #     # default="E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/_ird_recordings/S3/19-07-12-10-21-36",    +
    #     # default="E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/_ird_recordings/S3/19-07-12-10-22-37",    +
    #     # default="E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/_ird_recordings/S3/19-07-12-10-23-34",    +
    #     # default="E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/_ird_recordings/S3/19-07-12-10-25-04",    +
    #     # default="E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/_ird_recordings/S3/19-07-12-10-26-42",    +
    #     # default="E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/_ird_recordings/S3/19-07-12-10-27-51",    +
    #     # default="E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/_ird_recordings/S3/19-07-12-10-29-28",    +
    #     ### SUBJECT 4 ###
    #     # default="E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/_ird_recordings/S4/19-07-12-12-10-24",    +
    #     # default="E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/_ird_recordings/S4/19-07-12-12-11-24",    +
    #     # default="E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/_ird_recordings/S4/19-07-12-12-12-33",    +
    #     # default="E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/_ird_recordings/S4/19-07-12-12-13-39",    +
    #     # default="E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/_ird_recordings/S4/19-07-12-12-14-40",    +
    #     # default="E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/_ird_recordings/S4/19-07-12-12-15-39",    +
    #     # default="E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/_ird_recordings/S4/19-07-12-12-17-02",    +
    #     # default="E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/_ird_recordings/S4/19-07-12-12-18-12",    +
    #     # default="E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/_ird_recordings/S4/19-07-12-12-19-22",    +
    #     # default="E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/_ird_recordings/S4/19-07-12-12-20-30",    +
    #     # default="E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/_ird_recordings/S4/19-07-12-12-21-22",    +
    #     # default="E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/_ird_recordings/S4/19-07-12-12-22-34",    +
    #     # default="E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/_ird_recordings/S4/19-07-12-12-23-52",    +
    #     # default="E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/_ird_recordings/S4/19-07-12-12-26-59",    +
    #     help="path to sequence files",
    # )
    # parser.add_argument(
    #     "--sequence_filename", 
    #     ### SUBJECT 1 ###
    #     # default="INF_Running_S1_01",              +
    #     # default="INF_JumpingJack_S1_01",          +
    #     # default="INF_Bending_S1_01",              +
    #     # default="INF_PunchingKicking_S1_01",      +
    #     # default="INF_Basketball_S1_01",           +
    #     # default="INF_LayingDown_S1_01",           +
    #     # default="INF_SittingFloor_S1_01",         -
    #     # default="INF_SittingStanding_S1_01",      +
    #     # default="INF_Talking_S1_01",              +
    #     ### SUBJECT 2 ###
    #     # default="INF_JumpingJack_S2_01",          +
    #     # default="INF_Bending_S2_01",              +
    #     # default="INF_PunchingKicking_S2_01",      +
    #     default="INF_SittingStanding_S2_01",
    #     ### SUBJECT 3 ###
    #     # default="INF_Running_S3_01_eval",         +
    #     # default="INF_JumpingJack_S3_01,           +
    #     # default="INF_Bending_S3_01",              +
    #     # default="INF_PunchingKicking_S3_01",      +
    #     # default="INF_Basketball_S3_01",           +
    #     # default="INF_LayingDown_S3_01",           +
    #     # default="INF_SittingFloor_S3_01",         +
    #     # default="INF_SittingStanding_S3_01",      +
    #     # default="INF_Talking_S3_01",              +
    #     # default="INF_PickingDroppingObj_S3_01",   +
    #     # default="INF_StretchingTalking_S3_01",    +
    #     # default="INF_TalkingWalking_S3_01",       +
    #     # default="INF_WatchingScaryMovie_S3_01",   +
    #     # default="INF_InflightSafety_S3_01",       +
    #     ### SUBJECT 4 ###
    #     # default="INF_Running_S4_01",              +
    #     # default="INF_JumpingJack_S4_01",          +
    #     # default="INF_Bending_S4_01",              +
    #     # default="INF_PunchingKicking_S4_01",      +
    #     # default="INF_Basketball_S4_01",           +
    #     # default="INF_LayingDown_S4_01",           +
    #     # default="INF_SittingFloor_S4_01",         +
    #     # default="INF_SittingStanding_S4_01",      +
    #     # default="INF_Talking_S4_01",              +
    #     # default="INF_PickingDroppingObj_S4_01",   +
    #     # default="INF_StretchingTalking_S4_01",    +
    #     # default="INF_TalkingWalking_S4_01",       +
    #     # default="INF_WatchingScaryMovie_S4_01",   +
    #     # default="INF_InflightSafety_S4_02",       +
    #     help="path to sequence files",
    # )
    parser.add_argument(
        "--resolution",     
        nargs="*", type=int, 
        default = [320, 180],
        help="input resolution",
    )
    parser.add_argument(
        "--frames2save",     
        nargs="*", type=int, 
        default = [20, 200, 280, 324, 400],
        help="frame ids to be saved in .png",
    )

    args = parser.parse_args()

    rs_step = 1
    rs_threshold = 100
    rs_floor = 820 # 2 x 410 mm = 820 mm is the theoretical distance from the floor based on the IKEA box structure

    COLORS = get_COLORS()

    seq_id = 0
    for sequence_path in sequence_paths:
        sequence_filename = sequence_files[seq_id]
        print(sequence_filename)

        device_repo_path = os.path.join(sequence_path,"../../../device_repository.json")
        if not os.path.exists(device_repo_path):
            raise ValueError("{0} does not exist".format(device_repo_path))            
        device_repo = load_intrinsics_repository(os.path.join(device_repo_path))
        device_repo_rgb = load_intrinsics_repository(os.path.join(device_repo_path), stream='RGB')
        device_repo_RT = load_rotation_translation(os.path.join(device_repo_path))
    
        extr_files = [current_ for current_ in os.listdir(os.path.join(sequence_path, "../pose")) if ".extrinsics" in current_]

        extrinsics = {}
        views = []

        json_file_txt = open(os.path.join(sequence_path, "../pose/extrinsics.json"), 'r').read()
        all_extr = json.loads(json_file_txt)

        for view_id in range(len(all_extr['Viewpoints'])):
            extrinsics[all_extr['Viewpoints'][view_id]['name']] = torch.from_numpy(numpy.asarray(all_extr['Viewpoints'][view_id]['extrinsics'])).reshape([4,4]).type(torch.FloatTensor)
            extrinsics[all_extr['Viewpoints'][view_id]['name']][0:3, 3] *= 1000
            # extrinsics[extr.split(".")[0]] = load_extrinsics(os.path.join(sequence_path, "../pose", extr))[0]
            views.append(all_extr['Viewpoints'][view_id]['name'])

        gt_joints = load_joints_seq(os.path.join(sequence_path, sequence_filename + ".joints"))
        gt_markers = load_markers_seq(os.path.join(sequence_path, sequence_filename + ".markers"))

        # gt_joints = numpy.expand_dims(gt_joints[:, 0, :, :], axis=1)
        # gt_markers = numpy.expand_dims(gt_markers[:, 0, :, :], axis=1)

        time_step = 8.33333
        
        transform_vicon = numpy.load(os.path.join(sequence_path, "../pose/", "extr.npy"))

        
    
        rotation_gt = torch.from_numpy(transform_vicon[0:3, 0:3]).type(torch.float)
        translation_gt = torch.tensor(transform_vicon[0:3, 3]).reshape(3, 1).type(torch.float)

        
        gt_markers_t = torch.from_numpy(gt_markers).reshape(gt_markers.shape[0], gt_markers.shape[1], gt_markers.shape[2], gt_markers.shape[3]).permute(0, 3, 1, 2).type(torch.float)
        gt_joints_t = torch.from_numpy(gt_joints).reshape(gt_joints.shape[0], gt_joints.shape[1], gt_joints.shape[2], gt_joints.shape[3]).permute(0, 3, 1, 2).type(torch.float)
        # transform vicon
        gt_markers_t = transform_points(gt_markers_t, rotation_gt, translation_gt)
        gt_joints_t = transform_points(gt_joints_t, rotation_gt, translation_gt)
      



        rgbd_skip = load_rgbd_skip(os.path.join(sequence_path, "../offsets.txt"), os.path.basename(sequence_path))
        h4d_seq = H4DSequence(os.path.join(sequence_path, "Dump"), ["M72e", "M72h", "M72i", "M72j"], skip=rgbd_skip, test_mode=args.test_mode)
    
        all_sequence_2d = {}
        all_sequence_3d = {}
        all_sequence_bbox = {}
        global_sequence_3d = numpy.zeros([h4d_seq.num_of_frames, gt_joints_t.shape[3], gt_joints_t.shape[2], gt_joints_t.shape[1]])

        for view in h4d_seq.camera_ids:  
            all_sequence_2d[view] = numpy.zeros([h4d_seq.num_of_frames, gt_joints_t.shape[2], gt_joints_t.shape[3], gt_joints_t.shape[1]-1])
            all_sequence_3d[view] = numpy.zeros([h4d_seq.num_of_frames, gt_joints_t.shape[2], gt_joints_t.shape[3], gt_joints_t.shape[1]])
            all_sequence_bbox[view] = numpy.zeros([h4d_seq.num_of_frames, gt_joints_t.shape[2], 4], dtype=numpy.int16)

        for i in range(h4d_seq.num_of_frames-1): ## -1 to remove T-Pose
            view_id = 0
            markers_2d_obs = {}
            markers2d = {}
            markers2d_of_camera = {}
            markers2d_of_vicon = {}
            markers_con2d = {}
            markers3d = {}
            rotation = {}
            translation = {}
            rotation_inv = {}
            translation_inv = {}
            intr = {}
            intr_inv = {}
            intr_rgb = {}
            intr_rgb_inv = {}
            points_3d_all = {}
            pp = []
            fs = []
            marker_3d_points = []


            gt_index = int(round(h4d_seq.cameras[h4d_seq.camera_ids[0]][i].timestamp / time_step, 0))

            if gt_index >= gt_joints.shape[0]:
                break
            
            data_out = os.path.join(sequence_path, sequence_filename + "_" + os.path.basename(sequence_path) + "_data_final_raw_calib_new")
            if (not os.path.exists(data_out)):
                os.makedirs(data_out)

            if (h4d_seq.cameras[view][i].groupframe_id > rs_threshold and h4d_seq.cameras[view][i].groupframe_id % rs_step == 0):
                marker_pos_file = open(os.path.join(data_out, "txt_" + str(h4d_seq.cameras[view][i].groupframe_id) + "_3d_rs.txt"), "w")        
                # save_ply(str(h4d_seq.cameras[view][i].groupframe_id) + "_gt.ply", gt_markers_t[gt_index].unsqueeze(0), 1.0)
                save_gt_sample(torch.cat([gt_markers_t[gt_index], gt_joints_t[gt_index]], dim=2), os.path.join(data_out, "txt_" + str(h4d_seq.cameras[view][i].groupframe_id) + "_3d_gt.txt"))

            for view in h4d_seq.camera_ids:      
                print(view + " " + str(h4d_seq.cameras[view][i].groupframe_id))
                

                depth_t = torch.from_numpy(\
                    h4d_seq.cameras[view][i].depth_img.reshape(1, 1, h4d_seq.cameras[view][i].depth_img.shape[0], h4d_seq.cameras[view][i].depth_img.shape[1])).float()

                uv_grid = create_image_domain_grid(args.resolution[0], args.resolution[1])

                intr[view], intr_inv[view] = get_intrinsics(view, device_repo, 4)
                intr_rgb[view], intr_rgb_inv[view] = get_intrinsics(view, device_repo_rgb, 1)
                points_3d_all[view] = deproject_depth_to_points(depth_t, uv_grid, intr_inv[view], floor_y=rs_floor)
                rotation[view], translation[view] = extract_rotation_translation(extrinsics[view].unsqueeze(0))
                rotation_inv[view] = torch.inverse(rotation[view])
                translation_inv[view] = - rotation_inv[view] @ translation[view]
                points_3d_t = transform_points(points_3d_all[view], rotation[view], translation[view])

                # filename = "transformed_point_cloud_%s_%d.ply" % (view, i)
                # save_ply(filename, points_3d_t, 1, color=ply_colors[view_id])
                points_3d = points_3d_all[view].permute(2, 3, 1, 0).squeeze()
                points_3d_t = points_3d_t.permute(2, 3, 1, 0).squeeze()

                ################## projection ####################
                img_c = h4d_seq.cameras[view][i].color_img.copy()
                #img_c = cv2.resize(h4d_seq.cameras[view][i].depth_img.copy(), (1280, 720), interpolation = cv2.INTER_AREA).astype(numpy.uint16)
                #img_c = cv2.cvtColor(img_c, cv2.COLOR_GRAY2BGR)

                gt_markers_view_aligned = transform_points(gt_markers_t[gt_index].unsqueeze(0), rotation_inv[view], translation_inv[view])
                gt_joints_view_aligned = transform_points(gt_joints_t[gt_index].unsqueeze(0), rotation_inv[view], translation_inv[view])
                if ("INF" not in sequence_filename):  
                    R_rgb = torch.from_numpy(device_repo_RT[view]['R']).inverse()
                    t_rgb = - R_rgb @ (1000 * torch.from_numpy(device_repo_RT[view]['t']))
                    gt_markers_view_aligned = transform_points(gt_markers_view_aligned, \
                        R_rgb, t_rgb)
                    gt_joints_view_aligned = transform_points(gt_joints_view_aligned, \
                        R_rgb, t_rgb)
                
                # projected = project_points_to_uvs(gt_markers_view_aligned / 1000.0, intr)           
                marker_visibility = numpy.zeros([gt_markers_t.shape[2], gt_markers_t.shape[3]], dtype=int)
                marker_keypoints = numpy.zeros([gt_markers_t.shape[2], gt_markers_t.shape[3], 2], dtype=int)           

                for p in range(gt_markers_t.shape[2]):
                    for j in range(gt_markers_t.shape[3]):
                        if ("INF" in sequence_filename):
                            uv = project_single_point_to_uv(gt_markers_view_aligned[0, :, p, j], 4 * intr[view])
                            marker_keypoints[p, j] = uv
                        else:
                            uv = project_single_point_to_uv(gt_markers_view_aligned[0, :, p, j], intr_rgb[view])
                            marker_keypoints[p, j] = uv

                        if (args.show_markers): 
                            if ("INF" in sequence_filename):      
                                if (uv[0]/4 < args.resolution[0] and uv[1]/4 < args.resolution[1]):      
                                    depth_diff = numpy.abs(int(depth_t[0, 0, int(uv[1]/4), int(uv[0]/4)]) - gt_markers_view_aligned[0, 2, p, j])                            
                                    if (depth_diff < 50):
                                        marker_visibility[p, j] = 1
                                        img_c = cv2.drawMarker(img_c, 
                                                            (int(uv[0]), int(uv[1])), 
                                                            COLORS[format(j+1, '02d')],
                                                            markerType=cv2.MARKER_CROSS,
                                                            markerSize=15,
                                                            thickness=2)
                                    else:
                                        img_c = cv2.drawMarker(img_c, 
                                                            (int(uv[0]), int(uv[1])), 
                                                            COLORS[format(j+1, '02d')],
                                                            markerType=cv2.MARKER_DIAMOND,
                                                            markerSize=5,
                                                            thickness=1)
                            else:
                                img_c = cv2.drawMarker(img_c, 
                                                    (int(uv[0]), int(uv[1])), 
                                                    COLORS[format(j+1, '02d')],
                                                    markerType=cv2.MARKER_CROSS,
                                                    markerSize=15,
                                                    thickness=2)
            

                keypoints = numpy.zeros([gt_joints_t.shape[2], gt_joints_t.shape[3], 2], dtype=int)
                keypoints3d = numpy.zeros([gt_joints_t.shape[2], gt_joints_t.shape[3], 3], dtype=float)
                for p in range(gt_joints_t.shape[2]):
                    for j in range(gt_joints_t.shape[3]):                
                        keypoints3d[p, j] = gt_joints_view_aligned[0, :, p, j].cpu().numpy()
                        if ("INF" in sequence_filename):
                            uv = project_single_point_to_uv(gt_joints_view_aligned[0, :, p, j], 4 * intr[view])
                            keypoints[p, j] = uv
                        else:
                            uv = project_single_point_to_uv(gt_joints_view_aligned[0, :, p, j], intr_rgb[view])
                            keypoints[p, j] = uv

                        # print("uv: " + str(uv) + " p: " + str(p) + " j: " + str(j))
                        if (args.show_joints):
                            img_c = cv2.drawMarker(img_c, 
                                                (int(uv[0]), int(uv[1])), 
                                                COLORS[format(j+1, '02d')],
                                                markerType=cv2.MARKER_STAR,
                                                markerSize=10,
                                                thickness=2)

                all_sequence_2d[view][i] = keypoints
                all_sequence_3d[view][i] = keypoints3d

                # rectangle def and drawing
                p_offset = 20
                for p in range(gt_joints_t.shape[2]):
                    min_x = int(numpy.min(marker_keypoints[p, :, 0]) - p_offset)
                    min_y = int(numpy.min(marker_keypoints[p, :, 1]) - p_offset)
                    max_x = int(numpy.max(marker_keypoints[p, :, 0]) + p_offset)
                    max_y = int(numpy.max(marker_keypoints[p, :, 1]) + p_offset)
                    
                    width = args.resolution[0] * 4 
                    height = args.resolution[1] * 4 
                    all_sequence_bbox[view][i, p, 0] = min_x if min_x >= 0 else 0 
                    all_sequence_bbox[view][i, p, 1] = min_y if min_y >= 0 else 0 
                    all_sequence_bbox[view][i, p, 2] = max_x if max_x <= width - 1 else width - 1
                    all_sequence_bbox[view][i, p, 3] = max_y if max_y <= height - 1 else height - 1 

                    if (args.show_rect):
                        cv2.rectangle(img_c, (all_sequence_bbox[view][i, p, 0], all_sequence_bbox[view][i, p, 1]), \
                            (all_sequence_bbox[view][i, p, 2], all_sequence_bbox[view][i, p, 3]), (250, 100, 100), thickness=2)

                    if (args.show_joints):
                        draw_skeleton_joints(img_c, keypoints[p], COLORS)                

                    if (args.save_img and h4d_seq.cameras[view][i].groupframe_id > rs_threshold and h4d_seq.cameras[view][i].groupframe_id % rs_step == 0):
                        savefolder = os.path.join(sequence_path, "imgs")
                        if not os.path.exists(savefolder):
                            os.makedirs(savefolder)
                        cv2.imwrite(os.path.join(savefolder, "{}_{}.png".format(str(i), view)), \
                                    img_c)
                ########## marker matching ###################
                if (True):
                    gray_img = cv2.cvtColor(h4d_seq.cameras[view][i].color_img, cv2.COLOR_BGR2GRAY)
                
                    _, mask_img = cv2.threshold(gray_img, 170, 255, cv2.THRESH_BINARY)
                    
                    # cv2.imshow("gray", mask_img)
                    # cv2.waitKey(0)

                    contours = cv2.findContours(mask_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                    contours_area = []
                    markers2d[view] = []
                    markers2d_of_camera[view] = []
                    markers2d_of_vicon[view] = []
                    markers_2d_obs[view] = []
                    markers_con2d[view] = []
                    markers3d[view] = []
                    # calculate area and filter into new array
                    for con in contours[0]:
                        area = cv2.contourArea(con)
                        if 3 < area < 100:
                            contours_area.append(con)

                    # check if contour is of circular shape
                    for con in contours_area:
                        perimeter = cv2.arcLength(con, True)
                        area = cv2.contourArea(con)
                        if perimeter == 0:
                            break
                        circularity = 4*math.pi*(area/(perimeter*perimeter))
                        # print (circularity)
                        if 0.5 < circularity < 1.5:                        
                            markers_con2d[view].append(con)               

                    
                    h, w = h4d_seq.cameras[view][i].depth_img.shape
                                
                    for marker in markers_con2d[view]:
                        # compute the center of the contour
                        M = cv2.moments(marker)
                        cX = int(M["m10"] / M["m00"] / 4)
                        cY = int(M["m01"] / M["m00"] / 4)
                    
                        # offset = 2
                        # points4depth = []
                        # for x in range(-offset, offset):
                        #     for y in range(-offset, offset):
                        #         if (cY+y > -1 and cY+y < h and \
                        #             cX+x > -1 and cX+x < w):
                        #             points4depth.append([cY+y, cX+x, points_3d[cY+y, cX+x][2].cpu().numpy()])

                        # points4depth.sort()

                        # if len(points4depth):
                        #     sorted(points4depth, key=lambda x: x[2])
                        #     selected_point = points4depth[len(points4depth)//2]
                        
                        selected_point = [cY, cX, points_3d[cY, cX][2].cpu().numpy()]
                        if selected_point[2] > 50 and selected_point[2] < 3500:
                            if (selected_point[0] > -1 and selected_point[0] < h and \
                                selected_point[1] > -1 and selected_point[1] < w): # and \
                                    # points_3d[cY, cX][1].cpu().numpy() > rs_floor):
                                    markers2d[view].append((4 * selected_point[1], 4 * selected_point[0]))  
                                    marker_point_3d = points_3d_t[selected_point[0], selected_point[1]].cpu().numpy()
                                    markers3d[view].append(marker_point_3d)
                                    marker_3d_points.append(marker_point_3d)
                            
                    gt_markers_t_temp = gt_markers_t.clone().squeeze()[gt_index].cpu().numpy()
                    
                if (args.show_img):
                    cv2.imshow("color_" + view, cv2.transpose(img_c))
                    view_id += 1
            
            if (args.show_img):
                cv2.waitKey(100)
            marker_3d_points_np = numpy.asarray(marker_3d_points)

            doClustering = True # TODO: put it to args
            if (doClustering):
                t1_c = time.clock()
                clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=20).fit(marker_3d_points_np)
                t2_c = time.clock()
                print("t_clustering: " + str(t2_c - t1_c))

                clusters = []
                for cluster_id in range(0, clustering.n_clusters_):
                    clusters.append([])
                    for point_id in range(0, len(marker_3d_points)):
                        if clustering.labels_[point_id] == cluster_id:
                            clusters[cluster_id].append(marker_3d_points[point_id])
                            
                # keep the cluster with more than 1 vertex
                remove_single_markers = False
                if (remove_single_markers):
                    clusters = [cluster for cluster in clusters if len(cluster) > 1]
                
                centers = numpy.zeros([len(clusters), 3])
                projections_ = numpy.zeros([2, len(clusters)])

                for cluster_id in range(0, len(centers)):
                    for j in range(0, len(clusters[cluster_id])):
                        centers[cluster_id, :] += clusters[cluster_id][j]
                    centers[cluster_id, :] /= len(clusters[cluster_id])
                    # marker_3d_final.append(centers[cluster_id, :])
                    if (h4d_seq.cameras[view][i].groupframe_id > rs_threshold and h4d_seq.cameras[view][i].groupframe_id % rs_step == 0):
                        marker_pos_file.write("{0} {1} {2}\n".format(centers[cluster_id, 0], centers[cluster_id, 1], centers[cluster_id, 2]))
            else:
                for cluster_id in range(marker_3d_points_np.shape[0]):
                    if (h4d_seq.cameras[view][i].groupframe_id > rs_threshold and h4d_seq.cameras[view][i].groupframe_id % rs_step == 0):
                        marker_pos_file.write("{0} {1} {2}\n".format(marker_3d_points_np[cluster_id, 0], marker_3d_points_np[cluster_id, 1], marker_3d_points_np[cluster_id, 2]))

        
            if (h4d_seq.cameras[view][i].groupframe_id > rs_threshold and h4d_seq.cameras[view][i].groupframe_id % rs_step == 0):
                marker_pos_file.close()

            gt_joints_t_temp = gt_joints_t.clone()[gt_index].cpu().numpy()

            if (not os.path.exists(os.path.join(sequence_path, "Dump", "gposes3d"))):
                os.makedirs(os.path.join(sequence_path, "Dump", "gposes3d"))
            numpy.save(os.path.join(sequence_path, "Dump", "gposes3d", str(i) + ".npy"), numpy.transpose(gt_joints_t_temp, (1, 2, 0)))

            global_sequence_3d[i] = numpy.transpose(gt_joints_t_temp, (2, 1, 0)).copy()

            if (h4d_seq.cameras[view][i].groupframe_id > rs_threshold and h4d_seq.cameras[view][i].groupframe_id % rs_step == 0 and args.vis_3d):

                
                
                # ###########
                # translation_gt = torch.tensor([59.0, 80.0, 820.0]).reshape(3, 1)
                # r = R.from_euler('xyz',[0, -2, 124.5], degrees=True)
                # rotation_gt_np = r.as_dcm()
                # rotation_gt = torch.from_numpy(rotation_gt_np).type(torch.float)

                # rotation_gt_inv = torch.inverse(rotation_gt)
                # translation_gt_inv = - rotation_gt_inv @ translation_gt

                # trans_init = numpy.zeros([4, 4], dtype=numpy.float)
                # trans_init[0:3, 0:3] = rotation_gt_inv.cpu().numpy()
                # trans_init[0:3, 3] = translation_gt_inv.squeeze(1).cpu().numpy()
                # trans_init[3, 3] = 1
                
                
                # ##########
                # pcd_rs = open3d.geometry.PointCloud()
                # pcd_rs.points = open3d.utility.Vector3dVector(centers)

                # pcd_gt = open3d.geometry.PointCloud()
                # pcd_gt.points = open3d.utility.Vector3dVector(gt_markers_t_temp.copy().transpose())

                # # trans_init = np.asarray([
                # #          [1.0, 0.0, 0.0, 0.0],
                # #          [0.0, 1.0, 0.0, 0.0],
                # #          [0.0, 0.0, 1.0, 0.0],
                # #          [0.0, 0.0, 0.0, 1.0]])

                # transform_ = open3d.registration.registration_icp(pcd_gt, pcd_rs, 200, trans_init, 
                #             open3d.registration.TransformationEstimationPointToPoint(), open3d.registration.ICPConvergenceCriteria(max_iteration = 5000))

                


                # print(transform_)
                
                # pcd_gt.transform(transform_.transformation)
                # numpy.save(os.path.join(sequence_path, "../pose/", "extr.npy"), transform_.transformation)
                # # open3d.registration.evaluate_registration(pcd_gt, pcd_rs, 30, trans_init)
           
                ax = plt.axes(projection='3d')

                markers = ['o', '^', 'x', '+', '*']            
                # ax.scatter(gt_markers_t_temp[0], gt_markers_t_temp[1], gt_markers_t_temp[2], marker=markers[0]) 
                # pcd_gt_points = numpy.asarray(pcd_gt.points)
                # ax.scatter(pcd_gt_points[:, 0], pcd_gt_points[:, 1], pcd_gt_points[:, 2], marker=markers[1]) 

                # pcd_rs_points = numpy.asarray(pcd_rs.points)

                ax.scatter(gt_markers_t_temp.copy().transpose()[:, 0], gt_markers_t_temp.copy().transpose()[:, 1], gt_markers_t_temp.copy().transpose()[:, 2], marker=markers[2]) 
                ax.set_xlabel('X Label')
                ax.set_ylabel('Y Label')
                ax.set_zlabel('Z Label')


                ax.set_xlabel('X Label')
                ax.set_ylabel('Y Label')
                ax.set_zlabel('Z Label')

                if (doClustering):
                    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], marker=markers[4]) 
                    # ax.scatter(pcd_rs_points[:, 0], pcd_rs_points[:, 1], pcd_rs_points[:, 2], marker=markers[4]) 

                    ax.set_xlabel('X Label')
                    ax.set_ylabel('Y Label')
                    ax.set_zlabel('Z Label')

                 
                else:
                    ax.scatter(marker_3d_points_np[:, 0], marker_3d_points_np[:, 1], marker_3d_points_np[:, 2], marker=markers[4]) 
                    # ax.scatter(pcd_rs_points[:, 0], pcd_rs_points[:, 1], pcd_rs_points[:, 2], marker=markers[4]) 
                    ax.set_xlabel('X Label')
                    ax.set_ylabel('Y Label')
                    ax.set_zlabel('Z Label')

                plt.show()



            # if (len(numpy.unique(pindices)) == 53):
            if (args.do_sba):            
                view_id = 0
                viewpoint_indices = []
                point_indices = []
                x_true_values = [] 
                x_pred_values = [] 
                cameraArray = numpy.zeros([4, 6])

                for view in h4d_seq.camera_ids:
                    r = R.from_matrix(numpy.squeeze(rotation_inv[view].cpu().numpy(), axis=0))

                    cameraArray[view_id, :3] = r.as_rotvec()
                    # cameraArray[view_id, :3] = R.as_euler(r.as_euler()
                    cameraArray[view_id, 3:6] = numpy.squeeze(numpy.squeeze(translation_inv[view].cpu().numpy(), axis=0), axis=1)
                    # cameraArray[view_id, 3] = -500
                    # cameraArray[view_id, 4] = 0
                    # cameraArray[view_id, 6] = -500
                    # cameraArray[view_id, 3:6] = numpy.squeeze(numpy.squeeze(translation[view].cpu().numpy(), axis=0), axis=1)
                    # cameraArray[view_id, 6] = 4 * intr[view][0, 0]
                    # cameraArray[view_id, 7] = 4 * intr[view][0, 2]
                    # cameraArray[view_id, 8] = 4 * intr[view][1, 2]

                    counter = 0
                    pp.append(numpy.array((int(4 * intr[view][0, 2]), int(4 * intr[view][1, 2]))))
                    fs.append(4 * intr[view][0, 0].cpu().numpy())
                    for id in markers_2d_obs[view]:
                        viewpoint_indices.append(view_id)
                        point_indices.append(id)
                        x_true_values.append(markers2d_of_camera[view][counter] - pp[view_id])
                        x_pred_values.append(markers2d_of_vicon[view][counter])
                        counter += 1
                    view_id += 1


                x_true = numpy.array(x_true_values)
                x_pred = numpy.array(x_pred_values)
                vindices = numpy.array(viewpoint_indices)
                pindices = numpy.array(point_indices)
            
                sba = PySBA(cameraArray, gt_markers_t_temp.transpose(), x_true, vindices, pindices, pp, \
                    [h4d_seq.cameras["M72i"][i].color_img,
                    h4d_seq.cameras["M72j"][i].color_img,
                    h4d_seq.cameras["M72e"][i].color_img,
                    h4d_seq.cameras["M72h"][i].color_img], fs
                )
                results = sba.bundleAdjust()
                ax = plt.axes(projection='3d')

                ax.scatter(results[1].transpose()[0], results[1].transpose()[1], results[1].transpose()[2], marker=markers[4]) 
                ax.scatter(gt_markers_t_temp[0], gt_markers_t_temp[1], gt_markers_t_temp[2], marker=markers[0]) 

                ax.set_xlabel('X Label')
                ax.set_ylabel('Y Label')
                ax.set_zlabel('Z Label')

                plt.show()


                ax = plt.axes(projection='3d')
                view_id = 0
                for view in h4d_seq.camera_ids:
                    r_new = R.from_rotvec(results[0][view_id][:3])
                    r_new_mat = r_new.as_dcm()
                    t_new = results[0][view_id][3:6]

                    # rotation_new = torch.transpose(torch.from_numpy(r_new_mat).type(torch.float), 0, 1)
                    rotation_new = torch.from_numpy(r_new_mat).type(torch.float)
                    rotation_new_inv = torch.inverse(rotation_new)
                    translation_new = torch.tensor(t_new, dtype=torch.float32).reshape(3, 1)
                    # translation_new[2] = - translation_new[2]
                    translation_new_inv = - rotation_new_inv @ translation_new

                    # points_3d_t_new = transform_points(points_3d_all[view], rotation_new, translation_new)
                    points_3d_t_new = transform_points(points_3d_all[view], rotation_new_inv, translation_new_inv)
                    filename = "new_transformed_point_cloud_%s_%d.ply" % (view, i)
                    save_ply(filename, points_3d_t_new, 1, color=ply_colors[view_id])

                    transform_mat = torch.cat((rotation_new_inv, torch.transpose(translation_new_inv, 0, 1)))
                    numpy.savetxt(view + "_new.extrinsics", transform_mat.cpu().numpy())

                    ax.scatter(numpy.array(markers3d[view]).transpose()[0], numpy.array(markers3d[view]).transpose()[1], numpy.array(markers3d[view]).transpose()[2], marker=markers[view_id]) #, c=COLORS[format(view_id + 1, "02d")])
                    view_id += 1





        for view in h4d_seq.camera_ids:
            numpy.save(os.path.join(sequence_path, sequence_filename) + "_" + view + "_2d.npy", all_sequence_2d[view])
            numpy.save(os.path.join(sequence_path, sequence_filename) + "_" + view + "_3d.npy", all_sequence_3d[view])
            numpy.save(os.path.join(sequence_path, sequence_filename) + "_" + view + "_bbox.npy", all_sequence_bbox[view])

        numpy.save(os.path.join(sequence_path, sequence_filename) + "_global_3d.npy", global_sequence_3d)
        seq_id += 1
    

if __name__ == "__main__":
    main()
