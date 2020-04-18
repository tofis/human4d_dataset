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

ply_colors = [ 'red', 'blue', 'orange', 'green', 'brown' ]

def main():
    parser = argparse.ArgumentParser(description="PyTorch Show Pose and Pointcloud")
    parser.add_argument(
        "--show_markers",
        default=True,      
        help="render marker positions",
    )
    parser.add_argument(
        "--show_joints",
        default=True,      
        help="render joint positions",
    )
    parser.add_argument(
        "--sequence_path",
        default="E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/19-07-12-10-12-49",      
        # default="G:/MULTI4D_Dataset/multi/rgbd_subjects1and2/19-07-12-13-05-08",      
        help="path to sequence files",
    )
    parser.add_argument(
        "--sequence_filename", 
        default="INF_Running_S3_01_eval",      
        # default="RGB_WatchingFootball_S1S2_02_eval",      
        help="path to sequence files",
    )
    parser.add_argument(
        "--resolution",     
        nargs="*", type=int, 
        default = [320, 180],
        help="input resolution",
    )

    args = parser.parse_args()

    # trying CDF format

    # test_sequence_2d = numpy.ones([10, 33 * 2])
    # size = numpy.zeros([10])

    # cdf = pycdf.CDF(args.sequence_filename + '__.cdf', '')
    # cdf['Pose'] = pycdf.numpy.zeros([10, 66])
    # for i in range(10):
    #     cdf['Pose'].insert(i, test_sequence_2d[i])
    # cdf.close()

    COLORS = get_COLORS()
    h4d_seq = H4DSequence(os.path.join(args.sequence_path, "Dump"), ["M72i", "M72j", "M72e", "M72h"])
   
    device_repo_path = os.path.join(args.sequence_path,"device_repository.json")
    if not os.path.exists(device_repo_path):
        raise ValueError("{0} does not exist".format(device_repo_path))            
    device_repo = load_intrinsics_repository(os.path.join(device_repo_path))

    extr_files = [current_ for current_ in os.listdir(args.sequence_path) if ".extrinsics" in current_]

    extrinsics = {}
    paths = {}
    views = []

    for extr in extr_files:
        extrinsics[extr.split(".")[0]] = load_extrinsics(os.path.join(args.sequence_path, extr))[0]
        paths[extr.split(".")[0]] = os.path.join(args.sequence_path, extr.split(".")[0])
        views.append(extr.split(".")[0])

    gt_joints = load_joints_seq(os.path.join(args.sequence_path, args.sequence_filename + ".joints"))
    gt_markers = load_markers_seq(os.path.join(args.sequence_path, args.sequence_filename + ".markers"))
    time_step = 8.33333
    translation_gt = torch.tensor([59.0,  80.0, 820.0]).reshape(3, 1)
    r = R.from_euler('xyz',[0, -2, 124.5], degrees=True)
    rotation_gt_np = r.as_matrix()

    q = r.inv().as_quat()

    rotation_gt = torch.from_numpy(rotation_gt_np).type(torch.float)

    rotation_gt_inv = torch.inverse(rotation_gt)
    translation_gt_inv = - rotation_gt_inv @ translation_gt
    gt_markers_t = torch.from_numpy(gt_markers).reshape(gt_markers.shape[0], gt_markers.shape[1], gt_markers.shape[2], 1).permute(0, 2, 1, 3).type(torch.float)
    gt_joints_t = torch.from_numpy(gt_joints).reshape(gt_joints.shape[0], gt_joints.shape[1], gt_joints.shape[2], 1).permute(0, 2, 1, 3).type(torch.float)
    # transform vicon
    gt_markers_t = transform_points(gt_markers_t, rotation_gt_inv, translation_gt_inv)
    gt_joints_t = transform_points(gt_joints_t, rotation_gt_inv, translation_gt_inv)       

    all_sequence_2d = {}
    all_sequence_3d = {}
    for view in h4d_seq.camera_ids:  
        all_sequence_2d[view] = numpy.zeros([h4d_seq.num_of_frames, 33, 2])
        all_sequence_3d[view] = numpy.zeros([h4d_seq.num_of_frames, 33, 3])

    #for i in range(h4d_seq.num_of_frames):
    for i in range(281):
        #index = i for h4d_seq.cameras["M72i"][i].groupframe_id == i
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

        
        for view in h4d_seq.camera_ids:      
            print(view + " " + str(h4d_seq.cameras[view][i].groupframe_id))
            gt_index = int(h4d_seq.cameras[view][i].timestamp / time_step)

            if gt_index > gt_joints.shape[0]:
                continue

            depth_t = torch.from_numpy(\
                h4d_seq.cameras[view][i].depth_img.reshape(1, 1, h4d_seq.cameras[view][i].depth_img.shape[0], h4d_seq.cameras[view][i].depth_img.shape[1])).float()

            uv_grid = create_image_domain_grid(args.resolution[0], args.resolution[1])

            intr[view], intr_inv[view] = get_intrinsics(view, device_repo, 4)
            points_3d = deproject_depth_to_points(depth_t, uv_grid, intr_inv[view], floor_y=1000)
            rotation[view], translation[view] = extract_rotation_translation(extrinsics[view].unsqueeze(0))
            rotation_inv[view] = torch.inverse(rotation[view])
            translation_inv[view] = - rotation_inv[view] @ translation[view]
            points_3d_t = transform_points(points_3d, rotation[view], translation[view])
            points_3d = points_3d.permute(2, 3, 1, 0).squeeze()
            points_3d_t = points_3d_t.permute(2, 3, 1, 0).squeeze()

            ################## projection ####################
            img_c = h4d_seq.cameras[view][i].color_img.copy()
            #img_c = cv2.resize(h4d_seq.cameras[view][i].depth_img.copy(), (1280, 720), interpolation = cv2.INTER_AREA).astype(numpy.uint16)
            #img_c = cv2.cvtColor(img_c, cv2.COLOR_GRAY2BGR)

            gt_markers_view_aligned = transform_points(gt_markers_t[gt_index].unsqueeze(0), rotation_inv[view], translation_inv[view])
            gt_joints_view_aligned = transform_points(gt_joints_t[gt_index].unsqueeze(0), rotation_inv[view], translation_inv[view])
            # projected = project_points_to_uvs(gt_markers_view_aligned / 1000.0, intr)
            
            marker_visibility = numpy.zeros([53], dtype=int)
            marker_keypoints = numpy.zeros([53, 2], dtype=int)
            

            for j in range(53):
                uv = project_single_point_to_uv(gt_markers_view_aligned[0, :, j, 0], 4 * intr[view])
                marker_keypoints[j] = uv

                if (args.show_markers): 
                    if ("INF" in args.sequence_filename):             
                        depth_diff = numpy.abs(int(depth_t[0, 0, int(uv[1]/4), int(uv[0]/4)]) - gt_markers_view_aligned[0, 2, j, 0])
                        if (depth_diff < 50):
                            marker_visibility[j] = 1
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
            # rectangle def and drawing
            min_x = numpy.min(marker_keypoints[:, 0])
            min_y = numpy.min(marker_keypoints[:, 1])
            max_x = numpy.max(marker_keypoints[:, 0])
            max_y = numpy.max(marker_keypoints[:, 1])

            cv2.rectangle(img_c, (min_x, min_y), (max_x, max_y), (0, 100, 100), thickness=1)

            keypoints = numpy.zeros([33, 2], dtype=int)
            keypoints3d = numpy.zeros([33, 3], dtype=int)
            for j in range(33):
                keypoints3d[j] = gt_joints_view_aligned[0, :, j, 0].cpu().numpy()
                uv = project_single_point_to_uv(gt_joints_view_aligned[0, :, j, 0], 4 * intr[view])
                keypoints[j] = uv
                if (args.show_joints):
                    img_c = cv2.drawMarker(img_c, 
                                        (int(uv[0]), int(uv[1])), 
                                        COLORS[format(j+1, '02d')],
                                        markerType=cv2.MARKER_STAR,
                                        markerSize=10,
                                        thickness=2)

            all_sequence_2d[view][i] = keypoints
            all_sequence_3d[view][i] = keypoints3d

            if (args.show_joints):
                draw_skeleton_joints(img_c, keypoints, COLORS)
                

            ########## marker matching ###################
            gray_img = cv2.cvtColor(h4d_seq.cameras[view][i].color_img, cv2.COLOR_BGR2GRAY)
           
            _, mask_img = cv2.threshold(gray_img, 150, 255, cv2.THRESH_BINARY)
            
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
                if 5 < area < 150:
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
            

                if (cY > 0.0 * h and cY < 1.0 * h and \
                    cX > 0.0 * w and cX < 1.0 * w and \
                    points_3d[cY, cX][2].cpu().numpy() > 300 and \
                    points_3d[cY, cX][2].cpu().numpy() < 2900):
                    markers2d[view].append((4 * cX, 4 * cY))  
                    markers3d[view].append(points_3d_t[cY, cX].cpu().numpy())  

                    # cv2.drawContours(img_c, [marker], -1, (0, 255, 0), 1)
                    # cv2.circle(img_c, (4 * cX, 4 *cY), 3, (255, 0, 0), -1)
                    # cv2.putText(img_c, "center", (4 * cX - 20, 4 * cY - 20),
                    # cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            
            gt_markers_t_temp = gt_markers_t.clone().squeeze()[gt_index].cpu().numpy()
            
            bip_graph = nx.Graph()
            for x in range(gt_markers_t_temp.shape[1]):
                bip_graph.add_node(format(x, "05d"), bipartite=0)
                for y in range(len(markers3d[view])):
                    bip_graph.add_node(format(y, "03d"), bipartite=1)
                    
            
            for x in range(gt_markers_t_temp.shape[1]):
                for y in range(len(markers3d[view])):
                    cost = numpy.linalg.norm(gt_markers_t_temp[:, x] - markers3d[view][y])
                    if (cost < 500):
                        bip_graph.add_edge(format(x, "05d"), format(y, "03d"), weight = cost)
                    else:
                        bip_graph.add_edge(format(x, "05d"), format(y, "03d"), weight = 10000)

            matches = nx.algorithms.bipartite.minimum_weight_full_matching(bip_graph)

            vicon_id = 0
            for match_id in matches:
                if (len(match_id) == 5):
                    print(match_id + " " + matches[match_id] + ": " + str(bip_graph[match_id][matches[match_id]]['weight']))
                    rs_id = int(matches[match_id])                    
                    
                    if (bip_graph[match_id][matches[match_id]]['weight'] < 50):
                        markers_2d_obs[view].append(int(match_id))

                        markers2d_of_camera[view].append(markers2d[view][rs_id])
                        markers2d_of_vicon[view].append((int(marker_keypoints[int(match_id)][0]), int(marker_keypoints[int(match_id)][1])))

                        cv2.line(img_c, 
                                    markers2d[view][rs_id], 
                                    (int(marker_keypoints[int(match_id)][0]), int(marker_keypoints[int(match_id)][1])), 
                                    COLORS["{:02d}".format(int(match_id) + 1)], 3)
                    vicon_id += 1

            cv2.imshow("color_" + view, cv2.transpose(img_c))
            view_id += 1
        
        for view in h4d_seq.camera_ids:
            numpy.save(os.path.join(args.sequence_path, args.sequence_filename) + "_" + view + "_2d.npy", all_sequence_2d[view])
            numpy.save(os.path.join(args.sequence_path, args.sequence_filename) + "_" + view + "_3d.npy", all_sequence_3d[view])


        gt_markers_t_temp = gt_markers_t.clone().squeeze()[gt_index].cpu().numpy()


        view_id = 0
        viewpoint_indices = []
        point_indices = []
        x_true_values = [] 
        x_pred_values = [] 
        cameraArray = numpy.zeros([4, 9])

        for view in h4d_seq.camera_ids:
            r = R.from_matrix(numpy.squeeze(rotation_inv[view].cpu().numpy(), axis=0))

            cameraArray[view_id, :3] = r.as_rotvec()
            # cameraArray[view_id, :3] = R.as_euler(r.as_euler()
            cameraArray[view_id, 3:6] = numpy.squeeze(numpy.squeeze(translation_inv[view].cpu().numpy(), axis=0), axis=1)
            # cameraArray[view_id, 3] = -500
            # cameraArray[view_id, 4] = 0
            # cameraArray[view_id, 6] = -500
            # cameraArray[view_id, 3:6] = numpy.squeeze(numpy.squeeze(translation[view].cpu().numpy(), axis=0), axis=1)
            cameraArray[view_id, 6] = 4 * intr[view][0, 0]
            # cameraArray[view_id, 7] = 4 * intr[view][0, 2]
            # cameraArray[view_id, 8] = 4 * intr[view][1, 2]

            counter = 0
            for id in markers_2d_obs[view]:
                viewpoint_indices.append(view_id)
                point_indices.append(id)
                x_true_values.append(markers2d_of_camera[view][counter])
                x_pred_values.append(markers2d_of_vicon[view][counter])
                counter += 1
            view_id += 1


        x_true = numpy.array(x_true_values)
        x_pred = numpy.array(x_pred_values)
        vindices = numpy.array(viewpoint_indices)
        pindices = numpy.array(point_indices)

       
        if (len(numpy.unique(pindices)) == 53):
            sba = PySBA(cameraArray, gt_markers_t_temp.transpose(), x_true, vindices, pindices)
            results = sba.bundleAdjust()

            # n_visible = len(viewpoint_indices)
            # # A = numpy.zeros([n_visible, 2, len(numpy.ndarray.flatten(rotation[view].cpu().numpy())) + len(numpy.ndarray.flatten(translation[view].cpu().numpy()))])
            # A = numpy.random.random(size=(n_visible, 2, len(numpy.ndarray.flatten(rotation[view].cpu().numpy())) + len(numpy.ndarray.flatten(translation[view].cpu().numpy()))))  
            # # A = numpy.random.random(size=(n_visible, 2, 4))  # n_pose_params = 4
            # B = numpy.random.random(size=(n_visible, 2, 3))  # n_point_params = 3
            # sba = SBA(numpy.array(viewpoint_indices), numpy.array(point_indices), do_check_args=True)
            # results = sba.compute(x_true, x_pred, A, B)

            # print("done")

            ax = plt.axes(projection='3d')
            markers = ['o', '^', 'x', '+', '*']

            ax.scatter(results[1].transpose()[0], results[1].transpose()[1], results[1].transpose()[2], marker=markers[4]) 
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')

            plt.show()


        # ax = plt.axes(projection='3d')
        # markers = ['o', '^', 'x', '+', '*']
        # view_id = 0
        # for view in h4d_seq.camera_ids:
        #     ax.scatter(numpy.array(markers3d[view]).transpose()[0], numpy.array(markers3d[view]).transpose()[1], numpy.array(markers3d[view]).transpose()[2], marker=markers[view_id]) #, c=COLORS[format(view_id + 1, "02d")])
        #     view_id += 1

        # ax.scatter(gt_markers_t_temp[0], gt_markers_t_temp[1], gt_markers_t_temp[2], marker=markers[4]) 
        # ax.set_xlabel('X Label')
        # ax.set_ylabel('Y Label')
        # ax.set_zlabel('Z Label')

        # plt.show()
        # # bundle_adjustement_opt = PySBA()
        
        # cv2.waitKey(100)
    
    

if __name__ == "__main__":
    main()
