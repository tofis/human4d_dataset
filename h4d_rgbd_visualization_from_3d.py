import os
from shutil import copyfile

import numpy
import cv2
from visualization import turbo_colormap

import torch

from importers import *
from vision import *
from exporters import *
from visualization import *
from structs import *

root = 'E:/Datasets/HUMAN4D/S1/19-07-12-07-32-22/Dump'

resolution = [320, 180]

out = os.path.join(root, 'out')
if (not os.path.exists(out)):
    os.makedirs(out)
sample_id = 100
sample_id_s = str(sample_id)

device_repo_path = os.path.join(root,"../../../device_repository.json")
if not os.path.exists(device_repo_path):
    raise ValueError("{0} does not exist".format(device_repo_path))            
device_repo = load_intrinsics_repository(os.path.join(device_repo_path))
device_repo_rgb = load_intrinsics_repository(os.path.join(device_repo_path), stream='RGB')
device_repo_RT = load_rotation_translation(os.path.join(device_repo_path))

extr_files = [current_ for current_ in os.listdir(os.path.join(root, "../../pose")) if ".extrinsics" in current_]

extrinsics = {}
paths = {}
intr_color = {}
intr_depth = {}
rotation = {}
translation = {}
rotation_inv = {}
translation_inv = {}
gtbbox = {}

views = []

for extr in extr_files:
    extrinsics[extr.split(".")[0]] = load_extrinsics(os.path.join(root, "../../pose", extr))[0]
    paths[extr.split(".")[0]] = os.path.join(root, "../../pose", extr.split(".")[0])
    views.append(extr.split(".")[0])

rgbd_skip = load_rgbd_skip(os.path.join(root, "../../offsets.txt"), os.path.basename(root.split('/')[4]))

gt3d = numpy.expand_dims(numpy.load(os.path.join(root, 'gposes3d', str(sample_id-rgbd_skip) + '.npy')), axis=0)

gt_joints_t = torch.from_numpy(gt3d).reshape(gt3d.shape[0], gt3d.shape[1], gt3d.shape[2], gt3d.shape[3]).permute(0, 3, 1, 2).type(torch.float)

uv_grid = create_image_domain_grid(resolution[0], resolution[1])

for view in views:
    intr_color[view], _ = get_intrinsics(view, device_repo_rgb, 1)    
    intr_depth[view], _ = get_intrinsics(view, device_repo, 4)    

colorz = [color for color in os.listdir(os.path.join(root, 'color')) \
    if sample_id_s in color.split('_')[0]]
depthz = [depth for depth in os.listdir(os.path.join(root, 'depth')) \
    if sample_id_s in depth.split('_')[0]]

npyz = {}
for view in views:
    npyz[view] = [npy for npy in os.listdir(os.path.join(root, '../')) \
        if view in npy and '2d' in npy][0]

npybbz = {}
for view in views:
    npybbz[view] = [npy for npy in os.listdir(os.path.join(root, '../')) \
        if view in npy and 'bbox' in npy][0]

for view in views:
    gtbbox[view] = numpy.load(os.path.join(root, '../', npybbz[view]))[sample_id-rgbd_skip]
    depth = [depth_file for depth_file in depthz if view in depth_file][0]
    color = [img_file for img_file in colorz if view in img_file][0]
    depth_img = readpgm(os.path.join(root, 'depth', depth)).astype(numpy.float) / 10.0
    depth_img_mask = depth_img < 3000
    depth_img *= depth_img_mask
    color_img = cv2.imread(os.path.join(root, 'color', color))       

    p_offset = 20
    
    rotation[view], translation[view] = extract_rotation_translation(extrinsics[view].unsqueeze(0))
    rotation_inv[view] = torch.inverse(rotation[view])
    translation_inv[view] = - rotation_inv[view] @ translation[view]

    gt_joints_view_aligned = transform_points(gt_joints_t, rotation_inv[view], translation_inv[view])

    R_rgbd = device_repo_RT[view]['R']
    t_rgbd = device_repo_RT[view]['t'] * 1000.0
    R_rgbd_inv = numpy.linalg.inv(R_rgbd)
    t_rgbd_inv = - R_rgbd_inv @ t_rgbd

    gt_joints_view_aligned_color = transform_points(gt_joints_view_aligned, torch.from_numpy(R_rgbd_inv), torch.from_numpy(t_rgbd_inv))

    depth_norm = (depth_img / numpy.max(depth_img) * 255).astype(numpy.uint8)
    depth_img_c = cv2.cvtColor(depth_norm, cv2.COLOR_GRAY2RGB)    

    depth_img /= numpy.max(depth_img)
    depth_img = (depth_img * 255).astype(numpy.uint8)
    colored_depth_img = numpy.zeros([depth_img.shape[0], depth_img.shape[1], 3], dtype=numpy.float32)
    for x in range(colored_depth_img.shape[1]):
        for y in range(colored_depth_img.shape[0]):
            colored_depth_img[y, x] = turbo_colormap.turbo_colormap_data[depth_img[y, x]]
    colored_depth_img = (colored_depth_img * 255).astype(numpy.uint8)

    keypoints_gt_color = numpy.zeros([gt_joints_t.shape[2], gt_joints_t.shape[3], 2])
    keypoints_gt_depth = numpy.zeros([gt_joints_t.shape[2], gt_joints_t.shape[3], 2])
    
    for p in range(gt_joints_t.shape[2]):
        for j in range(gt_joints_t.shape[3]):         
            uv = project_single_point_to_uv(gt_joints_view_aligned_color[0, :, p, j], intr_color[view])
            keypoints_gt_color[p, j] = uv

            uv = project_single_point_to_uv(gt_joints_view_aligned[0, :, p, j], intr_depth[view])
            keypoints_gt_depth[p, j] = uv
            
    p_offset = 20
    for p in range(gt_joints_t.shape[2]): 
        for j in range(gt_joints_t.shape[3]): 
            uv =  keypoints_gt_color[p, j]            
            color_img = cv2.circle(color_img, 
                (int(uv[0]), int(uv[1])), 
                radius=4,
                color=turbo_colormap.get_colors(200, p)[j],
                thickness=2)  

            uv =  keypoints_gt_depth[p, j]
            colored_depth_img = cv2.circle(colored_depth_img, 
                (int(uv[0]), int(uv[1])), 
                radius=2,
                color=turbo_colormap.get_colors(10, p)[j],
                thickness=1)       

        color_img = cv2.rectangle(color_img, (gtbbox[view][p, 0], gtbbox[view][p, 1]), (gtbbox[view][p, 2], gtbbox[view][p, 3]), \
             turbo_colormap.get_colors(150, p)[0], thickness=1)

        colored_depth_img = cv2.rectangle(colored_depth_img, (gtbbox[view][p, 0] // 4, gtbbox[view][p, 1] // 4), (gtbbox[view][p, 2] // 4, gtbbox[view][p, 3] // 4), \
             turbo_colormap.get_colors(150, p)[0], thickness=1)

    cv2.imshow("colored", cv2.rotate(color_img, cv2.ROTATE_90_CLOCKWISE))
    cv2.imshow("depth", cv2.rotate(colored_depth_img, cv2.ROTATE_90_CLOCKWISE))   
    cv2.waitKey(5000)




