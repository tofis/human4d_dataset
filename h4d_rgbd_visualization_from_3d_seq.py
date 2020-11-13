import os
from shutil import copyfile

import numpy
import cv2
from visualization import turbo_colormap

import torch

from importers import *
from vision import *
from exporters import *
from optimization import *
from visualization import *
from structs import *


# initial arguments and params
# root = 'G:/MULTI4D_Dataset/HUMAN4D/S12/19-07-12-13-08-49/Dump'
# root = 'G:/MULTI4D_Dataset/HUMAN4D/S3/19-07-12-09-55-48/Dump'
# root = 'G:/MULTI4D_Dataset/HUMAN4D/S2/19-07-12-08-54-26/Dump'
# root = 'G:/MULTI4D_Dataset/HUMAN4D/S34/19-07-12-13-58-54/Dump'
# root = 'G:/MULTI4D_Dataset/HUMAN4D/S1/19-07-12-07-49-24/Dump'
root = 'G:/MULTI4D_Dataset/HUMAN4D/S3/19-07-12-10-02-40/Dump'

resolution = [320, 180]
save_files = True

out = os.path.join(root, 'out')
if (not os.path.exists(out)):
    os.makedirs(out)

# COLORS = get_COLORS()
# COLORS = turbo_colormap.turbo_colormap_data
# COLORS = turbo_colormap.get_colors(150)

device_repo_path = os.path.join(root,"../../../device_repository.json")
if not os.path.exists(device_repo_path):
    raise ValueError("{0} does not exist".format(device_repo_path))            
device_repo = load_intrinsics_repository(os.path.join(device_repo_path))
device_repo_rgb = load_intrinsics_repository(os.path.join(device_repo_path), stream='RGB')
device_repo_RT = load_rotation_translation(os.path.join(device_repo_path))

extr_files = [current_ for current_ in os.listdir(os.path.join(root, "../../pose")) if ".extrinsics" in current_]

extrinsics = {}
paths = {}
intr = {}
intr_inv = {}
rotation = {}
translation = {}
rotation_inv = {}
translation_inv = {}
gt2d = {}
gtbbox = {}

views = []

for extr in extr_files:
    extrinsics[extr.split(".")[0]] = load_extrinsics(os.path.join(root, "../../pose", extr))[0]
    paths[extr.split(".")[0]] = os.path.join(root, "../../pose", extr.split(".")[0])
    views.append(extr.split(".")[0])

rgbd_skip = load_rgbd_skip(os.path.join(root, "../../offsets.txt"), os.path.basename(root.split('/')[4]))




uv_grid = create_image_domain_grid(resolution[0], resolution[1])

for view in views:
    intr[view], intr_inv[view] = get_intrinsics(view, device_repo, 4)
    

colorz = [color for color in os.listdir(os.path.join(root, 'color'))]
depthz = [depth for depth in os.listdir(os.path.join(root, 'depth'))]

npyz = {}
for view in views:
    npyz[view] = [npy for npy in os.listdir(os.path.join(root, '../')) \
        if view in npy and '2d' in npy][0]

npybbz = {}
for view in views:
    npybbz[view] = [npy for npy in os.listdir(os.path.join(root, '../')) \
        if view in npy and 'bbox' in npy][0]

for color_file in colorz:
    sample_id_s = os.path.basename(color_file).split('_')[0]
    sample_id = int(sample_id_s)
    if sample_id < rgbd_skip:
        continue
    gt3d = numpy.expand_dims(numpy.load(os.path.join(root, 'gposes3d', str(sample_id-rgbd_skip) + '.npy')), axis=0)
    gt_joints_t = torch.from_numpy(gt3d).reshape(gt3d.shape[0], gt3d.shape[1], gt3d.shape[2], gt3d.shape[3]).permute(0, 3, 1, 2).type(torch.float)
    view = color_file.split('_')[1]
    
    gt2d[view] = numpy.load(os.path.join(root, '../', npyz[view]))[sample_id-rgbd_skip]
    gtbbox[view] = numpy.load(os.path.join(root, '../', npybbz[view]))[sample_id-rgbd_skip]
    # gtbbox[view] = gtbbox[view][::-1]

    # depth = [depth_file for depth_file in depthz if view in depth_file][0]
    # color = [img_file for img_file in colorz if view in img_file][0]
    # depth_img = readpgm(os.path.join(root, 'depth', depth)).astype(numpy.float) / 10.0
    # depth_img_mask = depth_img < 3000
    # depth_img *= depth_img_mask
    color_img = cv2.imread(os.path.join(root, 'color', color_file))

    ######### color
    for p in range(gt2d[view].shape[0]):
        for j in range(gt2d[view].shape[1]):                
            # keypoints3d[p, j] = gt_joints_view_aligned[0, :, p, j].cpu().numpy()
            # if ("INF" in args.sequence_filename):
            uv = gt2d[view][p, j]
            # else:
            #     uv = project_single_point_to_uv(gt_joints_view_aligned[0, :, p, j], intr_rgb[view])
                # keypoints[p, j] = uv

            print("uv: " + str(uv) + " p: " + str(p) + " j: " + str(j))
            

    p_offset = 20
    for p in range(gt2d[view].shape[0]):
        # min_x = int(numpy.min(marker_keypoints[p, :, 0]) - p_offset)
        # min_y = int(numpy.min(marker_keypoints[p, :, 1]) - p_offset)
        # max_x = int(numpy.max(marker_keypoints[p, :, 0]) + p_offset)
        # max_y = int(numpy.max(marker_keypoints[p, :, 1]) + p_offset)
        
        # width = args.resolution[0] * 4 
        # height = args.resolution[1] * 4 
        # all_sequence_bbox[view][i, p, 0] = min_x if min_x >= 0 else 0 
        # all_sequence_bbox[view][i, p, 1] = min_y if min_y >= 0 else 0 
        # all_sequence_bbox[view][i, p, 2] = max_x if max_x <= width - 1 else width - 1
        # all_sequence_bbox[view][i, p, 3] = max_y if max_y <= height - 1 else height - 1 

        # cv2.rectangle(img_c, (all_sequence_bbox[view][i, p, 0], all_sequence_bbox[view][i, p, 1]), \
        #     (all_sequence_bbox[view][i, p, 2], all_sequence_bbox[view][i, p, 3]), (250, 100, 100), thickness=2)

        draw_skeleton_joints(color_img, gt2d[view][p], turbo_colormap.get_colors(150, p), thickness=6)     
        for j in range(gt2d[view].shape[1]):
            uv = gt2d[view][p, j]
            # color_img = cv2.drawMarker(color_img, 
            #     (int(uv[0]), int(uv[1])), 
            #     # COLORS[format(j+1, '02d')],
            #     turbo_colormap.get_colors(200)[j],
            #     markerType=cv2.MARKER_DIAMOND,
            #     markerSize=10,
            #     thickness=2)
            color_img = cv2.circle(color_img, 
                (int(uv[0]), int(uv[1])), 
                # COLORS[format(j+1, '02d')],
                radius=6,
                color=turbo_colormap.get_colors(200, p)[j],
                thickness=3)
        
        # color_img = cv2.rectangle(color_img, (gtbbox[view][p, 0], gtbbox[view][p, 1]), (gtbbox[view][p, 2], gtbbox[view][p, 3]),\
        #      turbo_colormap.get_colors(150, p)[0], thickness=2)
        # if (i in args.frames2save):
        # path = os.path.join(root, "perfcap", "{}_{}.png".format(str(sample_id), view))
        # if not os.path.exists(path):
        #     os.makedirs(path)
        # cv2.imwrite(path, \
        #         color_img)
    # cv2.imshow("colored", cv2.rotate(color_img, cv2.ROTATE_90_CLOCKWISE))
    # cv2.waitKey(1)

    rotation[view], translation[view] = extract_rotation_translation(extrinsics[view].unsqueeze(0))
    rotation_inv[view] = torch.inverse(rotation[view])
    translation_inv[view] = - rotation_inv[view] @ translation[view]

    # gt_markers_view_aligned = transform_points(gt3d.unsqueeze(0), rotation_inv[view], translation_inv[view])
    gt_joints_view_aligned = transform_points(gt_joints_t, rotation_inv[view], translation_inv[view])

    # depth_norm = (depth_img / numpy.max(depth_img) * 255).astype(numpy.uint8)
    # # depth_norm = (depth_img / 3000 * 255).astype(numpy.uint8)
    # depth_img_c = cv2.cvtColor(depth_norm, cv2.COLOR_GRAY2RGB)

    keypoints = numpy.zeros([gt_joints_t.shape[2], gt_joints_t.shape[3], 2])
    for p in range(gt_joints_t.shape[2]):
        for j in range(gt_joints_t.shape[3]):                
            # keypoints3d[p, j] = gt_joints_view_aligned[0, :, p, j].cpu().numpy()
            # if ("INF" in args.sequence_filename):
            uv = project_single_point_to_uv(gt_joints_view_aligned[0, :, p, j], intr[view])
            keypoints[p, j] = uv
            # else:
            #     uv = project_single_point_to_uv(gt_joints_view_aligned[0, :, p, j], intr_rgb[view])
                # keypoints[p, j] = uv

            print("uv: " + str(uv) + " p: " + str(p) + " j: " + str(j))
            
    # p_offset = 20
    # for p in range(gt_joints_t.shape[2]):
        # min_x = int(numpy.min(marker_keypoints[p, :, 0]) - p_offset)
        # min_y = int(numpy.min(marker_keypoints[p, :, 1]) - p_offset)
        # max_x = int(numpy.max(marker_keypoints[p, :, 0]) + p_offset)
        # max_y = int(numpy.max(marker_keypoints[p, :, 1]) + p_offset)
        
        # width = args.resolution[0] * 4 
        # height = args.resolution[1] * 4 
        # all_sequence_bbox[view][i, p, 0] = min_x if min_x >= 0 else 0 
        # all_sequence_bbox[view][i, p, 1] = min_y if min_y >= 0 else 0 
        # all_sequence_bbox[view][i, p, 2] = max_x if max_x <= width - 1 else width - 1
        # all_sequence_bbox[view][i, p, 3] = max_y if max_y <= height - 1 else height - 1 

        # cv2.rectangle(img_c, (all_sequence_bbox[view][i, p, 0], all_sequence_bbox[view][i, p, 1]), \
        #     (all_sequence_bbox[view][i, p, 2], all_sequence_bbox[view][i, p, 3]), (250, 100, 100), thickness=2)

    #     draw_skeleton_joints(depth_img_c, keypoints[p], turbo_colormap.get_colors(150, p), thickness=2)     
    #     for j in range(gt_joints_t.shape[3]): 
    #         uv =  keypoints[p, j]
    #         # depth_img_c = cv2.drawMarker(depth_img_c, 
    #         #     (int(uv[0]), int(uv[1])), 
    #         #     # COLORS[format(j+1, '02d')],
    #         #     turbo_colormap.get_colors(200)[j],
    #         #     markerType=cv2.MARKER_CROSS,
    #         #     markerSize=3,
    #         #     thickness=1)
            
    #         depth_img_c = cv2.circle(depth_img_c, 
    #             (int(uv[0]), int(uv[1])), 
    #             # COLORS[format(j+1, '02d')],
    #             radius=1,
    #             color=turbo_colormap.get_colors(200, p)[j],
    #             thickness=1)

    #     depth_img_c = cv2.rectangle(depth_img_c, (gtbbox[view][p, 0] // 4, gtbbox[view][p, 1] // 4), (gtbbox[view][p, 2] // 4, gtbbox[view][p, 3] // 4), \
    #         turbo_colormap.get_colors(150, p)[0], thickness=1)

    #     # color_img = cv2.rectangle(color_img, (gtbbox[view][p, 0], gtbbox[view][p, 1]), (gtbbox[view][p, 2], gtbbox[view][p, 3]),\
    #     #      turbo_colormap.get_colors(150, p)[0], thickness=2)

    #     # if (i in args.frames2save):
    #     #     cv2.imwrite(os.path.join(args.sequence_path, "{}_{}.png".format(str(i), view)), \
    #     #             img_c)
    # cv2.imshow("colored", cv2.rotate(depth_img_c, cv2.ROTATE_90_CLOCKWISE))
    # cv2.waitKey(1000)

    # points_3d_all[view] = deproject_depth_to_points(depth_t, uv_grid, intr_inv[view], floor_y=1000)
    # rotation[view], translation[view] = extract_rotation_translation(extrinsics[view].unsqueeze(0))
    # rotation_inv[view] = torch.inverse(rotation[view])
    # translation_inv[view] = - rotation_inv[view] @ translation[view]
    # points_3d_t = transform_points(points_3d_all[view], rotation[view], translation[view])


    # depth_img[depth_img > 3000] = 0
    # depth_img[depth_img < 1200] = 0
    # depth_img /= numpy.max(depth_img)
    # depth_img /= 10000
    # depth_img = (depth_img * 255).astype(numpy.uint8)
    # img = numpy.zeros([180, 320, 3], dtype=numpy.float32)
    # for x in range(img.shape[1]):
    #     for y in range(img.shape[0]):
    #         img[y, x] = turbo_colormap.turbo_colormap_data[depth_img[y, x]]
    # # img = cv2.LUT(depth_img, numpy.asarray(turbo_colormap.turbo_colormap_data).astype(numpy.float32))
    # img = (img * 255).astype(numpy.uint8)
    # cv2.imshow("show", cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))

    if (save_files):
        # cv2.imwrite(os.path.join(out, depth).replace('.pgm', '_col.png'),  cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))
        # cv2.imwrite(os.path.join(out, depth).replace('.pgm', '.png'),  cv2.rotate(depth_img_c, cv2.ROTATE_90_CLOCKWISE))
        cv2.imwrite(os.path.join(out, color_file),  cv2.rotate(color_img, cv2.ROTATE_90_CLOCKWISE))
    
    # cv2.waitKey(1000)




