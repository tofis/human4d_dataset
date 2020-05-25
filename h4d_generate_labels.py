import pickle
import numpy as np
import os.path as osp
from scipy.io import loadmat
from subprocess import call
from os import makedirs
import os
# from spacepy import pycdf
import sys

from importers import *
from vision import *

from scipy.spatial.transform import Rotation as R
from scipy import ndimage, misc
# from metadata import load_h36m_metadata
# metadata = load_h36m_metadata()

def _infer_box(pose3d, camera, rootIdx):
    root_joint = pose3d[rootIdx, :]
    tl_joint = root_joint.copy()
    tl_joint[0] -= 1000.0
    tl_joint[1] -= 900.0
    br_joint = root_joint.copy()
    br_joint[0] += 1000.0
    br_joint[1] += 1100.0
    tl_joint = np.reshape(tl_joint, (1, 3))
    br_joint = np.reshape(br_joint, (1, 3))

    tl2d = _weak_project(tl_joint, camera['fx'], camera['fy'], camera['cx'],
                         camera['cy']).flatten()

    br2d = _weak_project(br_joint, camera['fx'], camera['fy'], camera['cx'],
                         camera['cy']).flatten()
    return np.array([tl2d[0], tl2d[1], br2d[0], br2d[1]])


def _weak_project(pose3d, fx, fy, cx, cy):
    pose2d = pose3d[:, :2] / pose3d[:, 2:3]
    pose2d[:, 0] *= fx
    pose2d[:, 1] *= fy
    pose2d[:, 0] += cx
    pose2d[:, 1] += cy
    return pose2d

if __name__ == '__main__':
    # subject_list = [1, 5, 6, 7, 8, 9, 11]
    subject_list = ["S3"]
    # action_list = [x for x in range(2, 17)]
    # action_list = ["running"]
    action_list = ["talking"]
    subaction_list = ["sub"]
    # camera_list = [x for x in range(1, 5)]
    camera_list = ['M72e', 'M72h', 'M72i', 'M72j']

    train_list = [3]
    test_list = [3]

    # joint_idx = [0, 1, 2, 3, 6, 7, 8, 12, 16, 14, 15, 17, 18, 19, 25, 26, 27]
    joint_idx = [23,22,21,27,28,29,0,4,6,8,12,11,10,16,17,18,7]

    # with open('camera_data.pkl', 'rb') as f:
    #     camera_data = pickle.load(f)

    h36m_root = "E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/experimentation_dataset"
    device_repo_path = os.path.join(h36m_root, "device_repository.json")
    device_repo = load_intrinsics_repository(os.path.join(device_repo_path))

   

    train_db = []
    test_db = []
    cnt = 0

    for s in subject_list:
        extr_files = [current_ for current_ in os.listdir(os.path.join(h36m_root, s)) if ".extrinsics" in current_]
        extrinsics = {}
        paths = {}
        views = []

        for extr in extr_files:
            extrinsics[extr.split(".")[0]] = load_extrinsics(os.path.join(h36m_root, s, extr))[0]
            # paths[extr.split(".")[0]] = os.path.join(args.sequence_path, extr.split(".")[0])
            views.append(extr.split(".")[0])
        for a in action_list:
            for sa in subaction_list:
                cam_id = 0
                for c in camera_list:

                    rotation, translation = extract_rotation_translation(extrinsics[c].unsqueeze(0))
                    intr, intr_inv = get_intrinsics(c, device_repo, 1)
                    intr = intr.cpu().numpy()
                    intr_inv = intr_inv.cpu().numpy()

                    camera_dict = {}
                    camera_dict['R'] = rotation.cpu().numpy()
                    # camera_dict['T'] = translation.cpu().numpy()
                    # camera_dict['fx'] = intr[0, 0]
                    # camera_dict['fy'] = intr[1, 1]
                    # camera_dict['cx'] = intr[0, 2]
                    # camera_dict['cy'] = intr[1, 2]
                    tempt_T = translation.squeeze().clone().cpu().numpy()
                    camera_dict['T'] = translation.squeeze().clone().cpu().numpy()
                    camera_dict['T'][0] = tempt_T[1]
                    camera_dict['T'][1] = tempt_T[0]
                    camera_dict['fx'] = intr[1, 1]
                    camera_dict['fy'] = intr[0, 0]
                    camera_dict['cx'] = intr[1, 2]
                    camera_dict['cy'] = intr[0, 2]
                    camera_dict['k'] = 0
                    camera_dict['p'] = 0


                    subdir = '%s/%s/%s/%s' % (s, a, "imageSequence", c)
                    # subdir = 's_%s_act_%s_subact_%s_ca_%s' % (s, a, sa, c)
                    # subdir = subdir_format.format(s, a, sa, c)

                    # basename = metadata.get_base_filename('S{:d}'.format(s), '{:d}'.format(a), '{:d}'.format(sa), metadata.camera_ids[c-1])
                    # annotname = basename + '.cdf'

                    # poses_world = numpy.load(os.path.join(subject_path, action, subject + "_" + action + "_global_3d.npy"))[frame_idxs][:, valid_joints]

                    # subject = s
                    # annofile3d = osp.join(h36m_root, subject, 'Poses_D3_Positions_mono_universal', annotname)
                    # annofile3d_camera = osp.join('extracted', subject, 'Poses_D3_Positions_mono', annotname)
                    # annofile2d = osp.join('extracted', subject, 'Poses_D2_Positions', annotname)

                    # with pycdf.CDF(annofile3d) as data:
                    #     pose3d = np.array(data['Pose'])
                    #     pose3d = np.reshape(pose3d, (-1, 32, 3))

                    pose3d = numpy.load(os.path.join(h36m_root, s, a, s + "_" + a + "_global_3d.npy"))
                    pose3d_camera = numpy.load(os.path.join(h36m_root, s, a, s + "_" + a + "_" + c + "_3d.npy"))
                    pose2d = numpy.load(os.path.join(h36m_root, s, a, s + "_" + a + "_" + c + "_2d.npy"))

                    
                    # with pycdf.CDF(annofile3d_camera) as data:
                    #     pose3d_camera = np.array(data['Pose'])
                    #     pose3d_camera = np.reshape(pose3d_camera, (-1, 32, 3))

                    # with pycdf.CDF(annofile2d) as data:
                    #     pose2d = np.array(data['Pose'])
                    #     pose2d = np.reshape(pose2d, (-1, 32, 2))

                    nposes = min(pose3d.shape[0], pose2d.shape[0])
                    image_format = '{:d}.png'

                    r = R.from_euler('z', 90, degrees=True)

                    for i in range(nposes):
                        datum = {}
                        imagename = image_format.format(i)
                        imagepath = osp.join(h36m_root, subdir, imagename)
                        if not osp.isfile(imagepath):
                            print(imagepath)
                            print(nposes)
                        if osp.isfile(imagepath):
                            if (False):
                                img = cv2.imread(imagepath)
                                if (img.shape[0] < img.shape[1]):
                                    img = cv2.rotate(img, rotateCode=cv2.ROTATE_90_CLOCKWISE)
                                    cv2.imwrite(imagepath, img)
                            datum['image'] = imagepath
                            uv = pose2d[i, 0, joint_idx, :] # dim 1 number of people
                            uv[:,[0, 1]] = uv[:,[1, 0]]
                            uv[:, 0] = 720 - uv[:, 0]
                            datum['joints_2d'] = uv
                            datum['joints_3d'] = r.apply(pose3d[i, 0, joint_idx, :]) # dim 1 number of people
                            datum['joints_3d_camera'] = r.apply(pose3d_camera[i, 0, joint_idx, :]) # dim 1 number of people
                            datum['joints_vis'] = np.ones((17, 3))
                            datum['video_id'] = cnt
                            datum['image_id'] = i
                            print(datum['image_id'])
                            datum['subject'] = s
                            datum['action'] = a
                            datum['subaction'] = sa
                            datum['camera_id'] = cam_id
                            datum['source'] = 'h4d'
                            datum['camera'] = camera_dict

                            # box = _infer_box(datum['joints_3d_camera'], camera_dict, 0)
                            # center = (0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3]))
                            # scale = ((box[2] - box[0]) / 200.0, (box[3] - box[1]) / 200.0)
                            # datum['center'] = center
                            # datum['scale'] = scale
                            # datum['box'] = box

                            box = [0,0,1280,1280]
                            center = (0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3]))
                            scale = ((box[2] - box[0]) / 220.0, (box[3] - box[1]) / 220.0)
                            datum['center'] = center
                            datum['scale'] = scale
                            datum['box'] = box

                            if s in train_list:
                                train_db.append(datum)
                            else:
                                test_db.append(datum)

                    cnt += 1
                    cam_id += 1

    with open('h4d_train.pkl', 'wb') as f:
        pickle.dump(train_db, f)

    with open('h4d_validation.pkl', 'wb') as f:
        pickle.dump(test_db, f)






