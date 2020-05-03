#!/usr/bin/env python3

from os import path, makedirs, listdir
import os
from shutil import move
import numpy as np
import h5py
from subprocess import call
from tempfile import TemporaryDirectory
from tqdm import tqdm

from metadata import load_h4d_metadata

import cv2
from importers import *

import argparse

metadata = load_h4d_metadata()

# Subjects to include when preprocessing
included_subjects = {
    # 'S1': 1,
    # 'S2': 2,
    'S3': 3,
    # 'S4': 4,
    # 'S1S2': 5,
    # 'S3S4': 6
}

# Cameras to include when preprocessing
included_cameras = {
    'M72e': 1,
    'M72h': 2,
    'M72i': 3,
    'M72j': 4
}

# Actions to include when preprocessing
included_actions = {
    'running': 1,
    # 'junping_jack': 2,
    # 'bending': 3,
    # 'punching_n_kicking': 4,
    # 'basketball_dribbling': 5,
    # 'laying_down': 6,
    # 'sitting_down': 7,
    # 'sitting_on_a_chair': 8,
    # 'talking': 9,
    # 'object_dropping_n_picking': 10,
    # 'stretching_n_talking': 11,
    # 'talking_n_walking': 12,
    # 'watching_scary_movie': 13,
    # 'in-flight_safety_announcement': 14,
    # 'watching_football_together': 15,
    # 'dancing_together': 16,
    # 'physical_examination': 17,
    # 'whispering': 18,
    # 'card_trick': 19
}

# Sequences with known issues
blacklist = {
    ('S11', '100', '100', '10'),  # Video file is corrupted
}

parser = argparse.ArgumentParser(description="PyTorch Human4D Processor")
parser.add_argument(
        "--dataset_path",
        default="E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings",      
        # default="G:/MULTI4D_Dataset/multi/rgbd_subjects1and2/19-07-12-13-05-08",      
        help="path to sequence files",
    )

args = parser.parse_args()


# Rather than include every frame from every video, we can instead wait for the pose to change
# significantly before storing a new example.
def select_frame_indices_to_include(subject, poses_3d_univ):
    # To process every single frame, uncomment the following line:
    # return np.arange(0, len(poses_3d_univ))

    # Take every 64th frame for the protocol #2 test subjects
    # (see the "Compositional Human Pose Regression" paper)
    if subject == 'S9' or subject == 'S11':
        return np.arange(0, len(poses_3d_univ), 64)

    # Take only frames where movement has occurred for the protocol #2 train subjects
    frame_indices = []
    prev_joints3d = None
    threshold = 40 ** 2  # Skip frames until at least one joint has moved by 40mm
    for i, joints3d in enumerate(poses_3d_univ):
        if prev_joints3d is not None:
            max_move = ((joints3d - prev_joints3d) ** 2).sum(axis=-1).max()
            if max_move < threshold:
                continue
        prev_joints3d = joints3d
        frame_indices.append(i)
    return np.array(frame_indices)


# def infer_camera_intrinsics(points2d, points3d):
#     """Infer camera instrinsics from 2D<->3D point correspondences."""
#     pose2d = points2d.reshape(-1, 2)
#     pose3d = points3d.reshape(-1, 3)
#     x3d = np.stack([pose3d[:, 0], pose3d[:, 2]], axis=-1)
#     x2d = (pose2d[:, 0] * pose3d[:, 2])
#     alpha_x, x_0 = list(np.linalg.lstsq(x3d, x2d, rcond=-1)[0].flatten())
#     y3d = np.stack([pose3d[:, 1], pose3d[:, 2]], axis=-1)
#     y2d = (pose2d[:, 1] * pose3d[:, 2])
#     alpha_y, y_0 = list(np.linalg.lstsq(y3d, y2d, rcond=-1)[0].flatten())
#     return np.array([alpha_x, x_0, alpha_y, y_0])


def process_view(out_dir, subject, action, camera):
    subj_dir = path.join(args.dataset_path, subject, action)

    base_filename = subject + "_" + action

    # Load joint position annotations
    poses_2d = numpy.load(os.path.join(subj_dir, base_filename + "_" + camera + "_2d.npy"))
    poses_3d_univ = numpy.load(os.path.join(subj_dir, base_filename + "_" + camera + "_3d.npy"))
    poses_3d = numpy.load(os.path.join(subj_dir, base_filename + "_" + camera + "_3d.npy"))
   
    device_repo_path = os.path.join(args.dataset_path, "device_repository.json")
    device_repo = load_intrinsics_repository(os.path.join(device_repo_path))

    intr, intr_inv = get_intrinsics(camera, device_repo, 1)
    
    camera_int = intr.cpu().numpy()
    camera_int_univ = intr.cpu().numpy()

    frame_indices = select_frame_indices_to_include(subject, poses_3d_univ)
    frames = frame_indices + 1
    # video_file = path.join(subj_dir, 'Videos', base_filename + '.mp4')
    # frames_dir = path.join(out_dir, 'imageSequence', camera)
    # makedirs(frames_dir, exist_ok=True)

    # # Check to see whether the frame images have already been extracted previously
    # existing_files = {f for f in listdir(frames_dir)}
    # frames_are_extracted = True
    # for i in frames:
    #     filename = 'img_%06d.jpg' % i
    #     img = cv2.imread(path.join(frames_dir, filename))

    #     for p in poses_2d[i]:
    #         img = cv2.drawMarker(img, 
    #             (int(p[0]), int(p[1])), 
    #             (0, 255, 0),
    #             markerType=cv2.MARKER_SQUARE,
    #             markerSize=15,
    #             thickness=2)

    #     for p in poses_3d[i]:
    #         img = cv2.drawMarker(img, 
    #             (int(p[0]), int(p[1])), 
    #             (255, 55, 0),
    #             markerType=cv2.MARKER_CROSS,
    #             markerSize=15,
    #             thickness=2)

    #     cv2.imshow("test", img)
    #     cv2.waitKey()

    #     if filename not in existing_files:
    #         frames_are_extracted = False
    #         break

    

    # if not frames_are_extracted:
    #     with TemporaryDirectory() as tmp_dir:
    #         # Use ffmpeg to extract frames into a temporary directory
    #         call([
    #             'ffmpeg',
    #             '-nostats', '-loglevel', '0',
    #             '-i', video_file,
    #             '-qscale:v', '3',
    #             path.join(tmp_dir, 'img_%06d.jpg')
    #         ])

    #         # Move included frame images into the output directory
    #         for i in frames:
    #             filename = 'img_%06d.jpg' % i
    #             move(
    #                 path.join(tmp_dir, filename),
    #                 path.join(frames_dir, filename)
    #             )



    return {
        'pose/2d': poses_2d[frame_indices],
        'pose/3d-univ': poses_3d_univ[frame_indices],
        'pose/3d': poses_3d[frame_indices],
        'intrinsics/' + camera: camera_int,
        'intrinsics-univ/' + camera: camera_int_univ,
        'frame': frames,
        'camera': np.full(frames.shape, int(included_cameras[camera])),
        'subject': np.full(frames.shape, int(included_cameras[camera])),
        'action': np.full(frames.shape, int(included_actions[action])),
        'subaction': np.full(frames.shape, 100000000),
    }


def process_action(subject, action):
    datasets = {}

    out_dir = path.join('processed', subject, action)
    makedirs(out_dir, exist_ok=True)

    for camera in included_cameras:
        if (subject, action, camera) in blacklist:
            continue

        try:
            annots = process_view(out_dir, subject, action, camera)
        except:
            print('Error processing sequence, skipping: ', repr((subject, action, camera)))
            

        for k, v in annots.items():
            if k in datasets:
                datasets[k].append(v)
            else:
                datasets[k] = [v]

    if len(datasets) == 0:
        return

    datasets = {k: np.concatenate(v) for k, v in datasets.items()}

    with h5py.File(path.join(out_dir, 'annot.h5'), 'w') as f:
        for name, data in datasets.items():
            f.create_dataset(name, data=data)


def process_all():
    sequence_mappings = metadata.sequence_mappings

    actions = []

    for subject in included_subjects.keys():
        actions += [
            (subject, action)
            for action in included_actions
        ]

    for subject, action in tqdm(actions, ascii=True, leave=False):
        process_action(subject, action)

if __name__ == '__main__':
  process_all()
