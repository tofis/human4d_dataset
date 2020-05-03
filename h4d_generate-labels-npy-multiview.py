"""
    Generate 'labels.npy' for multiview 'human36m.py'
    from https://github.sec.samsung.net/RRU8-VIOLET/multi-view-net/

    Usage: `python3 generate-labels-npy-multiview.py <path/to/Human3.6M-root> <path/to/una-dinosauria-data/h36m> <path/to/bboxes-Human36M-squared.npy>`
"""
import os, sys
import numpy as np
import h5py

import torch

from importers import *
from vision import *

# Change this line if you want to use Mask-RCNN or SSD bounding boxes instead of H36M's "ground truth".
BBOXES_SOURCE = 'GT' # or 'MRCNN' or 'SSD'

retval = {
    'subject_names': ['S3'], #, 'S2', 'S6', 'S7', 'S8', 'S9', 'S11'],
    'camera_names': ['M72e', 'M72h', 'M72i', 'M72j'],
    'action_names': [
        'running'
    ]
        # 'Discussion-1', 'Discussion-2',
        # 'Eating-1', 'Eating-2',
        # 'Greeting-1', 'Greeting-2',
        # 'Phoning-1', 'Phoning-2',
        # 'Posing-1', 'Posing-2',
        # 'Purchases-1', 'Purchases-2',
        # 'Sitting-1', 'Sitting-2',
        # 'SittingDown-1', 'SittingDown-2',
        # 'Smoking-1', 'Smoking-2',
        # 'TakingPhoto-1', 'TakingPhoto-2',
        # 'Waiting-1', 'Waiting-2',
        # 'Walking-1', 'Walking-2',
        # 'WalkingDog-1', 'WalkingDog-2',
        # 'WalkingTogether-1', 'WalkingTogether-2']
}
retval['cameras'] = np.empty(
    (len(retval['subject_names']), len(retval['camera_names'])),
    dtype=[
        ('R', np.float32, (3,3)),
        ('t', np.float32, (3,1)),
        ('K', np.float32, (3,3)),
        ('dist', np.float32, 5)
    ]
)

table_dtype = np.dtype([
    ('subject_idx', np.int8),
    ('action_idx', np.int8),
    ('frame_idx', np.int16),
    ('keypoints', np.float32, (17,3)), # roughly MPII format
    ('bbox_by_camera_tlbr', np.int16, (len(retval['camera_names']),4))
])
retval['table'] = []

# h36m_root = sys.argv[1]
h36m_root = "E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings"

# destination_file_path = os.path.join(h36m_root, "extra", f"human36m-multiview-labels-{BBOXES_SOURCE}bboxes.npy")
destination_file_path = os.path.join("E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings", f"human4d-multiview-labels-{BBOXES_SOURCE}bboxes.npy")
# destination_file_path = "E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/BBs/bboxes-H4D-GT.npy"

# una_dinosauria_root = sys.argv[2] 
una_dinosauria_root = "E:/VCL/Users/tofis/Data/DATASETS/h36m-fetch/extra/una-dinosauria-data/h36m"
cameras_params = h5py.File(os.path.join(una_dinosauria_root, 'cameras.h5'), 'r')

# Fill retval['cameras']
for subject_idx, subject in enumerate(retval['subject_names']):
    for camera_idx, camera in enumerate(retval['camera_names']):
        # assert len(cameras_params[subject.replace('S', 'subject')]) == 4
        device_repo_path = os.path.join(h36m_root, "device_repository.json")
        device_repo = load_intrinsics_repository(os.path.join(device_repo_path))

        intr, intr_inv = get_intrinsics(camera, device_repo, 1)
        
        camera_int = intr.cpu().numpy()
        camera_int_univ = intr.cpu().numpy()

        extr_files = [current_ for current_ in os.listdir(os.path.join(h36m_root, subject)) if ".extrinsics" in current_]

        extrinsics = {}
        paths = {}
        views = []

        for extr in extr_files:
            extrinsics[extr.split(".")[0]] = load_extrinsics(os.path.join(h36m_root, subject, extr))[0]
            # paths[extr.split(".")[0]] = os.path.join(args.sequence_path, extr.split(".")[0])
            views.append(extr.split(".")[0])
        rotation, translation = extract_rotation_translation(extrinsics[camera].unsqueeze(0))
        
        # camera_params = cameras_params[subject.replace('S', 'subject')]['camera%d' % (camera_idx+1)]
        camera_retval = retval['cameras'][subject_idx][camera_idx]
        
        # def camera_array_to_name(array):
        #     return ''.join(chr(int(x[0])) for x in array)
        # assert camera_array_to_name(camera_params['Name']) == camera

        camera_retval['R'] = torch.inverse(rotation).cpu().numpy()
        camera_retval['t'] = (-torch.inverse(rotation) @ translation).cpu().numpy()

        camera_retval['K'] = intr.cpu().numpy()
        # camera_retval['K'][:2, 2] = camera_params['c'][:, 0]
        # camera_retval['K'][0, 0] = camera_params['f'][0]
        # camera_retval['K'][1, 1] = camera_params['f'][1]
        # camera_retval['K'][2, 2] = 1.0

        # camera_retval['dist'][:2] = camera_params['k'][:2, 0]
        # camera_retval['dist'][2:4] = camera_params['p'][:, 0]
        # camera_retval['dist'][4] = camera_params['k'][2, 0]
        camera_retval['dist'][:2] = 0
        camera_retval['dist'][2:4] = 0
        camera_retval['dist'][4] = 0

# Fill bounding boxes
# bboxes = np.load(sys.argv[3], allow_pickle=True).item()
bboxes = np.load("E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/BBs/bboxes-H4D-GT.npy", allow_pickle=True).item()

def square_the_bbox(bbox):
    top, left, bottom, right = bbox
    width = right - left
    height = bottom - top

    if height < width:
        center = (top + bottom) * 0.5
        top = int(round(center - width * 0.5))
        bottom = top + width
    else:
        center = (left + right) * 0.5
        left = int(round(center - height * 0.5))
        right = left + height

    return top, left, bottom, right

for subject in bboxes.keys():
    for action in bboxes[subject].keys():
        for camera, bbox_array in bboxes[subject][action].items():
            for frame_idx, bbox in enumerate(bbox_array):
                bbox[:] = square_the_bbox(bbox)

if BBOXES_SOURCE is not 'GT':
    def replace_gt_bboxes_with_cnn(bboxes_gt, bboxes_detected_path, detections_file_list):
        """
            Replace ground truth bounding boxes with boxes from a CNN detector.
        """
        with open(bboxes_detected_path, 'r') as f:
            import json
            bboxes_detected = json.load(f)

        with open(detections_file_list, 'r') as f:
            for bbox, filename in zip(bboxes_detected, f):
                # parse filename
                filename = filename.strip()
                filename, frame_idx = filename[:-15], int(filename[-10:-4])-1
                filename, camera_name = filename[:-23], filename[-8:]
                slash_idx = filename.rfind('/')
                filename, action_name = filename[:slash_idx], filename[slash_idx+1:]
                subject_name = filename[filename.rfind('/')+1:]

                bbox, _ = bbox[:4], bbox[4] # throw confidence away
                bbox = square_the_bbox([bbox[1], bbox[0], bbox[3]+1, bbox[2]+1]) # LTRB to TLBR
                bboxes_gt[subject_name][action_name][camera_name][frame_idx] = bbox

    detections_paths = {
        'MRCNN': {
            'train': "/Vol1/dbstore/datasets/Human3.6M/extra/train_human36m_MRCNN.json",
            'test': "/Vol1/dbstore/datasets/Human3.6M/extra/test_human36m_MRCNN.json"
        },
        'SSD': {
            'train': "/Vol1/dbstore/datasets/k.iskakov/share/ssd-detections-train-human36m.json",
            'test': "/Vol1/dbstore/datasets/k.iskakov/share/ssd-detections-human36m.json"
        }
    }

    replace_gt_bboxes_with_cnn(
        bboxes,
        detections_paths[BBOXES_SOURCE]['train'],
        "/Vol1/dbstore/datasets/Human3.6M/train-images-list.txt")

    replace_gt_bboxes_with_cnn(
        bboxes,
        detections_paths[BBOXES_SOURCE]['test'],
        "/Vol1/dbstore/datasets/Human3.6M/test-images-list.txt")

# fill retval['table']
# from action_to_una_dinosauria import action_to_una_dinosauria

for subject_idx, subject in enumerate(retval['subject_names']):
    subject_path = os.path.join(h36m_root, subject)
    actions = os.listdir(subject_path)
    try:
        actions.remove('MySegmentsMat') # folder with bbox *.mat files
    except ValueError:
        pass

    for action_idx, action in enumerate(retval['action_names']):
        action_path = os.path.join(subject_path, action, 'imageSequence')
        if not os.path.isdir(action_path):
            raise FileNotFoundError(action_path)

        for camera_idx, camera in enumerate(retval['camera_names']):
            camera_path = os.path.join(action_path, camera)
            if os.path.isdir(camera_path):
                # frame_idxs = sorted([int(name.split('_')[0]) for name in os.listdir(camera_path)])
                frame_idxs = sorted([int(name.split('.')[0]) for name in os.listdir(camera_path)])
                assert len(frame_idxs) > 15, 'Too few frames in %s' % camera_path # otherwise WTF
                break
        else:
            raise FileNotFoundError(action_path)

        # 16 joints in MPII order + "Neck/Nose"
        # valid_joints = (3,2,1,6,7,8,0,12,13,15,27,26,25,17,18,19) + (14,)
        valid_joints = (23,22,21,27,28,29,0,4,6,8,12,11,10,16,17,18) + (7,)
        # with h5py.File(os.path.join(una_dinosauria_root, subject, 'MyPoses', '3D_positions',
        #                             '%s.h5' % action_to_una_dinosauria[subject].get(action, action.replace('-', ' '))), 'r') as poses_file:
        # poses_world = np.array(poses_file['3D_positions']).T.reshape(-1, 32, 3)[frame_idxs][:, valid_joints]
        # poses_world = numpy.load(os.path.join(subject_path, action, subject + "_" + action + "_" + camera + "_3d.npy"))[frame_idxs][:, valid_joints]
        poses_world = numpy.load(os.path.join(subject_path, action, subject + "_" + action + "_global_3d.npy"))[frame_idxs][:, valid_joints]

        table_segment = np.empty(len(frame_idxs), dtype=table_dtype)
        table_segment['subject_idx'] = subject_idx
        table_segment['action_idx'] = action_idx
        table_segment['frame_idx'] = frame_idxs
        table_segment['keypoints'] = poses_world
        table_segment['bbox_by_camera_tlbr'] = 0 # let a (0,0,0,0) bbox mean that this view is missing

        for (camera_idx, camera) in enumerate(retval['camera_names']):
            camera_path = os.path.join(action_path, camera)
            if not os.path.isdir(camera_path):
                print('Warning: camera %s isn\'t present in %s/%s' % (camera, subject, action))
                continue
            
            for bbox, frame_idx in zip(table_segment['bbox_by_camera_tlbr'], frame_idxs):
                bbox[camera_idx] = bboxes[subject][action][camera][frame_idx]

        retval['table'].append(table_segment)

retval['table'] = np.concatenate(retval['table'])
assert retval['table'].ndim == 1

print("Total frames in Human3.6Million:", len(retval['table']))
np.save(destination_file_path, retval)
