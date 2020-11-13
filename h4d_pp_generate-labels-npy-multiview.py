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
h36m_root = "E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/_ird_recordings"


        
retval = {
    # 'subject_names': ['S1', 'S2', 'S3', 'S4'], #, 'S2', 'S6', 'S7', 'S8', 'S9', 'S11'],
    'subject_names': ['S1', 'S2', 'S3', 'S4'], #, 'S2', 'S6', 'S7', 'S8', 'S9', 'S11'],
    'camera_names': ['M72e', 'M72h', 'M72i', 'M72j'],
    'action_names': [
        # 'running',
        # 'junping_jack',
        # 'bending',
        'punching_n_kicking',
        # 'basketball_dribbling',
        # 'laying_down',
        # 'sitting_down',
        'sitting_on_a_chair',
        # 'talking',
        # 'object_dropping_n_picking',
        'stretching_n_talking',
        # 'talking_n_walking',
        # 'watching_scary_movie',
        # 'in-flight_safety_announcement'
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

metadata = {}
test_set = []

# with open(os.path.join(h36m_root, 'metadata_single_ir.txt')) as metadata_file:
#     lines = metadata_file.readlines()
#     metadata = {}
#     for s in retval['subject_names']:
#         metadata[s] = {}

#     for line in lines:
#         values = line.split('\t')
#         s = values[0]
#         if s in metadata.keys():
#             metadata[values[0]][values[1]] = values[2].strip()

with open(os.path.join(h36m_root, 'metadata_single_ir.txt')) as metadata_file:
    lines = metadata_file.readlines()
    for s in retval['subject_names']:
        metadata[s] = {}
    subjects = []
    for line in lines:
        values = line.split('\t')
        if (values[1] in retval['action_names']):
            s = values[0]
            # if s in metadata.keys() and s not in subjects:
            #     subjects.append(s)
            if s in metadata.keys():
                path = os.path.join(h36m_root, values[0], values[2].strip())
                data_folders = [folder for folder in os.listdir(path) if '_data_final' in folder]
                if len(data_folders):
                    data_folder = data_folders[0]
                    metadata[values[0]][values[1]] = {}
                    metadata[values[0]][values[1]]['folder'] = values[2].strip()
                    metadata[values[0]][values[1]]['fidx'] = []
                    
                    all_txt_samples = [txt_file for txt_file in os.listdir(os.path.join(path, data_folder)) if 'gt' in txt_file]

                    for filetxt in all_txt_samples:
                        color_id = filetxt.split('_')[1]
                        color_filenames = [colorfile for colorfile in os.listdir(os.path.join(path, 'Dump/color')) if color_id + '_' in colorfile]
                        for color_filename in color_filenames:
                            full_filaname = os.path.join(path, 'Dump/color', color_filename) 
                            # print(full_filaname)
                            if len(color_id) > 2 and full_filaname not in test_set: # len(color_id) > 3
                                test_set.append(full_filaname)
                                if int(color_id) not in metadata[values[0]][values[1]]['fidx']:
                                    metadata[values[0]][values[1]]['fidx'].append(int(color_id))
                                
                                
                    # if len(metadata[values[0]][values[1]]['fidx']) > 20:
                    #     break
                # break

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

# destination_file_path = os.path.join(h36m_root, "extra", f"human36m-multiview-labels-{BBOXES_SOURCE}bboxes.npy")
destination_file_path = os.path.join(h36m_root, f"human4d-multiview-labels-{BBOXES_SOURCE}bboxes.npy")
# destination_file_path = "E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/BBs/bboxes-H4D-GT.npy"

# una_dinosauria_root = sys.argv[2] 
# una_dinosauria_root = "E:/VCL/Users/tofis/Data/DATASETS/h36m-fetch/extra/una-dinosauria-data/h36m"
# cameras_params = h5py.File(os.path.join(una_dinosauria_root, 'cameras.h5'), 'r')

# Fill retval['cameras']
for subject_idx, subject in enumerate(retval['subject_names']):
    for camera_idx, camera in enumerate(retval['camera_names']):
        # assert len(cameras_params[subject.replace('S', 'subject')]) == 4
        device_repo_path = os.path.join(h36m_root, "device_repository.json")
        device_repo = load_intrinsics_repository(os.path.join(device_repo_path))

        intr, intr_inv = get_intrinsics(camera, device_repo, 1)
        
        camera_int = intr.cpu().numpy()
        camera_int_univ = intr.cpu().numpy()

        extr_files = [current_ for current_ in os.listdir(os.path.join(h36m_root, subject, 'pose')) if ".extrinsics" in current_]

        extrinsics = {}
        paths = {}
        views = []

        for extr in extr_files:
            extrinsics[extr.split(".")[0]] = load_extrinsics(os.path.join(h36m_root, subject, 'pose', extr))[0]
            # paths[extr.split(".")[0]] = os.path.join(args.sequence_path, extr.split(".")[0])
            views.append(extr.split(".")[0])
        rotation, translation = extract_rotation_translation(extrinsics[camera].unsqueeze(0))
        
        # camera_params = cameras_params[subject.replace('S', 'subject')]['camera%d' % (camera_idx+1)]
        camera_retval = retval['cameras'][subject_idx][camera_idx]
        
        # def camera_array_to_name(array):
        #     return ''.join(chr(int(x[0])) for x in array)
        # assert camera_array_to_name(camera_params['Name']) == camera

        # camera_retval['R'] = torch.inverse(rotation).cpu().numpy()
        # camera_retval['t'] = (-torch.inverse(rotation) @ translation).cpu().numpy()
        camera_retval['R'] = rotation.cpu().numpy()
        camera_retval['t'] = translation.cpu().numpy()

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

npy_bb = os.path.join(h36m_root, 'BBs', 'bboxes-H4D-GT.npy')
bboxes = np.load(npy_bb, allow_pickle=True).item()

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
        if action not in metadata[subject].keys():
            continue

        rgbd_skip = load_rgbd_skip(os.path.join(subject_path, "offsets.txt"), metadata[subject][action]['folder'])

        action_path = os.path.join(subject_path, metadata[subject][action]['folder'], 'Dump', 'color')
        if not os.path.isdir(action_path):
            raise FileNotFoundError(action_path)

        for camera_idx, camera in enumerate(retval['camera_names']):
            # camera_path = os.path.join(action_path, camera)
            # if os.path.isdir(camera_path):
            #     # frame_idxs = sorted([int(name.split('_')[0]) for name in os.listdir(camera_path)])
            #     frame_idxs = sorted([int(name.split('.')[0]) for name in os.listdir(camera_path)])
            #     assert len(frame_idxs) > 15, 'Too few frames in %s' % camera_path # otherwise WTF
            #     break
            
            if os.path.isdir(action_path):

                # with open(os.path.join(h36m_root, 'metadata_single.txt')) as metadata_file:

                frame_idxs = [fid-rgbd_skip for fid in metadata[subject][action]['fidx']]
                # if rgbd_skip > 0:
                #     frame_idxs = sorted([int(name.split('.')[0].split('_')[0]) for name in os.listdir(action_path) if camera in name])[:-rgbd_skip]
                # else:
                #     frame_idxs = sorted([int(name.split('.')[0].split('_')[0]) for name in os.listdir(action_path) if camera in name])

                assert len(frame_idxs) > 15, 'Too few frames in %s' % camera_path # otherwise WTF
                break
        else:
            raise FileNotFoundError(action_path)

        # 16 joints in MPII order + "Neck/Nose"
        # valid_joints = (3,2,1,6,7,8,0,12,13,15,27,26,25,17,18,19) + (14,)
        valid_joints = (23,22,21,27,28,29,0,4,6,8,12,11,10,16,17,18) + (7,)

        # 21: RightUpLeg, 22: RightLeg, 23: RightFoot, 27: LeftUpLeg, 28: LeftLeg, 29: LeftFoot, 0: Hips, 4: Spine3, 6: Neck(spine),
        # 8: Neck, 12: RightWrist, 11: RightElbow, 10: RightShoulder, 16: LeftShoulder, 17: LeftElbow, 18: LeftWrist, 7: Head
# 0	Hips
# 1	Spine
# 2	Spine1
# 3	Spine2
# 4	Spine3
# 5	Neck
# 6	Neck1
# 7	Head
# 8	HeadEnd
# 9	RightShoulder
# 10	RightArm
# 11	RightForeArm
# 12	RightHand
# 13	RightHandThumb1
# 14	RightHandMiddle1
# 15	LeftShoulder
# 16	LeftArm
# 17	LeftForeArm
# 18	LeftHand
# 19	LeftHandThumb1
# 20	LeftHandMiddle1

# 24	RightForeFoot
# 25	RightToeBase
# 26	RightToeBaseEnd
# 27	LeftUpLeg
# 28	LeftLeg
# 29	LeftFoot
# 30	LeftForeFoot
# 31	LeftToeBase
# 32	LeftToeBaseEnd
        # with h5py.File(os.path.join(una_dinosauria_root, subject, 'MyPoses', '3D_positions',
        #                             '%s.h5' % action_to_una_dinosauria[subject].get(action, action.replace('-', ' '))), 'r') as poses_file:
        # poses_world = np.array(poses_file['3D_positions']).T.reshape(-1, 32, 3)[frame_idxs][:, valid_joints]
        # poses_world = numpy.load(os.path.join(subject_path, action, subject + "_" + action + "_" + camera + "_3d.npy"))[frame_idxs][:, valid_joints]
        folder_files = [_file for _file in os.listdir(os.path.join(
            subject_path,
            metadata[subject][action]['folder']))
            if '_global_3d.npy' in _file]
        bbfile = folder_files[0]

        # if (subject == 'S3'):
        #     poses_world = numpy.load(os.path.join(subject_path, metadata[subject][action], bbfile))[frame_idxs][:, 0, valid_joints, :] # 0 in dim 1 for person_id
        # else:
        npy_poses = numpy.load(os.path.join(subject_path, metadata[subject][action]['folder'], bbfile))
        frame_idxs_final = [fid for fid in frame_idxs if fid < len(npy_poses)]
        poses_world = npy_poses[frame_idxs_final][:, valid_joints, 0, :] # 0 in dim 1 for person_id

        table_segment = np.empty(len(frame_idxs_final), dtype=table_dtype)
        table_segment['subject_idx'] = subject_idx
        table_segment['action_idx'] = action_idx
        table_segment['frame_idx'] = frame_idxs_final
        table_segment['keypoints'] = poses_world
        table_segment['bbox_by_camera_tlbr'] = 0 # let a (0,0,0,0) bbox mean that this view is missing

        for (camera_idx, camera) in enumerate(retval['camera_names']):
            camera_path = os.path.join(action_path, camera)
            # if not os.path.isdir(camera_path):
            #     print('Warning: camera %s isn\'t present in %s/%s' % (camera, subject, action))
            #     continue
            
            for bbox, frame_idx in zip(table_segment['bbox_by_camera_tlbr'], frame_idxs_final):
                bbox[camera_idx] = bboxes[subject][action][camera][frame_idx]

        retval['table'].append(table_segment)

retval['table'] = np.concatenate(retval['table'])
assert retval['table'].ndim == 1

print("Total frames in Human3.6Million:", len(retval['table']))
np.save(destination_file_path, retval)
