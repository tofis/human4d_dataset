"""
    Read bbox *.mat files from Human3.6M and convert them to a single *.npy file.
    Example of an original bbox file:
    <path-to-Human3.6M-root>/S1/MySegmentsMat/ground_truth_bb/WalkingDog 1.54138969.mat

    Usage:
    python3 collect-bboxes.py <path-to-Human3.6M-root> <num-processes>
"""
import os, sys
import numpy
import h5py

# dataset_root = sys.argv[1]
# dataset_root = "E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data"
# data_path = os.path.join(dataset_root, "Recordings/experimentation_dataset")

data_path = 'G:/MULTI4D_Dataset/HUMAN4D'
# subjects = [x for x in os.listdir(data_path) if x.startswith('S')]
subjects = ['S1', 'S2', 'S3', 'S4']

with open(os.path.join(data_path, 'metadata_single.txt')) as metadata_file:
    lines = metadata_file.readlines()
    metadata = {}
    for s in subjects:
        metadata[s] = {}

    for line in lines:
        values = line.split('\t')
        metadata[values[0]][values[1]] = values[2].strip()
# assert len(subjects) == 7
assert len(subjects) == 4

destination_dir = os.path.join(data_path, "BBs")
os.makedirs(destination_dir, exist_ok=True)
destination_file_path = os.path.join(destination_dir, "bboxes-H4D-GT.npy")

# Some bbox files do not exist, can be misaligned, damaged etc.
# from h4d_action_to_bbox_filename import action_to_bbox_filename

from collections import defaultdict
nesteddict = lambda: defaultdict(nesteddict)

bboxes_retval = nesteddict()

def load_bboxes(data_path, subject, action, camera):
    print(subject, action, camera)
    
    if ('!' in metadata[subject][action]):
        return
    # def mask_to_bbox(mask):
    #     h_mask = mask.max(0)
    #     w_mask = mask.max(1)

    #     top = h_mask.argmax()
    #     bottom = len(h_mask) - h_mask[::-1].argmax()

    #     left = w_mask.argmax()
    #     right = len(w_mask) - w_mask[::-1].argmax()

    #     return top, left, bottom, right

    try:
        # try:
        #     corrected_action = action_to_bbox_filename[subject][action]
        # except KeyError:
        #     corrected_action = action.replace('-', ' ')

        # TODO use pathlib
        # bboxes_path = os.path.join(
        #     data_path,
        #     subject,
        #     metadata[subject][action],
        #     '%s_%s_%s_bbox.npy' % (subject, action, camera))
        folder_files = [_file for _file in os.listdir(os.path.join(
            data_path,
            subject,
            metadata[subject][action]))
            if 'bbox' in _file and camera in _file]
        bbfile = folder_files[0]

        bboxes_path = os.path.join(
            data_path,
            subject,
            metadata[subject][action],
            bbfile)

        retval = numpy.load(bboxes_path)
        retval = retval[:, 0, :] # TODO: person id
            
        print("ok")
            # retval[]
        # with h5py.File(bboxes_path, 'r') as h5file:
        #     retval = np.empty((len(h5file['Masks']), 4), dtype=np.int32)

        for frame_idx in range(len(retval)):            
            top, left, bottom, right = retval[frame_idx]
        # if right-left < 2 or bottom-top < 2:
        #     raise Exception(str(bboxes_path) + ' $ ' + str(frame_idx))
    except Exception as ex:
        # reraise with path information
        raise Exception(str(ex) + '; %s %s %s' % (subject, action, camera))
    
    return retval, subject, action, camera

# retval['S1']['Talking-1']['54534623'].shape = (n_frames, 4) # top, left, bottom, right
def add_result_to_retval(args):
    bboxes, subject, action, camera = args
    bboxes_retval[subject][action][camera] = bboxes

def freeze_defaultdict(x):
    x.default_factory = None
    for value in x.values():
        if type(value) is defaultdict:
            freeze_defaultdict(value)

import multiprocessing


if __name__ == '__main__':
    num_processes = int(1)
    pool = multiprocessing.Pool(num_processes)
    async_errors = []

    for subject in subjects:
        subject_path = os.path.join(data_path, subject)
        # actions = [action for action in os.listdir(subject_path) if "." not in action]
        actions = [ 'running',
        'junping_jack',
        'bending',
        'punching_n_kicking',
        'basketball_dribbling',
        'laying_down',
        'sitting_down',
        'sitting_on_a_chair',
        'talking',
        'object_dropping_n_picking',
        'stretching_n_talking',
        'talking_n_walking',
        'watching_scary_movie',
        'in-flight_safety_announcement']
        try:
            actions.remove('MySegmentsMat') # folder with bbox *.mat files
        except ValueError:
            pass

        for action in actions:
            cameras = 'M72e', 'M72h', 'M72i', 'M72j'
            if action not in metadata[subject].keys():
                continue
            for camera in cameras:
                async_result = pool.apply_async(
                    load_bboxes,
                    args=(data_path, subject, action, camera),
                    callback=add_result_to_retval)
                async_errors.append(async_result)

    pool.close()
    pool.join()

    # raise any exceptions from pool's processes
    for async_result in async_errors:
        async_result.get()

    # convert to normal dict
    freeze_defaultdict(bboxes_retval)
    numpy.save(destination_file_path, bboxes_retval)
