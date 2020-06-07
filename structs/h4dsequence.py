import os
import cv2
from structs import H4DFrame
from importers import *
from utils import *

class H4DSequence:
    def __init__(self, sequence_path, camera_ids, skip=20, test_mode=False):
        self.camera_ids = camera_ids
        self.cameras = {}
        self.init_groupframe_id = -1
        self.init_timestamp = {}
        init_timestamp_set = False
        for cam in camera_ids:
            self.cameras[cam] = []
        
        color_images = os.listdir(os.path.join(sequence_path, "color"))
        depth_images = os.listdir(os.path.join(sequence_path, "depth"))
        timestamps = os.listdir(os.path.join(sequence_path, "timestamp"))

        sort_nicely(color_images)
        sort_nicely(depth_images)
        sort_nicely(timestamps)

        skip *= len(camera_ids)

        if (not test_mode):
            all_frames = len(color_images)
        else:
            all_frames = 800
        for i in range (skip, all_frames):
        # for i in range (100):
            groupframe_id, cam, _, frame_id = color_images[i].split('_')
            frame_id = frame_id.split('.')[0]

            color_img = cv2.imread(os.path.join(sequence_path, "color", color_images[i]))
            depth_img = readpgm(os.path.join(sequence_path, "depth", depth_images[i])) / 10

            if (init_timestamp_set and int(groupframe_id) > self.init_groupframe_id):
                timestamp = float(open(os.path.join(sequence_path, "timestamp", timestamps[i])).readline()) - self.init_timestamp[cam]        
            else:
                timestamp = float(open(os.path.join(sequence_path, "timestamp", timestamps[i])).readline())
                self.init_timestamp[cam] = timestamp
                timestamp -= self.init_timestamp[cam]
                init_timestamp_set = True
                self.init_groupframe_id = int(groupframe_id)


            
            h4d_frame = H4DFrame(groupframe_id, frame_id, color_img, depth_img, timestamp)
            self.cameras[cam].append(h4d_frame)
            # self.cameras[cam]
        
        self.num_of_frames = len(self.cameras[cam])

