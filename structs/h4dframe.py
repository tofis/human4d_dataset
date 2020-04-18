import os

class H4DFrame:
    def __init__(self, groupframe_id, frame_id, color_img, depth_img, timestamp, depth_thres=3000):
        self.groupframe_id = int(groupframe_id)
        self.frame_id = int(frame_id)
        self.color_img = color_img
        self.depth_img = depth_img
        self.depth_img[depth_img > 3000] = 0
        self.timestamp = timestamp