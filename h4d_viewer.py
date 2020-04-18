import os
import numpy
import argparse

from structs import *
from visualization import *

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
    parser.add_argument(
        "--device_list",     
        nargs="*",
        default = ["M72i", "M72j", "M72e", "M72h"],
        help="list of devices/cameras",
    )

    args = parser.parse_args()
    COLORS = get_COLORS()

    poses2d = {}
    poses3d = {}
    for view in args.device_list:    
        poses2d[view] = numpy.load(os.path.join(args.sequence_path, args.sequence_filename + "_" + view + "_2d.npy"))
        poses3d[view] = numpy.load(os.path.join(args.sequence_path, args.sequence_filename + "_" + view + "_3d.npy"))

    h4d_seq = H4DSequence(os.path.join(args.sequence_path, "Dump"), args.device_list)

    for i in range(h4d_seq.num_of_frames):
        for view in h4d_seq.camera_ids:
            img = h4d_seq.cameras[view][i].color_img
            for uv in poses2d[view][i]:
                img = cv2.drawMarker(img, 
                                        (int(uv[0]), int(uv[1])), 
                                        (200, 50, 150),
                                        markerType=cv2.MARKER_STAR,
                                        markerSize=10,
                                        thickness=2)
            cv2.imshow(view + "_" + str(i), numpy.transpose(img, (1, 0, 2)))
            draw_skeleton_joints(img, poses2d[view][i], COLORS)

            cv2.waitKey()


    print("end.-")


if __name__ == "__main__":
    main()