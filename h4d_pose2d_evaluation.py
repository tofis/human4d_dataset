import numpy
import os
import argparse
import csv
from utils import *
from importers import *


# //     {0,  "Nose"},
# //     {1,  "Neck"},
# //     {2,  "RShoulder"},
# //     {3,  "RElbow"},
# //     {4,  "RWrist"},
# //     {5,  "LShoulder"},
# //     {6,  "LElbow"},
# //     {7,  "LWrist"},
# //     {8,  "MidHip"},
# //     {9,  "RHip"},
# //     {10, "RKnee"},
# //     {11, "RAnkle"},
# //     {12, "LHip"},
# //     {13, "LKnee"},
# //     {14, "LAnkle"},
# //     {15, "REye"},
# //     {16, "LEye"},
# //     {17, "REar"},
# //     {18, "LEar"},
# //     {19, "LBigToe"},
# //     {20, "LSmallToe"},
# //     {21, "LHeel"},
# //     {22, "RBigToe"},
# //     {23, "RSmallToe"},
# //     {24, "RHeel"},
# //     {25, "Background"}
openpose2gt = [
8,      # 0	Hips
-1,     # 1	Spine
-1,     # 2	Spine1
-1,     # 3	Spine2
-1,     # 4	Spine3
1,      # 5	Neck
-1,     # 6	Neck1
-1,     # 7	Head
-1,     # 8	HeadEnd
-1,      # 9	RightShoulder
2,      # 10	RightArm
3,      # 11	RightForeArm
4,      # 12	RightHand
-1,     # 13	RightHandThumb1
-1,     # 14	RightHandMiddle1
-1,     # 15	LeftShoulder
5,      # 16	LeftArm
6,      # 17	LeftForeArm
7,      # 18	LeftHand
-1,     # 19	LeftHandThumb1
-1,     # 20	LeftHandMiddle1
9,      # 21	RightUpLeg
10,     # 22	RightLeg
11,     # 23	RightFoot
-1,     # 24	RightForeFoot
-1,     # 25	RightToeBase
-1,     # 26	RightToeBaseEnd
12,     # 27	LeftUpLeg
13,     # 28	LeftLeg
14,     # 29	LeftFoot
-1,     # 30	LeftForeFoot
-1,     # 31	LeftToeBase
-1      # 32	LeftToeBaseEnd
]


def calculate_dist(kpt_gt, kpt_pred):
    return numpy.linalg.norm(numpy.subtract(kpt_gt, kpt_pred))

def main():
    parser = argparse.ArgumentParser(description="PyTorch Show Pose and Pointcloud")    
    parser.add_argument(
        "--dataset_path",
        default="G:/MULTI4D_Dataset/Human4D/S3",      
        help="path to gt npy sequence files",
    )
    # parser.add_argument(
    #     "--pred_path",
    #     default="G:/MULTI4D_Dataset/core/Subject3",      
    #     help="path to gt npy sequence files",
    # )
    parser.add_argument(
        "--cameras",
        default=["M72e", "M72h", "M72i", "M72j"]
    )
    parser.add_argument(
        "--show_data",
        default=True
    )
    parser.add_argument(
        "--test_set_ids",
        default="G:/MULTI4D_Dataset/single_person/single_files.txt"
    )
    args = parser.parse_args()


    test_set = {}
    if (args.test_set_ids != ""):        
        f = open(args.test_set_ids, 'r')
        lines = f.readlines()
        for line in lines:
            values = line.split(' ')
            seq_key = values[1].split('\\')[2]
            if seq_key not in test_set.keys():
                test_set[seq_key] = []
            test_set[seq_key].append(int(values[0]))


    # gt_folders = os.listdir(args.gt_path)
    # pred_folders = os.listdir(args.pred_path)
    folders = [folder for folder in os.listdir(args.dataset_path) if '-' in folder]

    for i in range(len(folders)):
        offset = load_rgbd_skip(os.path.join(args.dataset_path, "offsets.txt"), folders[i])
        npy_files_gt = [npy for npy in os.listdir(os.path.join(args.dataset_path, folders[i])) if '_2d' in npy]
        npy_bb_files_gt = [npy for npy in os.listdir(os.path.join(args.dataset_path, folders[i])) if '_bbox' in npy]
        npy_files_pred = {}
        for cam in args.cameras:
            npy_files_pred[cam] = [npy for npy in os.listdir(os.path.join(args.dataset_path, folders[i], 'Dump/openpose/color_joints_processed')) if cam in npy]
            sort_nicely(npy_files_pred[cam])
            npy_files_pred[cam] = npy_files_pred[cam][offset:]

        # npy_files_pred = [npy for npy in os.listdir(os.path.join(args.pred_path, pred_folders[i])) if '_2d' in npy]
        for cam in args.cameras:
            npy_cam_gt = [gtfile for gtfile in npy_files_gt if cam in gtfile].pop(0)
            npy_bb_cam_gt = [gtfile for gtfile in npy_bb_files_gt if cam in gtfile].pop(0)
            # npy_cam_pred = [predfile for predfile in npy_files_pred if cam in predfile].pop(0)

            gt = numpy.load(os.path.join(args.dataset_path, folders[i], npy_cam_gt))
            bb_gt = numpy.load(os.path.join(args.dataset_path, folders[i], npy_bb_cam_gt))
            pred = numpy.zeros([len(npy_files_pred[cam]), 1, 25, 2])

            assert gt.shape[0] == pred.shape[0]

            for fid in range(pred.shape[0]):
                npy = numpy.load(os.path.join(args.dataset_path, folders[i], 'Dump/openpose/color_joints_processed', npy_files_pred[cam][fid]))
                if (npy.shape[0] == pred[fid].shape[0]):
                    pred[fid] = npy[:,:,2:]
                else:
                    pred[fid, :, :, :] = -10000

            # pred = numpy.load(os.path.join(args.pred_path, pred_folders[i], npy_cam_pred))
            
            # abs_error = numpy.zeros([gt.shape[0], gt.shape[1] * gt.shape[2]])
            # squared_error = numpy.zeros([gt.shape[0], gt.shape[1] * gt.shape[2]])
            abs_error = numpy.zeros([len(test_set[folders[i]]), 1 + pred.shape[1] * pred.shape[2]])
            squared_error = numpy.zeros([len(test_set[folders[i]]), 1 + pred.shape[1] * pred.shape[2]])
            pck_error = numpy.zeros([len(test_set[folders[i]]), 1 + pred.shape[1] * pred.shape[2]])


            # we should evaluate only the test dataset fids
            # for fid in range(pred.shape[0]):
            for k in range(len(test_set[folders[i]])):
                # for fid in test_set[folders[i]]:
                fid = test_set[folders[i]][k]
                abs_error[k, 0] = fid
                for p in range(pred.shape[1]):
                    if (args.show_data):
                        img = numpy.zeros([720, 1280, 3])
                        # img = cv2.rectangle(img, (bb_gt[fid, 0, 0], bb_gt[fid, 0, 1]), (bb_gt[fid, 0, 2], bb_gt[fid, 0, 3]), (255, 0, 0))
                        img = cv2.rectangle(img, (bb_gt[fid, 0, 0], bb_gt[fid, 0, 1]), \
                            (bb_gt[fid, 0, 2], bb_gt[fid, 0, 3]), (255, 0, 0))                        
                        
                    for j in range(pred.shape[2]):
                        if (openpose2gt[j] != -1):
                            # if (args.show_data):
                            #     img = cv2.drawMarker(img, (int(gt[fid, p, j, 0]), int(gt[fid, p, j, 1])), (0, 255, 0), markerType=cv2.MARKER_CROSS)
                            #     img = cv2.drawMarker(img, (int(pred[fid, p, openpose2gt[j], 0]), int(pred[fid, p, openpose2gt[j], 1])), (0, 0, 255), markerType=cv2.MARKER_SQUARE)
                            # abs_error[i, (p+1) * j] = calculate_dist(gt[fid, p, j], pred[fid, p, openpose2gt[j]])
                            if (args.show_data):
                                img = cv2.drawMarker(img, (int(gt[fid, p, j, 0]), int(gt[fid, p, j, 1])), (0, 255, 0), markerType=cv2.MARKER_CROSS)
                                img = cv2.drawMarker(img, (int(pred[fid, p, openpose2gt[j], 0]), int(pred[fid, p, openpose2gt[j], 1])), (0, 0, 255), markerType=cv2.MARKER_SQUARE)
                            error = calculate_dist(gt[fid, p, j], pred[fid, p, openpose2gt[j]])
                            abs_error[k, 1 + (p+1) * j] = error
                            squared_error[k, 1 + (p+1) * j] = error * error
                            max_edge = max(bb_gt[fid, 0, 2]-bb_gt[fid, 0, 0], bb_gt[fid, 0, 3]-bb_gt[fid, 0, 1])
                            print(folders[i] + ' ' + str(max_edge))
                            pck_error[k, 1 + (p+1) * j] = 1 if (error / max_edge) < 0.1 else 0
                        else:
                            abs_error[k, 1 + (p+1) * j] = -5000
                            squared_error[k, 1 + (p+1) * j] = -5000
                            pck_error[k, 1 + (p+1) * j] = -5000

                    if (args.show_data):                   
                        cv2.imshow('img', cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))
                        cv2.waitKey(10)

            out_folder = os.path.join(args.dataset_path, folders[i], 'Dump/openpose/results')
            if not os.path.exists(out_folder):
                os.makedirs(out_folder)
            numpy.savetxt(os.path.join(out_folder, cam + "_abs.csv"), abs_error, delimiter=',')
            numpy.savetxt(os.path.join(out_folder, cam + "_squared.csv"), squared_error, delimiter=',')
            numpy.savetxt(os.path.join(out_folder, cam + "_pck.csv"), pck_error, delimiter=',')

if __name__ == "__main__":
    main()
