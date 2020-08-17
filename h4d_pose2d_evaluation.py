import numpy
import os
import argparse
import csv
from utils import *
from importers import *


# {0:'head', 1:'neck', 2:'Rsho', 3:'Relb', 4:'Rwri', ...
#  5:'Lsho', 6:'Lelb', 7:'Lwri', ...
#  8:'Rhip', 9:'Rkne', 10:'Rank', ...
#  11:'Lhip', 12:'Lkne', 13:'Lank'
#  14: Spine, 15: Rfoot , 16:'Lfoot'}
gt2vnect = [
-1,     # 0	Hips
-1,     # 1	Spine
-1,     # 2	Spine1
-1,     # 3	Spine2
-1,     # 4	Spine3
-1,     # 5	Neck
-1,     # 6	Neck1
-1,     # 7	Head
-1,     # 8	HeadEnd
-1,     # 9	RightShoulder
 5,      # 10	RightArm
 6,      # 11	RightForeArm
 7,      # 12	RightHand
-1,     # 13	RightHandThumb1
-1,     # 14	RightHandMiddle1
-1,     # 15	LeftShoulder
 2,      # 16	LeftArm
 3,      # 17	LeftForeArm
 4,      # 18	LeftHand
-1,     # 19	LeftHandThumb1
-1,     # 20	LeftHandMiddle1
11,      # 21	RightUpLeg
12,     # 22	RightLeg
13,     # 23	RightFoot
-1,     # 24	RightForeFoot
-1,     # 25	RightToeBase
-1,     # 26	RightToeBaseEnd
 8,     # 27	LeftUpLeg
 9,     # 28	LeftLeg
10,     # 29	LeftFoot
-1,     # 30	LeftForeFoot
-1,     # 31	LeftToeBase
-1      # 32	LeftToeBaseEnd
]

# 0 "nose", 
# 1 "left_eye",
# 2 "right_eye",
# 3 "left_ear",
# 4 "right_ear",
# 5 "left_shoulder",
# 6 "right_shoulder",
# 7 "left_elbow",
# 8 "right_elbow",
# 9 "left_wrist",
# 10 "right_wrist",
# 11 "left_hip",
# 12 "right_hip",
# 13 "left_knee",
# 14 "right_knee",
# 15 "left_ankle",
# 16 "right_ankle" 
gt2alphapose = [
-1,     # 0	Hips
-1,     # 1	Spine
-1,     # 2	Spine1
-1,     # 3	Spine2
-1,     # 4	Spine3
-1,     # 5	Neck
-1,     # 6	Neck1
-1,     # 7	Head
-1,     # 8	HeadEnd
-1,     # 9	RightShoulder
6,      # 10	RightArm
8,      # 11	RightForeArm
10,      # 12	RightHand
-1,     # 13	RightHandThumb1
-1,     # 14	RightHandMiddle1
-1,     # 15	LeftShoulder
5,      # 16	LeftArm
7,      # 17	LeftForeArm
9,      # 18	LeftHand
-1,     # 19	LeftHandThumb1
-1,     # 20	LeftHandMiddle1
12,      # 21	RightUpLeg
14,     # 22	RightLeg
16,     # 23	RightFoot
-1,     # 24	RightForeFoot
-1,     # 25	RightToeBase
-1,     # 26	RightToeBaseEnd
11,     # 27	LeftUpLeg
13,     # 28	LeftLeg
15,     # 29	LeftFoot
-1,     # 30	LeftForeFoot
-1,     # 31	LeftToeBase
-1      # 32	LeftToeBaseEnd
]


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
gt2openpose = [
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
        default="G:/MULTI4D_Dataset/Human4D",      
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
        # default="G:/MULTI4D_Dataset/single_person/single_files.txt"
        default="G:/MULTI4D_Dataset/multiperson_volset/multi_files.txt"
    )
    args = parser.parse_args()


    test_set = {}
    if (args.test_set_ids != ""):        
        f = open(args.test_set_ids, 'r')
        lines = f.readlines()
        for line in lines:
            values = line.split(' ')
            seq_key = values[1].split('\\')[2]
            if seq_key not in test_set.keys() and '!' not in seq_key:
                test_set[seq_key] = []
            if seq_key in test_set.keys():
                test_set[seq_key].append(int(values[0]))


    # gt_folders = os.listdir(args.gt_path)
    # pred_folders = os.listdir(args.pred_path)
    subject_folders = [folder for folder in os.listdir(args.dataset_path) if 'S' in folder]
    errors = {}
    max_edges = {}
    head_edges = {}
    mae = {}
    rmse = {}
    pck = {}
    pck_05 = {}
    for s in subject_folders:
        errors[s] = {}
        max_edges[s] = []
        head_edges[s] = []
        errors[s]['mae'] = []
        errors[s]['rmse'] = []
        errors[s]['pck_0.1'] = []
        errors[s]['pck_0.05'] = []
        s_root = os.path.join(args.dataset_path, s)
        folders = [folder for folder in os.listdir(s_root) if '-' in folder]

        for i in range(len(folders)):
            if (not folders[i] in test_set):
                print(s + " " + folders[i])
                continue
            offset = load_rgbd_skip(os.path.join(s_root, "offsets.txt"), folders[i])
            npy_files_gt = [npy for npy in os.listdir(os.path.join(s_root, folders[i])) if '_2d' in npy]
            if len(npy_files_gt) == 0:               
                print(s + " " + folders[i])
                continue

            # continue
            npy_bb_files_gt = [npy for npy in os.listdir(os.path.join(s_root, folders[i])) if '_bbox' in npy]
            npy_files_pred = {}
            for cam in args.cameras:
                npy_files_pred[cam] = [npy for npy in os.listdir(os.path.join(s_root, folders[i], 'Dump/alphapose/color_joints_processed')) if cam in npy]
                # npy_files_pred[cam] = [npy for npy in os.listdir(os.path.join(s_root, folders[i], 'Dump/openpose/color_joints_processed')) if cam in npy]
                # npy_files_pred[cam] = [npy for npy in os.listdir(os.path.join(s_root, folders[i], 'Dump/vnect/vnect_npys')) if cam in npy]
                sort_nicely(npy_files_pred[cam])
                npy_files_pred[cam] = npy_files_pred[cam][offset:]

            # npy_files_pred = [npy for npy in os.listdir(os.path.join(args.pred_path, pred_folders[i])) if '_2d' in npy]
            for cam in args.cameras:
                npy_cam_gt = [gtfile for gtfile in npy_files_gt if cam in gtfile].pop(0)
                npy_bb_cam_gt = [gtfile for gtfile in npy_bb_files_gt if cam in gtfile].pop(0)
                # npy_cam_pred = [predfile for predfile in npy_files_pred if cam in predfile].pop(0)

                gt = numpy.load(os.path.join(s_root, folders[i], npy_cam_gt))
                bb_gt = numpy.load(os.path.join(s_root, folders[i], npy_bb_cam_gt))
                # pred = numpy.zeros([len(npy_files_pred[cam]), 2, 25, 2]) # openpose
                pred = numpy.zeros([len(npy_files_pred[cam]), 2, 17, 2]) # alphapose
                # pred = numpy.zeros([len(npy_files_pred[cam]), 2, 17, 2]) # vnect

                # assert gt.shape[0] == pred.shape[0]
                length = min(gt.shape[0], pred.shape[0])
                for fid in range(pred.shape[0]):
                    npy = numpy.load(os.path.join(s_root, folders[i], 'Dump/alphapose/color_joints_processed', npy_files_pred[cam][fid]))
                    # npy = numpy.load(os.path.join(s_root, folders[i], 'Dump/openpose/color_joints_processed', npy_files_pred[cam][fid]))
                    # npy = numpy.load(os.path.join(s_root, folders[i], 'Dump/vnect/vnect_npys', npy_files_pred[cam][fid]))
                    if (npy.shape[0] == pred[fid].shape[0]):
                        pred[fid] = npy[:,:,2:] # openpose
                        # pred[fid] = npy
                    else:
                        pred[fid, :, :, :] = -10000


                # pred = numpy.load(os.path.join(args.pred_path, pred_folders[i], npy_cam_pred))
                
                # abs_error = numpy.zeros([gt.shape[0], gt.shape[1] * gt.shape[2]])
                # squared_error = numpy.zeros([gt.shape[0], gt.shape[1] * gt.shape[2]])
                abs_error = numpy.zeros([len(test_set[folders[i]]), 1 + pred.shape[1] * len(gt2alphapose)])
                squared_error = numpy.zeros([len(test_set[folders[i]]), 1 + pred.shape[1] * len(gt2alphapose)])
                pck_error = numpy.zeros([len(test_set[folders[i]]), 1 + pred.shape[1] * len(gt2alphapose)])
                pck_error_05 = numpy.zeros([len(test_set[folders[i]]), 1 + pred.shape[1] * len(gt2alphapose)])
              
                # we should evaluate only the test dataset fids
                # for fid in range(pred.shape[0]):
                for k in range(len(test_set[folders[i]])):

                    
                    # for fid in test_set[folders[i]]:
                    fid = test_set[folders[i]][k] - offset
                    abs_error[k, 0] = fid
                    error = 0
                    counter = 0
                    img = cv2.imread(os.path.join(s_root, folders[i], 'Dump/color', npy_files_pred[cam][fid].replace(".npy", ".png")))
                    if (pred.shape[1] > 1):                        
                        gt_center_0 = numpy.zeros([2])
                        pred_center_0 = numpy.zeros([2])
                        pred_center_1 = numpy.zeros([2])
                        counter = 0
                        # for j in range(len(gt2openpose)):
                        for j in range(len(gt2alphapose)):
                            # if (gt2openpose[j] != -1 and pred[fid, 0, gt2openpose[j], 0] > 0 and pred[fid, 1, gt2openpose[j], 0] > 0):
                            #     gt_center_0 += gt[fid, 0, j]
                            #     pred_center_0 += pred[fid, 0, gt2openpose[j]]
                            #     pred_center_1 += pred[fid, 1, gt2openpose[j]]
                            #     counter += 1 
                            if (gt2alphapose[j] != -1 and pred[fid, 0, gt2alphapose[j], 0] > 0 and pred[fid, 1, gt2alphapose[j], 0] > 0):
                                gt_center_0 += gt[fid, 0, j]
                                pred_center_0 += pred[fid, 0, gt2alphapose[j]]
                                pred_center_1 += pred[fid, 1, gt2alphapose[j]]
                                counter += 1  
                                # img = cv2.drawMarker(img, (int(gt[fid, 0, j, 0]), int(gt[fid, 0, j, 1])), (100, 200, 255), markerType=cv2.MARKER_DIAMOND)
                                # img = cv2.drawMarker(img, (int(pred[fid, 0, gt2openpose[j], 0]), int(pred[fid, 0, gt2openpose[j], 1])), (100, 0, 255), markerType=cv2.MARKER_DIAMOND)
                                # img = cv2.drawMarker(img, (int(pred[fid, 1, gt2openpose[j], 0]), int(pred[fid, 1, gt2openpose[j], 1])), (0, 100, 255), markerType=cv2.MARKER_DIAMOND)
                               

    
                        gt_center_0 /= counter
                        pred_center_0 /= counter
                        pred_center_1 /= counter

                        dist00 = numpy.linalg.norm(gt_center_0 - pred_center_0)
                        dist01 = numpy.linalg.norm(gt_center_0 - pred_center_1)
                        if (dist00 > dist01):
                            idx = [1, 0]
                        else:
                            idx = [0, 1]
                    else:
                        idx = [0]

                    head_edge_values = numpy.zeros([pred.shape[1]])
                    error = numpy.zeros([pred.shape[1]])
                    counter = numpy.zeros([pred.shape[1]])
                    for p in range(pred.shape[1]):
                        # img = cv2.drawMarker(img, (int(gt_center_0[0]), int(gt_center_0[1])), (0, 255, 100), markerType=cv2.MARKER_DIAMOND)
                        # img = cv2.drawMarker(img, (int(pred_center_0[0]), int(pred_center_0[1])), (255, 100, 0), markerType=cv2.MARKER_DIAMOND)
                        # img = cv2.drawMarker(img, (int(pred_center_1[0]), int(pred_center_1[1])), (100, 0, 255), markerType=cv2.MARKER_DIAMOND)
                        # cv2.imshow("test", img)
                        # if (args.show_data):
                        # img = numpy.zeros([720, 1280, 3])
                        
                        max_edge = max(bb_gt[fid, 0, 2]-bb_gt[fid, 0, 0], bb_gt[fid, 0, 3]-bb_gt[fid, 0, 1])

                        
                        head_edge_values[p] = numpy.linalg.norm(gt[fid, p, 8]-gt[fid, p, 5])
                        
                        if (args.show_data):
                            # for p in range(bb_gt.shape[0]):
                            img = cv2.rectangle(img, (bb_gt[fid, p, 0], bb_gt[fid, p, 1]), \
                                (bb_gt[fid, p, 2], bb_gt[fid, p, 3]), (255, 0, 0))

                            img = cv2.line(img, (int(gt[fid, p, 8, 0]), int(gt[fid, p, 8, 1])), (int(gt[fid, p, 5, 0]), int(gt[fid, p, 5, 1])), (156, 40, 240))

                        print(os.path.join(s_root, folders[i], 'Dump/color', npy_files_pred[cam][fid]))
                        for j in range(len(gt2alphapose)):
                            if (gt2alphapose[j] != -1):
                            # if (gt2openpose[j] != -1):
                            # if (gt2vnect[j] != -1):
                                

                                # cv2.imshow('img', cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))
                                # abs_error[i, (p+1) * j] = calculate_dist(gt[fid, p, j], pred[fid, p, gt2alphapose[j]])
                                if (args.show_data):
                                    img = cv2.drawMarker(img, (int(gt[fid, p, j, 0]), int(gt[fid, p, j, 1])), (0, 255, 0), markerType=cv2.MARKER_CROSS)
                                    img = cv2.drawMarker(img, (int(pred[fid, idx[p], gt2alphapose[j], 0]), int(pred[fid, idx[p], gt2alphapose[j], 1])), (0, 0, 255), markerType=cv2.MARKER_SQUARE)
                                    # img = cv2.line(img, (int(gt[fid, p, j, 0]), (int(pred[fid, idx[p], gt2alphapose[j], 0]), int(pred[fid, idx[p], gt2alphapose[j], 1])), (156, 40, 240))

                                # if (gt[fid, p, j][0] <= 0 or gt[fid, p, j][1] <= 0 or pred[fid, p, gt2alphapose[j]][0] <= 0 or pred[fid, p, gt2alphapose[j]][1] <= 0):
                                #     print(s + ' ' + folders[i] + ' ' + str(k) + ' ' + str(gt2alphapose[j]))
                                #     with open("log.log", 'a') as f:
                                #         f.write(s + ' ' + folders[i] + ' ' + str(k) + ' ' + str(gt2alphapose[j]))
                                    
                                #     continue

                                err = calculate_dist(gt[fid, p, j], pred[fid, idx[p], gt2alphapose[j]])
                                # err = calculate_dist(gt[fid, p, j], pred[fid, idx[p], gt2openpose[j]])
                                # err = calculate_dist(gt[fid, p, j], pred[fid, p, gt2vnect[j]])

                                if (err < 600):
                                    error[p] += err
                                    counter[p] += 1
                                # if (error > 500):
                                #     print(s + ' ERROR > 500 ' + folders[i] + ' ' + str(k) + ' ' + str(gt2alphapose[j]) + ' error: ' + str(error))
                                #     continue
                                #     # cv2.imshow('img', cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))
                                #     # cv2.waitKey()

                                abs_error[k, 1 + (idx[p]+1) * j] = err
                                squared_error[k, 1 + (idx[p]+1) * j] = err * err
                                #print(s + ' ' + folders[i] + ' ' + str(k) + ' ' + str(gt2alphapose[j]) + ' ' + str(max_edge))
                               
                                pck_error[k, 1 + (idx[p]+1) * j] = 1 if (err / max_edge) < 0.1 else 0
                                pck_error_05[k, 1 + (idx[p]+1) * j] = 1 if (err / max_edge) < 0.05 else 0
                        
                        # cv2.imshow('img', cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))
                        # cv2.waitKey(10) 

                        if (counter[p] == 0):
                            error[p] = 10000
                        else:
                            error[p] /= counter[p]
                    max_edges[s].append(max_edge)
                    head_edges[s].append(head_edge_values)
                    errors[s]['mae'].append(error)
                    errors[s]['rmse'].append(error * error)
                    errors[s]['pck_0.1'].append(pck_error[k, 1 + (idx[p]+1) * j])
                    # errors[s]['pck_0.05'].append(pck_error_05[k, 1 + (p+1) * j])

                    # if (args.show_data):                   
                    #     cv2.imshow('img', cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))
                    #     cv2.waitKey()

                    with open("files_out.txt", 'a') as log:
                        log.write(folders[i] + " " + str(k) + "\n")

                out_folder = os.path.join(s_root, folders[i], 'Dump/alphapose/results')
                # out_folder = os.path.join(s_root, folders[i], 'Dump/openpose/results')
                # out_folder = os.path.join(s_root, folders[i], 'Dump/vnect/results')
                if not os.path.exists(out_folder):
                    os.makedirs(out_folder)
                numpy.savetxt(os.path.join(out_folder, cam + "_abs.csv"), abs_error, delimiter=',')
                numpy.savetxt(os.path.join(out_folder, cam + "_squared.csv"), squared_error, delimiter=',')
                numpy.savetxt(os.path.join(out_folder, cam + "_pck.csv"), pck_error, delimiter=',')
                numpy.savetxt(os.path.join(out_folder, cam + "_pck_0.5.csv"), pck_error_05, delimiter=',')
        
        
        out_dir = os.path.join(args.dataset_path, "../results")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        numpy.savetxt(os.path.join(out_dir, s + "_mae.csv"), numpy.asarray(errors[s]['mae']), delimiter=',')
        numpy.savetxt(os.path.join(out_dir, s + "_max_edge.csv"), numpy.asarray(max_edges[s]), delimiter=',')
        numpy.savetxt(os.path.join(out_dir, s + "_head_edge.csv"), numpy.asarray(head_edges[s]), delimiter=',')
        
        mae[s] = numpy.asarray(errors[s]['mae']).mean()
        rmse[s] = numpy.sqrt(numpy.asarray(errors[s]['rmse']).mean())
        pck[s] = numpy.asarray(errors[s]['pck_0.1']).mean()
        pck_05[s] = numpy.asarray(errors[s]['pck_0.05']).mean()


    with open(os.path.join(args.dataset_path, 'eval_errors_single.csv'), 'w') as f:
        for key in mae.keys():
            f.write("%s,%f,%f,%f,%f\n"%(key, mae[key], rmse[key], pck_05[key], pck[key]))

if __name__ == "__main__":
    main()
