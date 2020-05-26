import numpy
import os
import argparse
import csv

def calculate_dist(kpt_gt, kpt_pred):
    return numpy.linalg.norm(numpy.subtract(kpt_gt, kpt_pred))

def main():
    parser = argparse.ArgumentParser(description="PyTorch Show Pose and Pointcloud")    
    parser.add_argument(
        "--gt_path",
        default="G:/MULTI4D_Dataset/core/Subject3",      
        help="path to gt npy sequence files",
    )
    parser.add_argument(
        "--pred_path",
        default="G:/MULTI4D_Dataset/core/Subject3",      
        help="path to gt npy sequence files",
    )
    parser.add_argument(
        "--cameras",
        default=["M72e", "M72h", "M72i", "M72j"]
    )
    args = parser.parse_args()

    gt_folders = os.listdir(args.gt_path)
    pred_folders = os.listdir(args.pred_path)

    assert len(gt_folders) == len(pred_folders)

    for i in range(len(gt_folders)):
        npy_files_gt = [npy for npy in os.listdir(os.path.join(args.gt_path, gt_folders[i])) if '_2d' in npy]
        npy_files_pred = [npy for npy in os.listdir(os.path.join(args.pred_path, pred_folders[i])) if '_2d' in npy]
        abs_error = []
        for cam in args.cameras:
            npy_cam_gt = [gtfile for gtfile in npy_files_gt if cam in gtfile].pop(0)
            npy_cam_pred = [predfile for predfile in npy_files_pred if cam in predfile].pop(0)

            gt = numpy.load(os.path.join(args.gt_path, gt_folders[i], npy_cam_gt))
            pred = numpy.load(os.path.join(args.pred_path, pred_folders[i], npy_cam_pred))
            abs_error = numpy.zeros([gt.shape[0], gt.shape[1] * gt.shape[2]])
            for fid in range(gt.shape[0]):
                for p in range(gt.shape[1]):
                    for j in range(gt.shape[2]):
                        abs_error[fid, p * j] = calculate_dist(gt[fid, p, j], pred[fid, p, j])
            numpy.savetxt(os.path.join(args.gt_path, gt_folders[i], cam + ".csv"), abs_error, delimiter=',')
        


        





if __name__ == "__main__":
    main()
