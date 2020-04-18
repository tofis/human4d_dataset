import numpy
import cv2
def draw_skeleton_joints(input_img, keypoints, COLORS):
    for i in range(0, 8):
        cv2.line(input_img, 
            (int(keypoints[i][0]), int(keypoints[i][1])), 
            (int(keypoints[i + 1][0]), int(keypoints[i + 1][1])), 
            COLORS["{:02d}".format(i + 1)], 1)   

    for i in range(9, 14):
        cv2.line(input_img, 
            (int(keypoints[i][0]), int(keypoints[i][1])), 
            (int(keypoints[i + 1][0]), int(keypoints[i + 1][1])), 
            COLORS["{:02d}".format(i + 1)], 1) 

    for i in range(15, 20):
        cv2.line(input_img, 
            (int(keypoints[i][0]), int(keypoints[i][1])), 
            (int(keypoints[i + 1][0]), int(keypoints[i + 1][1])), 
            COLORS["{:02d}".format(i + 1)], 1) 

    for i in range(21, 26):
        cv2.line(input_img, 
            (int(keypoints[i][0]), int(keypoints[i][1])), 
            (int(keypoints[i + 1][0]), int(keypoints[i + 1][1])), 
            COLORS["{:02d}".format(i + 1)], 1) 

    for i in range(27, 32):
        cv2.line(input_img, 
            (int(keypoints[i][0]), int(keypoints[i][1])), 
            (int(keypoints[i + 1][0]), int(keypoints[i + 1][1])), 
            COLORS["{:02d}".format(i + 1)], 1)    

    cv2.line(input_img, 
        (int(keypoints[0][0]), int(keypoints[0][1])), 
        (int(keypoints[21][0]), int(keypoints[21][1])), 
        COLORS["{:02d}".format(40)], 1)  

    cv2.line(input_img, 
        (int(keypoints[0][0]), int(keypoints[0][1])), 
        (int(keypoints[27][0]), int(keypoints[27][1])), 
        COLORS["{:02d}".format(50)], 1)  

    return input_img