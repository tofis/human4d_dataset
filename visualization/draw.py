import numpy
import cv2

def draw_skeleton_joints_19(skeleton_img, keypoints, color, thickness=3):      
        for i in range(0, 4):
            cv2.line(skeleton_img, 
                (int(keypoints[i][0]), int(keypoints[i][1])), 
                (int(keypoints[i + 1][0]), int(keypoints[i + 1][1])), 
                # COLORS["{:02d}".format(i + 1)], thickness)   
                color, thickness)   

        for i in range(5, 7):
            cv2.line(skeleton_img, 
               (int(keypoints[i][0]), int(keypoints[i][1])), 
                (int(keypoints[i + 1][0]), int(keypoints[i + 1][1])), 
                # COLORS["{:02d}".format(i + 1)], thickness)   
                color, thickness)    

        for i in range(8, 10):
            cv2.line(skeleton_img, 
                (int(keypoints[i][0]), int(keypoints[i][1])), 
                (int(keypoints[i + 1][0]), int(keypoints[i + 1][1])), 
                # COLORS["{:02d}".format(i + 1)], thickness)   
                color, thickness)   

        for i in range(11, 13):
            cv2.line(skeleton_img, 
                (int(keypoints[i][0]), int(keypoints[i][1])), 
                (int(keypoints[i + 1][0]), int(keypoints[i + 1][1])), 
                # COLORS["{:02d}".format(i + 1)], thickness)   
                color, thickness)   

        for i in range(15, 17):
            cv2.line(skeleton_img,
                (int(keypoints[i][0]), int(keypoints[i][1])), 
                (int(keypoints[i + 1][0]), int(keypoints[i + 1][1])), 
                # COLORS["{:02d}".format(i + 1)], thickness)   
                color, thickness)   

        # cv2.line(skeleton_img, 
        #         (int(w * labels[batch][p][3][0]), int(h * labels[batch][p][3][1])), 
        #         (int(w * labels[batch][p][5][0]), int(h * labels[batch][p][5][1])), 
        #         MARKERS["{:02d}".format(i + 1)], 1) 
        # cv2.line(skeleton_img, 
        #         (int(w * labels[batch][p][3][0]), int(h * labels[batch][p][3][1])), 
        #         (int(w * labels[batch][p][8][0]), int(h * labels[batch][p][8][1])), 
        #         MARKERS["{:02d}".format(i + 1)], 1) 

        return skeleton_img

def draw_skeleton_joints_(input_img, keypoints, COLORS, thickness=3):
    for i in range(0, 8):
        cv2.line(input_img, 
            (int(keypoints[i][0]), int(keypoints[i][1])), 
            (int(keypoints[i + 1][0]), int(keypoints[i + 1][1])), 
            # COLORS["{:02d}".format(i + 1)], thickness)   
            COLORS[i], thickness)   

    for i in range(9, 14):
        cv2.line(input_img, 
            (int(keypoints[i][0]), int(keypoints[i][1])), 
            (int(keypoints[i + 1][0]), int(keypoints[i + 1][1])), 
            # COLORS["{:02d}".format(i + 1)], thickness)   
            COLORS[i], thickness)   
    for i in range(15, 20):
        cv2.line(input_img, 
            (int(keypoints[i][0]), int(keypoints[i][1])), 
            (int(keypoints[i + 1][0]), int(keypoints[i + 1][1])), 
            # COLORS["{:02d}".format(i + 1)], thickness)   
            COLORS[i], thickness)   
    for i in range(21, 26):
        cv2.line(input_img, 
            (int(keypoints[i][0]), int(keypoints[i][1])), 
            (int(keypoints[i + 1][0]), int(keypoints[i + 1][1])), 
            # COLORS["{:02d}".format(i + 1)], thickness)   
            COLORS[i], thickness)   
    for i in range(27, 32):
        cv2.line(input_img, 
            (int(keypoints[i][0]), int(keypoints[i][1])), 
            (int(keypoints[i + 1][0]), int(keypoints[i + 1][1])), 
            # COLORS["{:02d}".format(i + 1)], thickness)   
            COLORS[i], thickness)      

    # cv2.line(input_img, 
    #     (int(keypoints[0][0]), int(keypoints[0][1])), 
    #     (int(keypoints[21][0]), int(keypoints[21][1])), 
    #     COLORS["{:02d}".format(40)], thickness)  

    # cv2.line(input_img, 
    #     (int(keypoints[0][0]), int(keypoints[0][1])), 
    #     (int(keypoints[27][0]), int(keypoints[27][1])), 
    #     COLORS["{:02d}".format(50)], thickness)  

    return input_img


def draw_skeleton_joints(input_img, keypoints, COLORS, thickness=3):
    for i in range(0, 8):
        cv2.line(input_img, 
            (int(keypoints[i][0]), int(keypoints[i][1])), 
            (int(keypoints[i + 1][0]), int(keypoints[i + 1][1])), 
            COLORS["{:02d}".format(i + 1)], thickness)   
            # COLORS[i], thickness)   

    for i in range(9, 14):
        cv2.line(input_img, 
            (int(keypoints[i][0]), int(keypoints[i][1])), 
            (int(keypoints[i + 1][0]), int(keypoints[i + 1][1])), 
            COLORS["{:02d}".format(i + 1)], thickness)   
            # COLORS[i], thickness)   
    for i in range(15, 20):
        cv2.line(input_img, 
            (int(keypoints[i][0]), int(keypoints[i][1])), 
            (int(keypoints[i + 1][0]), int(keypoints[i + 1][1])), 
            COLORS["{:02d}".format(i + 1)], thickness)   
            # COLORS[i], thickness)   
    for i in range(21, 26):
        cv2.line(input_img, 
            (int(keypoints[i][0]), int(keypoints[i][1])), 
            (int(keypoints[i + 1][0]), int(keypoints[i + 1][1])), 
            COLORS["{:02d}".format(i + 1)], thickness)   
            # COLORS[i], thickness)   
    for i in range(27, 32):
        cv2.line(input_img, 
            (int(keypoints[i][0]), int(keypoints[i][1])), 
            (int(keypoints[i + 1][0]), int(keypoints[i + 1][1])), 
            COLORS["{:02d}".format(i + 1)], thickness)   
            # COLORS[i], thickness)      

    # cv2.line(input_img, 
    #     (int(keypoints[0][0]), int(keypoints[0][1])), 
    #     (int(keypoints[21][0]), int(keypoints[21][1])), 
    #     COLORS["{:02d}".format(40)], thickness)  

    # cv2.line(input_img, 
    #     (int(keypoints[0][0]), int(keypoints[0][1])), 
    #     (int(keypoints[27][0]), int(keypoints[27][1])), 
    #     COLORS["{:02d}".format(50)], thickness)  

    return input_img