import os
from shutil import copyfile

import numpy
import cv2
from importers import *
from visualization import turbo_colormap


# root = 'G:/MULTI4D_Dataset/HUMAN4D/S2/19-07-12-08-58-57/Dump/'
root = 'E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/Recordings/_ird_recordings/S1/19-07-12-08-15-13/Dump'
out = os.path.join(root, 'out')
if (not os.path.exists(out)):
    os.makedirs(out)
sample_id = "523"
# cameras = ["M72e", "M72h", "M72i", "M72j"]

colorz = [color for color in os.listdir(os.path.join(root, 'color')) \
    if sample_id in color.split('_')[0]]
depthz = [depth for depth in os.listdir(os.path.join(root, 'depth')) \
    if sample_id in depth.split('_')[0]]

for color in colorz:    
    copyfile(os.path.join(root, 'color', color), os.path.join(out, color))

for depth in depthz:
    depth_img = readpgm(os.path.join(root, 'depth', depth)).astype(numpy.float) / 10.0
    # depth_img[depth_img > 3000] = 0
    # depth_img[depth_img < 1200] = 0
    depth_img /= numpy.max(depth_img)
    # depth_img /= 10000
    depth_img = (depth_img * 255).astype(numpy.uint8)
    img = numpy.zeros([180, 320, 3], dtype=numpy.float32)
    for x in range(img.shape[1]):
        for y in range(img.shape[0]):
            img[y, x] = turbo_colormap.turbo_colormap_data[depth_img[y, x]]
    # img = cv2.LUT(depth_img, numpy.asarray(turbo_colormap.turbo_colormap_data).astype(numpy.float32))
    img = (img * 255).astype(numpy.uint8)
    #cv2.imshow("show", cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))
    cv2.imwrite(os.path.join(out, depth).replace('.pgm', '.png'), cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))
    
    #cv2.waitKey()




