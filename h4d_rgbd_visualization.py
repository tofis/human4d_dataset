import os
import numpy
import cv2
from importers import readpgm
from visualization import turbo_colormap

# sequence root folder, including color and depth subfolders with RGBD data
root = 'E:/Datasets/HUMAN4D/S1/19-07-12-07-32-22/Dump'
out = os.path.join(root, 'out')
if (not os.path.exists(out)):
    os.makedirs(out)

colorz = os.listdir(os.path.join(root, 'color'))
depthz = os.listdir(os.path.join(root, 'depth'))

assert len(colorz) == len(depthz)

for i in range(len(depthz)):
    # depth values in 100um are converted to mm by diving by 10.0
    depth_img = readpgm(os.path.join(root, 'depth', depthz[i])).astype(numpy.float) / 10.0
    # normalization
    depth_img /= numpy.max(depth_img)
    depth_img = (depth_img * 255).astype(numpy.uint8)
    colored_depth_img = numpy.zeros([depth_img.shape[0], depth_img.shape[1], 3], dtype=numpy.float32)
    # turbo colormap visualization
    for x in range(colored_depth_img.shape[1]):
        for y in range(colored_depth_img.shape[0]):
            colored_depth_img[y, x] = turbo_colormap.turbo_colormap_data[depth_img[y, x]]            
    colored_depth_img = (colored_depth_img * 255).astype(numpy.uint8)

    # rotate to show the image properly due to vertical orientation
    cv2.imshow("depth", cv2.rotate(colored_depth_img, cv2.ROTATE_90_CLOCKWISE))

    color_img = cv2.imread(os.path.join(root, 'color', colorz[i]))
    # rotate to show the image properly due to vertical orientation
    cv2.imshow("color", cv2.rotate(color_img, cv2.ROTATE_90_CLOCKWISE))
    cv2.waitKey(1)




