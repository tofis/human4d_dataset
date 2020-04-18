# import c3d

# with open('E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/ARTANIM/Vicon/Ball_Calibrations/CalibrationBallSequence.c3d', 'rb') as handle:
#     reader = c3d.Reader(handle)
#     for i, (points, analog) in enumerate(reader.read_frames()):
#         print('Frame {}: {}'.format(i, points.round(2)))

import os
from ezc3d import c3d

path = 'E:/VCL/Users/tofis/Data/DATASETS/RGBDIRD_MOCAP_DATASET/Data/ARTANIM/Vicon/Ball_Calibrations'
file = 'Bart_CalibrationBallSequence_04.c3d'

c = c3d(os.path.join(path, file))
print(c['parameters']['POINT']['USED']['value'][0]);  # Print the number of points used
point_data = c['data']['points']

f = open (os.path.join(path, file.replace('c3d', 'txt')), 'w')
for i in range(0, point_data.shape[2]):
    f.write(str(point_data[0][0][i]) + " " + str(point_data[1][0][i]) + " " + str(point_data[2][0][i]) + "\n")

f.close()