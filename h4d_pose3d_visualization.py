import os
import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

pose3d = numpy.load('E:/Datasets/HUMAN4D/S1/19-07-12-07-32-22/Dump/gposes3d/167.npy').squeeze()
fig = plt.figure(figsize=(4, 6))

ax = fig.add_subplot(111, projection='3d')
lim = 700
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, 1.3 * lim)
ax.set_zlim(-lim, lim)
ax.view_init(elev=30, azim = 135)

x_0 = pose3d[:9, 0]
y_0 = pose3d[:9, 1]
z_0 = pose3d[:9, 2]

ax.plot(x_0, y_0, z_0, label='gt', color='blue')

x_1 = pose3d[9:15, 0]
y_1 = pose3d[9:15, 1]
z_1 = pose3d[9:15, 2]

ax.plot(x_1, y_1, z_1, color='blue')

x_2 = pose3d[15:21, 0]
y_2 = pose3d[15:21, 1]
z_2 = pose3d[15:21, 2]

ax.plot(x_2, y_2, z_2, color='blue')

x_3 = pose3d[21:27, 0]
y_3 = pose3d[21:27, 1]
z_3 = pose3d[21:27, 2]

ax.plot(x_3, y_3, z_3, color='blue')

x_4 = pose3d[27:33, 0]
y_4 = pose3d[27:33, 1]
z_4 = pose3d[27:33, 2]

ax.plot(x_4, y_4, z_4, color='blue')

ax.legend()
plt.show()

