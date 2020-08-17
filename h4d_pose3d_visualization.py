import os
import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



# pose3d_pred = numpy.load('C:/Users/tofis.ITI-510/Documents/GitHub/learnable-triangulation-pytorch/pred.npy')
full_results_3d = numpy.load('C:/Vol1/dbstore/datasets/k.iskakov/logs/multi-view-net-repr/eval_human36m_alg_AlgebraicTriangulationNet@02.07.2020-22-25-13/checkpoints/0000/results.pkl', allow_pickle=True)
pose3d_pred = full_results_3d['keypoints_3d']
pose3d_gt = full_results_3d['keypoints_3d_gt']
fig = plt.figure(figsize=(4, 6))

for frame_id in range(pose3d_gt.shape[0]):
    print(frame_id)
    ax = fig.add_subplot(111, projection='3d')
    lim = 700
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, 1.3 * lim)
    ax.set_zlim(-lim, lim)
    ax.view_init(elev=30, azim = 135)

    x_0 = pose3d_pred[frame_id, :6, 0]
    y_0 = pose3d_pred[frame_id, :6, 1]
    z_0 = pose3d_pred[frame_id, :6, 2]

    ax.plot(x_0, y_0, z_0, label='pred', color='red')
    x_1 = pose3d_pred[frame_id, 6:10, 0]
    y_1 = pose3d_pred[frame_id, 6:10, 1]
    z_1 = pose3d_pred[frame_id, 6:10, 2]

    ax.plot(x_1, y_1, z_1, color='red')

    x_2 = pose3d_pred[frame_id, 10:16, 0]
    y_2 = pose3d_pred[frame_id, 10:16, 1]
    z_2 = pose3d_pred[frame_id, 10:16, 2]

    ax.plot(x_2, y_2, z_2, color='red')

    # pose3d_gt = numpy.load('C:/Users/tofis.ITI-510/Documents/GitHub/learnable-triangulation-pytorch/gt.npy')

    x_0 = pose3d_gt[frame_id, :6, 0]
    y_0 = pose3d_gt[frame_id, :6, 1]
    z_0 = pose3d_gt[frame_id, :6, 2]

    ax.plot(x_0, y_0, z_0, label='gt', color='blue')
    x_1 = pose3d_gt[frame_id, 6:10, 0]
    y_1 = pose3d_gt[frame_id, 6:10, 1]
    z_1 = pose3d_gt[frame_id, 6:10, 2]

    ax.plot(x_1, y_1, z_1, color='blue')

    x_2 = pose3d_gt[frame_id, 10:16, 0]
    y_2 = pose3d_gt[frame_id, 10:16, 1]
    z_2 = pose3d_gt[frame_id, 10:16, 2]

    ax.plot(x_2, y_2, z_2, color='blue')

    ax.legend()

    plt.savefig(os.path.join('C:/Vol1/dbstore/datasets/k.iskakov/logs/multi-view-net-repr/eval_human36m_alg_AlgebraicTriangulationNet@02.07.2020-22-25-13/checkpoints/0000/out', str(frame_id) + '.png'))
    # plt.show()

