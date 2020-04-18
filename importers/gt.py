import numpy

def load_markers(filename, scale=10):
    f = open(filename, "r")
    lines = f.readlines()
    markers = numpy.zeros([53, 3], dtype=float)
    for i in range(len(lines)):
        values = lines[i].split(" ")
        markers[i, 0] = float(values[2])
        markers[i, 1] = float(values[4])
        markers[i, 2] = float(values[3])

    return markers * scale

def load_joints(filename, scale=10):
    f = open(filename, "r")
    lines = f.readlines()
    joints = numpy.zeros([33, 3], dtype=float)
    for i in range(len(lines)):
        values = lines[i].split(" ")
        joints[i, 0] = float(values[2])
        joints[i, 1] = float(values[4])
        joints[i, 2] = float(values[3])

    return joints * scale

def load_markers_seq(filename, scale=10, markers_num=53):
    f_clap = open(filename.replace(".markers", ".clap"), "r")
    clap_id = int(f_clap.readline().split(' ')[0])

    f = open(filename, "r")
    lines = f.readlines()
    number_of_frames = len(lines) // markers_num
    markers = numpy.zeros([number_of_frames, markers_num, 3], dtype=float)
    for i in range(len(lines)):
        f = i // markers_num
        m = i % markers_num        
        values = lines[i].split(" ")
        markers[f, m, 0] = float(values[2])
        markers[f, m, 1] = float(values[4])
        markers[f, m, 2] = float(values[3])

    return markers[clap_id:, :, :] * scale

def load_joints_seq(filename, scale=10, joints_num=33):
    f_clap = open(filename.replace(".joints", ".clap"), "r")
    clap_id = int(f_clap.readline().split(' ')[0])

    f = open(filename, "r")
    lines = f.readlines()
    number_of_frames = len(lines) // joints_num
    joints = numpy.zeros([number_of_frames, joints_num, 3], dtype=float)
    for i in range(len(lines)):
        f = i // joints_num
        j = i % joints_num        
        values = lines[i].split(" ")
        joints[f, j, 0] = float(values[2])
        joints[f, j, 1] = float(values[4])
        joints[f, j, 2] = float(values[3])

    return joints[clap_id:, :, :] * scale