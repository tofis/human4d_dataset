import torch
import cv2
import numpy

def save_image(filename, tensor, scale=255.0):
    b, _, __, ___ = tensor.size()
    for n in range(b):
        array = tensor[n, :, :, :].detach().cpu().numpy()
        array = array.transpose(1, 2, 0) * scale
        cv2.imwrite(filename.replace("#", str(n)), array)

def save_depth(filename, tensor, scale=1000.0):
    b, _, __, ___ = tensor.size()
    for n in range(b):
        array = tensor[n, :, :, :].detach().cpu().numpy()
        array = array.transpose(1, 2, 0) * scale
        array = numpy.uint16(array)
        cv2.imwrite(filename.replace("#", str(n)), array)

def save_data(filename, tensor, scale=1000.0):
    b, _, __, ___ = tensor.size()
    for n in range(b):
        array = tensor[n, :, :, :].detach().cpu().numpy()
        array = array.transpose(1, 2, 0) * scale
        array = numpy.float32(array)
        cv2.imwrite(filename.replace("#", str(n)), array)

def save_depth_from_3d(filename, tensor, scale=1000.0):
    b, _, __, ___ = tensor.size()
    for n in range(b):
        array = tensor[n, :, :, :].detach().cpu().numpy()
        depth_channel = numpy.zeros((1, array.shape[1], array.shape[2]))
        depth_channel[0,:,:] = array[2, :, :]
        depth_channel = depth_channel.transpose(1, 2, 0) * scale
        depth_channel = numpy.uint16(depth_channel)
        cv2.imwrite(filename.replace("#", str(n)), depth_channel)

def save_depth_from_unstructured_3d(filename, height, width, points3d, intrinsics, depth_thres):
    # points3d *= 1000.0
    depth_image = numpy.zeros((1, height, width), dtype=numpy.int16)
    for point3d in points3d:
        
        x2d = point3d[0] * intrinsics[0][0]/point3d[2] + intrinsics[0][2]        
        y2d = point3d[1] * intrinsics[1][1]/point3d[2] + intrinsics[1][2]        
        
        if (x2d >= 0 and x2d < width and y2d >=0 and y2d < height and point3d[2] < depth_thres):
            depth_image[0, int(y2d), int(x2d)] = point3d[2]*1000


    depth_image_transposed = depth_image.transpose(1, 2, 0)
    depth_image_transposed = numpy.uint16(depth_image_transposed)

    cv2.imwrite(filename, depth_image_transposed)
    return depth_image[0]

def save_depth_numpy(filename, depth_image):
    depth = numpy.zeros((1, depth_image.shape[0], depth_image.shape[1]), dtype=numpy.int16)
    depth[0] = depth_image
    depth_transposed = depth.transpose(1, 2, 0)
    depth_transposed = numpy.uint16(depth_transposed)
    cv2.imwrite(filename, depth_transposed)

def save_normals(filename, tensor, scale=255.0):
    b, _, __, ___ = tensor.size()    
    for n in range(b):   
        normals = tensor[n, :, :, :].detach().cpu().numpy()
        # transpose for consistent rendering, multiplied by scale
        normals = (normals.transpose(1, 2, 0) + 1) * scale // 2 + 1
        # image write
        cv2.imwrite(filename.replace("#", str(n)), normals)

def save_phong_normals(filename, tensor):
    b, _, __, ___ = tensor.size()
    for n in range(b):            
        # the z-component data of each normal vector is retrieved 
        z_comp = tensor[n, 2, :, :].detach().cpu().numpy()
        # phong image (1, z_comp.shape[0], z_comp.shape[1]) is initialized
        phong = numpy.zeros((1, z_comp.shape[0], z_comp.shape[1]))
        # z value is inversed to paint properly based on misalignment from camera's FOV direction 
        phong[0,:,:] = 1 - z_comp
        # get max uint16 value
        iui16 = numpy.iinfo(numpy.uint16)
        scale = iui16.max
        # transpose for consistent rendering
        phong = phong.transpose(1, 2, 0) * scale
        # to unsigned int16 
        phong = numpy.uint16(phong)
        # image write
        cv2.imwrite(filename.replace("#", str(n)), phong)