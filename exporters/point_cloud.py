import torch
import os
import numpy

def save_ply_from_3d_numpy(filename, points):
    with open(filename, "w") as ply_file:        
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write("element vertex {}\n".format(len(points)))
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")
        # if normals is not None:
        #     ply_file.write('property float nx\n')
        #     ply_file.write('property float ny\n')
        #     ply_file.write('property float nz\n')
        ply_file.write("property uchar red\n")
        ply_file.write("property uchar green\n")
        ply_file.write("property uchar blue\n")
        ply_file.write("end_header\n")

        for p in points:
            ply_file.write("{} {} {} {} {} {}\n".format(\
                p[0], p[1], p[2],\
                "255", "255", "255"))

def save_depth_numpy(filename, depth_image):
    depth = numpy.zeros((1, depth_image.shape[0], depth_image.shape[1]), dtype=numpy.int16)
    depth[0] = depth_image
    depth_transposed = depth.transpose(1, 2, 0)
    depth_transposed = numpy.uint16(depth_transposed)
    cv2.imwrite(filename, depth_transposed)
    return depth_transposed

def save_ply_from_depth_numpy(filename, depth_image, intrinsics):
    tensor_3d = torch.zeros([1, 3, depth_image.shape[0], depth_image.shape[1]], dtype=torch.float)
    with open(filename, "w") as ply_file:        
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write("element vertex {}\n".format(depth_image.size))
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")
        # if normals is not None:
        #     ply_file.write('property float nx\n')
        #     ply_file.write('property float ny\n')
        #     ply_file.write('property float nz\n')
        ply_file.write("property uchar red\n")
        ply_file.write("property uchar green\n")
        ply_file.write("property uchar blue\n")
        ply_file.write("end_header\n")

        for y in range(depth_image.shape[0]):
            for x in range(depth_image.shape[1]):        
                x3d = (x - intrinsics[0][2]) * depth_image[y][x] / intrinsics[0][0]         
                y3d = (y - intrinsics[1][2]) * depth_image[y][x] / intrinsics[1][1]              

                x_coord = x3d
                y_coord = y3d
                z_coord = depth_image[y][x]
                
                tensor_3d[0][0][y][x] = x_coord
                tensor_3d[0][1][y][x] = y_coord
                tensor_3d[0][2][y][x] = z_coord.astype(float)

                ply_file.write("{} {} {} {} {} {}\n".format(\
                    x_coord, y_coord, z_coord,\
                    "255", "255", "255"))
        return tensor_3d


def save_ply_original(filename, tensor, scale, color='black' , normals = None):    
    b, _, h, w = tensor.size()
    for n in range(b):
        coords = tensor[n, :, :, :].detach().cpu().numpy()
        x_coords = coords[0, :] * scale
        y_coords = coords[1, :] * scale
        z_coords = coords[2, :] * scale
        if normals is not None:
            norms = normals[n, : , : , :].detach().cpu().numpy()
            nx_coords = norms[0, :]
            ny_coords = norms[1, :]
            nz_coords = norms[2, :]
        with open(filename.replace("#", str(n)), "w") as ply_file:        
            ply_file.write("ply\n")
            ply_file.write("format ascii 1.0\n")
            ply_file.write("element vertex {}\n".format(w * h))
            ply_file.write("property float x\n")
            ply_file.write("property float y\n")
            ply_file.write("property float z\n")
            if normals is not None:
                ply_file.write('property float nx\n')
                ply_file.write('property float ny\n')
                ply_file.write('property float nz\n')
            ply_file.write("property uchar red\n")
            ply_file.write("property uchar green\n")
            ply_file.write("property uchar blue\n")
            ply_file.write("end_header\n")
            
            if normals is None:
                for x in torch.arange(w):
                    for y in torch.arange(h):
                        ply_file.write("{} {} {} {} {} {}\n".format(\
                            x_coords[y, x], y_coords[y, x], z_coords[y, x],\
                            "255" if color=='red' else "0",
                            "255" if color=='green' else "0",
                            "255" if color=='blue' else "0"))
            else:
                for x in torch.arange(w):
                    for y in torch.arange(h):
                        ply_file.write("{} {} {} {} {} {} {} {} {}\n".format(\
                            x_coords[y, x], y_coords[y, x], z_coords[y, x],\
                            nx_coords[y, x], ny_coords[y, x], nz_coords[y, x],\
                            "255" if color=='red' else "0",
                            "255" if color=='green' else "0",
                            "255" if color=='blue' else "0"))

def save_ply(filename, tensor, scale, axis=None, color='black' , normals = None):    
    b, _, h, w = tensor.size()
    for n in range(b):
        if (axis == None or axis == "xyz"):
            coords = tensor[n, :, :, :].detach().cpu().numpy()
            x_coords = coords[0, :] * scale
            y_coords = coords[1, :] * scale
            z_coords = coords[2, :] * scale
            if normals is not None:
                norms = normals[n, : , : , :].detach().cpu().numpy()
                nx_coords = norms[0, :]
                ny_coords = norms[1, :]
                nz_coords = norms[2, :]
            with open(filename.replace("#", str(n)), "w") as ply_file:        
                ply_file.write("ply\n")
                ply_file.write("format ascii 1.0\n")
                ply_file.write("element vertex {}\n".format(w * h))
                ply_file.write("property float x\n")
                ply_file.write("property float y\n")
                ply_file.write("property float z\n")
                if normals is not None:
                    ply_file.write('property float nx\n')
                    ply_file.write('property float ny\n')
                    ply_file.write('property float nz\n')
                ply_file.write("property uchar red\n")
                ply_file.write("property uchar green\n")
                ply_file.write("property uchar blue\n")
                ply_file.write("end_header\n")
                
                if normals is None:
                    for x in torch.arange(w):
                        for y in torch.arange(h):
                            ply_file.write("{} {} {} {} {} {}\n".format(\
                                x_coords[y, x], y_coords[y, x], z_coords[y, x],\
                                "255" if color=='red' else "0",
                                "255" if color=='green' else "0",
                                "255" if color=='blue' else "0"))
                else:
                    for x in torch.arange(w):
                        for y in torch.arange(h):
                            ply_file.write("{} {} {} {} {} {} {} {} {}\n".format(\
                                x_coords[y, x], y_coords[y, x], z_coords[y, x],\
                                nx_coords[y, x], ny_coords[y, x], nz_coords[y, x],\
                                "255" if color=='red' else "0",
                                "255" if color=='green' else "0",
                                "255" if color=='blue' else "0"))
        elif (axis == "xzy"):
            coords = tensor[n, :, :, :].detach().cpu().numpy()
            x_coords = coords[0, :] * scale
            y_coords = coords[2, :] * scale
            z_coords = coords[1, :] * scale
            if normals is not None:
                norms = normals[n, : , : , :].detach().cpu().numpy()
                nx_coords = norms[0, :]
                ny_coords = norms[2, :]
                nz_coords = norms[1, :]
            with open(filename.replace("#", str(n)), "w") as ply_file:        
                ply_file.write("ply\n")
                ply_file.write("format ascii 1.0\n")
                ply_file.write("element vertex {}\n".format(w * h))
                ply_file.write("property float x\n")
                ply_file.write("property float y\n")
                ply_file.write("property float z\n")
                if normals is not None:
                    ply_file.write('property float nx\n')
                    ply_file.write('property float ny\n')
                    ply_file.write('property float nz\n')
                ply_file.write("property uchar red\n")
                ply_file.write("property uchar green\n")
                ply_file.write("property uchar blue\n")
                ply_file.write("end_header\n")
                
                if normals is None:
                    for x in torch.arange(w):
                        for y in torch.arange(h):
                            ply_file.write("{} {} {} {} {} {}\n".format(\
                                x_coords[y, x], y_coords[y, x], z_coords[y, x],\
                                "255" if color=='red' else "0",
                                "255" if color=='green' else "0",
                                "255" if color=='blue' else "0"))
                else:
                    for x in torch.arange(w):
                        for y in torch.arange(h):
                            ply_file.write("{} {} {} {} {} {} {} {} {}\n".format(\
                                x_coords[y, x], y_coords[y, x], z_coords[y, x],\
                                nx_coords[y, x], ny_coords[y, x], nz_coords[y, x],\
                                "255" if color=='red' else "0",
                                "255" if color=='green' else "0",
                                "255" if color=='blue' else "0"))
        elif (axis == "x-zy"):
            coords = tensor[n, :, :, :].detach().cpu().numpy()
            x_coords = coords[0, :] * scale
            y_coords = -coords[2, :] * scale
            z_coords = coords[1, :] * scale
            if normals is not None:
                norms = normals[n, : , : , :].detach().cpu().numpy()
                nx_coords = norms[0, :]
                ny_coords = norms[2, :]
                nz_coords = norms[1, :]
            with open(filename.replace("#", str(n)), "w") as ply_file:        
                ply_file.write("ply\n")
                ply_file.write("format ascii 1.0\n")
                ply_file.write("element vertex {}\n".format(w * h))
                ply_file.write("property float x\n")
                ply_file.write("property float y\n")
                ply_file.write("property float z\n")
                if normals is not None:
                    ply_file.write('property float nx\n')
                    ply_file.write('property float ny\n')
                    ply_file.write('property float nz\n')
                ply_file.write("property uchar red\n")
                ply_file.write("property uchar green\n")
                ply_file.write("property uchar blue\n")
                ply_file.write("end_header\n")
                
                if normals is None:
                    for x in torch.arange(w):
                        for y in torch.arange(h):
                            ply_file.write("{} {} {} {} {} {}\n".format(\
                                x_coords[y, x], y_coords[y, x], z_coords[y, x],\
                                "255" if color=='red' else "0",
                                "255" if color=='green' else "0",
                                "255" if color=='blue' else "0"))
                else:
                    for x in torch.arange(w):
                        for y in torch.arange(h):
                            ply_file.write("{} {} {} {} {} {} {} {} {}\n".format(\
                                x_coords[y, x], y_coords[y, x], z_coords[y, x],\
                                nx_coords[y, x], ny_coords[y, x], nz_coords[y, x],\
                                "255" if color=='red' else "0",
                                "255" if color=='green' else "0",
                                "255" if color=='blue' else "0"))


def save_ply_merged_batch(filename, tensor, scale, axis=None, color='black' , normals = None, remove_zeros=True):    
    b, _, h, w = tensor.size()

    with open(filename, "w") as ply_file:   
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write("element vertex {}\n".format(b * w * h))
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")
        if normals is not None:
            ply_file.write('property float nx\n')
            ply_file.write('property float ny\n')
            ply_file.write('property float nz\n')
        ply_file.write("property uchar red\n")
        ply_file.write("property uchar green\n")
        ply_file.write("property uchar blue\n")
        ply_file.write("end_header\n")                

        for n in range(b):
            if (axis == None or axis == "xyz"):
                coords = tensor[n, :, :, :].detach().cpu().numpy()
                x_coords = coords[0, :] * scale
                y_coords = coords[1, :] * scale
                z_coords = coords[2, :] * scale

                if (remove_zeros and numpy.sum(coords) > 0):
                    if normals is not None:
                        norms = normals[n, : , : , :].detach().cpu().numpy()
                        nx_coords = norms[0, :]
                        ny_coords = norms[1, :]
                        nz_coords = norms[2, :]

                    if normals is None:
                        for x in torch.arange(w):
                            for y in torch.arange(h):
                                ply_file.write("{} {} {} {} {} {}\n".format(\
                                    x_coords[y, x], y_coords[y, x], z_coords[y, x],\
                                    "255" if color=='red' else "0",
                                    "255" if color=='green' else "0",
                                    "255" if color=='blue' else "0"))
                    else:
                        for x in torch.arange(w):
                            for y in torch.arange(h):
                                ply_file.write("{} {} {} {} {} {} {} {} {}\n".format(\
                                    x_coords[y, x], y_coords[y, x], z_coords[y, x],\
                                    nx_coords[y, x], ny_coords[y, x], nz_coords[y, x],\
                                    "255" if color=='red' else "0",
                                    "255" if color=='green' else "0",
                                    "255" if color=='blue' else "0"))
            elif (axis == "xzy"):
                coords = tensor[n, :, :, :].detach().cpu().numpy()
                x_coords = coords[0, :] * scale
                y_coords = coords[2, :] * scale
                z_coords = coords[1, :] * scale

                if (remove_zeros and numpy.sum(coords) > 0):
                    if normals is not None:
                        norms = normals[n, : , : , :].detach().cpu().numpy()
                        nx_coords = norms[0, :]
                        ny_coords = norms[2, :]
                        nz_coords = norms[1, :]
                        
                    if normals is None:
                        for x in torch.arange(w):
                            for y in torch.arange(h):
                                ply_file.write("{} {} {} {} {} {}\n".format(\
                                    x_coords[y, x], y_coords[y, x], z_coords[y, x],\
                                    "255" if color=='red' else "0",
                                    "255" if color=='green' else "0",
                                    "255" if color=='blue' else "0"))
                    else:
                        for x in torch.arange(w):
                            for y in torch.arange(h):
                                ply_file.write("{} {} {} {} {} {} {} {} {}\n".format(\
                                    x_coords[y, x], y_coords[y, x], z_coords[y, x],\
                                    nx_coords[y, x], ny_coords[y, x], nz_coords[y, x],\
                                    "255" if color=='red' else "0",
                                    "255" if color=='green' else "0",
                                    "255" if color=='blue' else "0"))
                        
            elif (axis == "x-zy"):
                coords = tensor[n, :, :, :].detach().cpu().numpy()
                x_coords = coords[0, :] * scale
                y_coords = -coords[2, :] * scale
                z_coords = coords[1, :] * scale
                if normals is not None:
                    norms = normals[n, : , : , :].detach().cpu().numpy()
                    nx_coords = norms[0, :]
                    ny_coords = norms[2, :]
                    nz_coords = norms[1, :]                
                    
                if normals is None:
                    for x in torch.arange(w):
                        for y in torch.arange(h):
                            ply_file.write("{} {} {} {} {} {}\n".format(\
                                x_coords[y, x], y_coords[y, x], z_coords[y, x],\
                                "255" if color=='red' else "0",
                                "255" if color=='green' else "0",
                                "255" if color=='blue' else "0"))
                else:
                    for x in torch.arange(w):
                        for y in torch.arange(h):
                            ply_file.write("{} {} {} {} {} {} {} {} {}\n".format(\
                                x_coords[y, x], y_coords[y, x], z_coords[y, x],\
                                nx_coords[y, x], ny_coords[y, x], nz_coords[y, x],\
                                "255" if color=='red' else "0",
                                "255" if color=='green' else "0",
                                "255" if color=='blue' else "0"))

