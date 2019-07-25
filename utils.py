import os
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from config import *

def spot_gen(cm):
    for spot in [x.split('x') for x in cm]:
        x_point = spot[0]
        y_point = spot[1]
        yield x_point, y_point


def img_cm_atm_gen(img_path, cm_path, atm_path):
    for cm_root, _, cm_files in os.walk(cm_path):
        for cm_file in cm_files:
            if cm_file.endswith(".tsv"):
                pattern = ".".join(cm_file.split(".")[0:-1])
                for img_root, _, img_files in os.walk(img_path):
                    for img_file in img_files:
                        if pattern in img_file:
                            for atm_root, _, atm_files in os.walk(atm_path):
                                for atm_file in atm_files:
                                    if pattern in atm_file and atm_file.startswith("transformation_matrix"):
                                        yield (os.path.join(img_root, img_file),
                                               os.path.join(cm_root, cm_file),
                                               os.path.join(atm_root, atm_file),
                                               pattern)


def img_cm_gen(img_path, cm_path, sample_name):
    for sample in sample_name:
        for cm_root, _, cm_files in os.walk(cm_path):
            for cm_file in cm_files:
                if cm_file.endswith(".txt") and cm_file.startswith(sample):
                    pattern = "_".join(sample.split("_")[0:2])
                    for img_root, _, img_files in os.walk(img_path):
                        for img_file in img_files:
                            if img_file.endswith(".jpg") and img_file.startswith(pattern):
                                assert "_".join(img_file.split("_")[0:2]) == "_".join(cm_file.split("_")[0:2])
                                yield (sample, os.path.join(img_path, img_file), os.path.join(cm_path, cm_file))


def offset_img(img, r_offset, g_offset, b_offset):
    new_img = img.copy()
    pixels = new_img.load()
    for i in range(img.size[0]):   #For every column
        for j in range(img.size[1]):    #For every row
            r, g, b = pixels[i,j]
            new_r, new_g, new_b = r+r_offset, g+g_offset, b+b_offset
            pixels[i,j] = int(new_r), int(new_g), int(new_b)
    return new_img 


"""
def scale_rgb(img, r_scale, g_scale, b_scale):
    new_img = img.copy()
    pixels = new_img.load()
    for i in range(img.size[0]):   #For every column
        for j in range(img.size[1]):    #For every row
            r, g, b = pixels[i,j]
            new_r, new_g, new_b = r*r_scale, g*g_scale, b*b_scale
            pixels[i,j] = int(new_r), int(new_g), int(new_b)
    return new_img  

"""


def scale_rgb(img, r_scale, g_scale, b_scale):
    source = img.split()
    R, G, B = 0, 1, 2
    red = source[R].point(lambda i:i*r_scale)
    green = source[G].point(lambda i:i*g_scale)
    blue = source[B].point(lambda i:i*b_scale)
    return Image.merge('RGB', [red, green, blue])


def remove_colour_cast(img):
    img = img.convert('RGB')
    img_array = np.array(img)
    #Calculate 99th percentile pixels values for each channel
    rp = np.percentile(img_array[:,:,0].ravel(), q = 99)
    gp = np.percentile(img_array[:,:,1].ravel(), q = 99)
    bp = np.percentile(img_array[:,:,2].ravel(), q = 99)
    #scale image based on percentile values
    return scale_rgb(img, 255/rp, 255/gp, 255/bp)


def tile(img, spots_center_gen, out_dir, atm):
    sample = os.path.split(out_dir)[-1]
    for x_coord, y_coord in spots_center_gen:
        if atm:
            x_pixel = float(x_coord) * float(atm[0]) + float(atm[6])
            y_pixel = float(y_coord) * float(atm[4]) + float(atm[7])
            x_0 = x_pixel - float(atm[0]) * 0.8 / 2
            y_0 = y_pixel - float(atm[4]) * 0.8 / 2
            x_1 = x_pixel + float(atm[0]) * 0.8 / 2
            y_1 = y_pixel + float(atm[4]) * 0.8 / 2
        else:
            unit_x = float(img.size[0]) / 32
            unit_y = float(img.size[0]) / 34
            x_pixel = float(x_coord) * unit_x
            y_pixel = float(y_coord) * unit_y
            x_0 = x_pixel - unit_x * 0.8 / 2
            y_0 = y_pixel - unit_y * 0.8 / 2
            x_1 = x_pixel + unit_x * 0.8 / 2
            y_1 = y_pixel + unit_y * 0.8 / 2
        tile = img.crop((x_0, y_0, x_1, y_1))
        tile.thumbnail(SIZE, Image.ANTIALIAS)
        tile_name = str(sample) + '-' + str(x_coord) + '-' + str(y_coord)
        print("generate tile of sample {} at spot {}x{}".format(str(sample), str(x_coord), str(y_coord)))
        tile.save(os.path.join(out_dir, tile_name + '.jpeg'), 'JPEG')


def mkdirs(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)



