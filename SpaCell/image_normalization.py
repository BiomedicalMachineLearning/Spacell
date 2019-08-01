from utils import offset_img, scale_rgb, remove_colour_cast, tile, mkdirs, spot_gen, img_cm_gen
import numpy as np
import pandas as pd
import shutil
import glob
import random
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import staintools
from staintools import stain_normalizer, LuminosityStandardizer
from staintools import ReinhardColorNormalizer
# from multiprocessing import Pool
import PIL
from PIL import Image, ImageDraw
from config import *

Image.MAX_IMAGE_PIXELS = None


def do_tile(input_task):
    sample, img_path, cm_path = input_task
    if ATM_PATH:
        atm_file = open(ATM_PATH, "r")
        atm = atm_file.read().split(" ")
    else:
        atm = None
    img = Image.open(img_path)
    
    # img normalization
    img_uncast = remove_colour_cast(img)
    img_std = LuminosityStandardizer.standardize(np.array(img_uncast))
    transformed = normalizer.transform(img_std)
    img = Image.fromarray(transformed)
    
    cm = pd.read_csv(cm_path, header=0, sep='\t', index_col=0)
    cm = cm.transpose()
    spots_center_gen = spot_gen(cm)
    tile_out = os.path.join(TILE_PATH, sample)
    mkdirs(tile_out)
    tile(img, spots_center_gen, tile_out, atm)


if __name__ == '__main__':
    template = Image.open(TEMPLATE_IMG)
    normalizer = staintools.StainNormalizer(method=NORM_METHOD)
    template_std = LuminosityStandardizer.standardize(np.array(template))
    normalizer.fit(template_std)
    meta_data = pd.read_csv(META_PATH, header=0, sep='\t')
    sample_name = list(meta_data.loc[:, SAMPLE_COLUMN])
    # input_list = []
    for item in img_cm_gen(IMG_PATH, CM_PATH, sample_name):
        # input_list.append((sample, img_path, cm_path))
        do_tile(item)
    # pool = Pool()
    # pool.map(do_tile, input_list)
    # pool.close()

