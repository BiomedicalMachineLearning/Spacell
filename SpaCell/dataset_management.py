import glob
import pandas as pd
import os
from config import *


if __name__ == '__main__':
    cm = pd.read_csv(os.path.join(DATASET_PATH, 'cm_norm.tsv'), header=0, sep='\t', index_col=0)
    if CONDITION_COLUMN:
        cm_ = cm.loc[cm[CONDITION_COLUMN] == CONDITION]
        cm = cm_
    col_cm = list(cm.index)
    img_files = glob.glob(TILE_PATH+'/*/*.jpeg')
    sorted_img = []
    sorted_cm = []
    for img in img_files:
        id_img = os.path.splitext(os.path.basename(img))[0].replace("-", "_")
        for c in col_cm:
            id_c = c.replace("x", "_")
            if id_img == id_c:
                sorted_img.append(img)
                sorted_cm.append(c)

    cm = cm.reindex(sorted_cm)
    df = pd.DataFrame(data={'img':sorted_img,
        'cm':sorted_cm, 
        'label':cm[LABEL_COLUMN]})
    df.to_csv(os.path.join(DATASET_PATH, 'dataset.tsv'), sep='\t')
    cm.to_csv(os.path.join(DATASET_PATH, "cm_final.tsv"), sep='\t')
