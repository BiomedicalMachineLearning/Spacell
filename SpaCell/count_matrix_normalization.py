import pandas as pd
import os
import glob
import numpy as np
from config import *


def add_label(dataframe, label, meta):
    label_list = []
    for spot in dataframe.index.values:
        sample_id = '_'.join(spot.split('_')[:-1])
        spot_label = meta.loc[sample_id, label]
        label_list.append(spot_label)
    dataframe[label] = label_list
    return dataframe



if __name__ == '__main__':
    meta_mouse = pd.read_csv(META_PATH, header=0, sep='\t', index_col=0)
    sample_name = list(meta_mouse.index)
    total_counts = pd.DataFrame()
    for file in glob.glob(CM_PATH+'*.txt'):
        sample_n = '_'.join(os.path.basename(file).split("_")[0:-4])
        if sample_n in sample_name:
            cm = pd.read_csv(file, header = 0,sep='\t', index_col=0)  # column:genes row:spots
            # reindex
            new_spots = ["{0}_{1}".format(sample_n, spot) for spot in cm.index]
            cm.index = new_spots
            total_counts = total_counts.append(cm, sort=False)

    # replace missing values with 0
    total_counts.replace([np.inf, -np.inf], np.nan)
    total_counts.fillna(0.0, inplace=True)

    num_spots = len(total_counts.index)
    num_genes = len(total_counts.columns)

    # Remove low quality spots
    min_genes_spot = round((total_counts != 0).sum(axis=1).quantile(THRESHOLD_SPOT))
    print("Number of expressed genes a spot must have to be kept ({}% of total expressed genes) {}".format(THRESHOLD_SPOT, min_genes_spot))
    total_counts = total_counts[(total_counts != 0).sum(axis=1) >= min_genes_spot]
    print("Dropped {} spots".format(num_spots - len(total_counts.index)))

    # Spots are columns and genes are rows
    total_counts = total_counts.transpose()

    # Remove low quality genes
    min_spots_gene = round(len(total_counts.columns) * THRESHOLD_GENE)
    print("Removing genes that are expressed in less than {} spots with a count of at least {}".format(min_spots_gene, MIN_EXP))
    total_counts = total_counts[(total_counts >= MIN_EXP).sum(axis=1) >= min_spots_gene]
    print("Dropped {} genes".format(num_genes - len(total_counts.index)))

    total_counts = total_counts.transpose()

    # normalization
    row_sum = total_counts.sum(axis=1)
    normal_total_counts = total_counts.div(row_sum, axis=0)

    # add label
    normal_total_counts = add_label(normal_total_counts, LABEL_COLUMN , meta_mouse)
    if CONDITION_COLUMN:
        normal_total_counts = add_label(normal_total_counts, CONDITION_COLUMN, meta_mouse)
    normal_total_counts.to_csv(os.path.join(DATASET_PATH,'cm_norm.tsv'), sep='\t')

