"""

python spacell_clustering.py \
-i /Users/xiao.tan/st_prostate/aligned_images/150210_1000L2_CN81_C2_P4.2_HE_EB.jpg \
-t /Users/xiao.tan/st_prostate/tile/150210_1000L2_CN81_C2_P4.2_HE_EB \
-c /Users/xiao.tan/st_prostate/adjusted_matrics/P4.2.tsv \
-a /Users/xiao.tan/st_prostate/tm/transformation_matrix150210_1000L2_CN81_C2_P4.2_Cy3_EB.txt \
-e 1 \
-k 2 \
-m ResNet50 \
-p -s -v -o /Users/xiao.tan/st_prostate/test

"""


import os
import numpy as np
import pandas as pd
from optparse import OptionParser
from config import *
from utils import mkdirs, spot_gen, tile_gen, parseAlignmentMatrix, k_means, scatter_plot
from model import ResNet, Inception_V3, Xception_imagenet, features_gen, autoencoder, combine_ae
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale
from sklearn.feature_selection import VarianceThreshold

def run_combine_model(cm, tfv, loss="mean_squared_error"):
    ae, encoder = combine_ae(cm.shape[1], tfv.shape[1], loss=loss)
    ae.fit([cm, tfv], [cm, tfv], batch_size=32, epochs=EPOCHS)
    bottleneck_representation = encoder.predict([cm, tfv])
    cluster = k_means(bottleneck_representation, CLUSTER)
    return cluster


def run_single_model(val, loss="mean_squared_error"):
    ae, encoder = autoencoder(val.shape[1], loss=loss)
    ae.fit(val, val, batch_size = 32, epochs = EPOCHS)
    bottleneck_representation = encoder.predict(val)
    cluster = k_means(bottleneck_representation, CLUSTER)
    return cluster, bottleneck_representation


def save_label(x_points, y_points, cluster, out_path):
    x_y_cluster = pd.DataFrame({'spot_x':x_points, 'spot_y':y_points, 'pred_colour':cluster})
    x_y_cluster.to_csv(out_path + '.tsv', header=True, index=False, sep="\t")


if __name__ == '__main__':
    parser = OptionParser(usage='Usage: %prog [options] <slide>')
    parser.add_option('-i', '--image_path', metavar='PATH', dest='image',
                      help='path to image')
    parser.add_option('-t', '--tile_path', metavar='PATH', dest='tile',
                      help='path to tile directory')
    parser.add_option('-c', '--count_matrix', metavar='PATH', dest='count_matrix',
                      help='path to count matrix')
    parser.add_option('-a', '--transformation_matrix', metavar='PATH', dest='transformation_matrix',
                      default=None,
                      help='path to affine transformation matrix')
    parser.add_option('-e', '--epochs', metavar='INT', dest='epochs',
                      type='int', default=100,
                      help='training epochs for autoencoder')
    parser.add_option('-k', '--cluster', metavar='INT', dest='cluster',
                      type='int', default=1,
                      help='expected cluster number')
    parser.add_option('-m', '--model', metavar='STR', dest='model',
                      type='str', default="ResNet50",
                      help='model for feature extractor. eg. ResNet50, Xception, InceptionV3')
    parser.add_option('-p', '--pca', dest='pca',
                      action="store_true", default=False,
                      help='PCA dimension reduction for input')
    parser.add_option('-s', '--scale', dest='scale',
                      action="store_true", default=False,
                      help='min max scale for input')
    parser.add_option('-v', '--var', dest='var',
                      action="store_true", default=False,
                      help='only use top 2048 variational genes')
    parser.add_option('-l', '--loss', metavar='STR', dest='loss',
                      type='str', default="binary_crossentropy",
                      help='loss function for autoencoder. eg. kullback_leibler_divergence, mean_squared_error, binary_crossentropy')
    parser.add_option('-o', '--output', metavar='PATH', dest='output',
                      help='output directory')

    (opts, args) = parser.parse_args()

    IMG_PATH = opts.image
    CM_PATH = opts.count_matrix
    ATM_PATH = opts.transformation_matrix
    OUT_PATH = opts.output
    TILE_PATH = opts.tile
    FEATURE_PATH = os.path.join(opts.output, 'feature')
    EPOCHS = opts.epochs
    CLUSTER = opts.cluster
    BASENAME = os.path.split(TILE_PATH)[-1]
    CLUSTER_PATH = os.path.join(os.path.join(opts.output, 'cluster'), BASENAME)
    MODEL = opts.model
    PCA_ = opts.pca
    VAR_ = opts.var
    SCALE_ = opts.scale
    LOSS = opts.loss

    ####### TILE FEATURE EXTRACT #########
    if MODEL == "ResNet50":
        model = ResNet()
    elif MODEL == "Xception":
        model = Xception_imagenet()
    elif MODEL == "InceptionV3":
        model = Inception_V3()
    else:
        raise NameError("Model should only be ResNet50, Xception or InceptionV3")

    tiles = tile_gen(TILE_PATH)
    FEATURE_PATH_MODEL = FEATURE_PATH + '_' + model.__name__
    CLUSTER_PATH_MODEL = os.path.join(CLUSTER_PATH, model.__name__)
    mkdirs(FEATURE_PATH_MODEL)
    mkdirs(CLUSTER_PATH_MODEL)
    features_gen(tiles, model, FEATURE_PATH_MODEL)

    ####### CLUSTERING #########

    df_cm = pd.read_csv(CM_PATH, header=0, sep='\t',index_col=0)
    # df_cluster = pd.read_csv(os.path.join(CLUSTER_PATH, TEST+'.csv'), header=0, index_col =0,sep=',')
    df_tfv = pd.read_csv(os.path.join(FEATURE_PATH_MODEL, BASENAME)+'.tsv', header=0, sep='\t')
    df_tfv_t = df_tfv.transpose()
    print("find {} spots in count matrix".format(len(df_cm.index.values.tolist())))
    print("find {} spots in tile feature".format(len(df_tfv_t.index.values.tolist())))
    assert len(df_tfv_t.index.values.tolist()) == len(df_cm.index.values.tolist())
    cm_spot = list(df_cm.index.values)
    # df_cluster = df_cluster.reindex(cm_spot)
    df_tfv_t = df_tfv_t.reindex(cm_spot)
    # cluster_info = list(df_cluster['sc3_4_clusters'].values)
    cm_val = df_cm.values
    tfv_val = df_tfv_t.values
    if ATM_PATH:
        atm = parseAlignmentMatrix(ATM_PATH)
    else:
        atm = None
    x_points = [float(i.split('x')[0]) for i in df_cm.index.values.tolist()]
    y_points = [float(i.split('x')[1]) for i in df_cm.index.values.tolist()]
    preprocessing = ""
    if VAR_:
        # 2048 tile features, 2048 top variance gene from ~17k genes
        selector = VarianceThreshold(threshold=0.0)
        cm_val = selector.fit_transform(cm_val)[:, :2048]
        preprocessing += 'top_var_gene_'
    if PCA_:
        # 501 PCs from tile features, 501 PCs from 17k genes
        cm_val = PCA().fit_transform(cm_val)
        tfv_val = PCA().fit_transform(tfv_val)
        preprocessing += "pca_"
    if SCALE_:
        # 2048 tile features, ~17k genes scaled to (0, 1)
        cm_val = minmax_scale(cm_val, feature_range=(0, 1), axis=0, copy=True)
        tfv_val = minmax_scale(tfv_val, feature_range=(0, 1), axis=0, copy=True)
        preprocessing += "scale_"

    if len(preprocessing) == 0:
        preprocessing == "no_preprocessing"
    else:
        preprocessing = preprocessing[0:-1]
    # combined model
    cluster = run_combine_model(cm_val, tfv_val, loss=LOSS)
    out_path = os.path.join(CLUSTER_PATH_MODEL, '{}-{}-{}-{}'
                            .format("combined_model", "both_data", LOSS, preprocessing))
    scatter_plot(x_points, y_points, colors=cluster, alignment=atm, image=IMG_PATH, output=out_path)
    save_label(x_points, y_points, cluster, out_path)
    # single model for gene counts
    cluster, br_gene = run_single_model(cm_val, loss=LOSS)
    out_path = os.path.join(CLUSTER_PATH_MODEL, '{}-{}-{}-{}'
                            .format("single_model", "gene_count", LOSS, preprocessing))
    scatter_plot(x_points, y_points, colors=cluster, alignment=atm, image=IMG_PATH, output=out_path)
    save_label(x_points, y_points, cluster, out_path)
    # single model for tile feature
    cluster, br_tile = run_single_model(tfv_val, loss=LOSS)
    out_path = os.path.join(CLUSTER_PATH_MODEL, '{}-{}-{}-{}'
                            .format("single_model", "tile_feature", LOSS, preprocessing))
    scatter_plot(x_points, y_points, colors=cluster, alignment=atm, image=IMG_PATH, output=out_path)
    save_label(x_points, y_points, cluster, out_path)
    # combine two latent spaces
    br_gene_scaled = minmax_scale(br_gene, feature_range=(0, 1), axis=0, copy=True)
    br_tile_scaled = minmax_scale(br_tile, feature_range=(0, 1), axis=0, copy=True)
    br_combine_scaled = np.concatenate((br_gene_scaled, br_tile_scaled), axis=1)
    cluster = k_means(br_combine_scaled, CLUSTER)
    out_path = os.path.join(CLUSTER_PATH_MODEL, '{}-{}-{}-{}'
                            .format("cmobine_latent_space", "both_data", LOSS, preprocessing))
    scatter_plot(x_points, y_points, colors=cluster, alignment=atm, image=IMG_PATH, output=out_path)
    save_label(x_points, y_points, cluster, out_path)