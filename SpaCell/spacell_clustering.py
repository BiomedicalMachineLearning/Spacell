import os
import pandas as pd
from optparse import OptionParser
from config import *
from utils import mkdirs, spot_gen, tile_gen, parseAlignmentMatrix, k_means, scatter_plot
from model import ResNet, features_gen, autoencoder, combine_ae



if __name__ == '__main__':
    parser = OptionParser(usage='Usage: %prog [options] <slide>')
    parser.add_option('-i', '--image_path', metavar='PATH', dest='image',
                      help='path to image')
    parser.add_option('-l', '--tile_path', metavar='PATH', dest='tile',
                      help='path to tile directory')
    parser.add_option('-c', '--count_matrix', metavar='PATH', dest='count_matrix',
                      help='path to count matrix')
    parser.add_option('-t', '--transformation_matrix', metavar='PATH', dest='transformation_matrix',
                      default=None,
                      help='path to transformation matrix')
    parser.add_option('-e', '--epochs', metavar='INT', dest='epochs',
                      type='int', default=100,
                      help='training epochs for autoencoder')
    parser.add_option('-k', '--cluster', metavar='INT', dest='cluster',
                      type='int', default=1,
                      help='expected cluster number')

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

    mkdirs(FEATURE_PATH)
    mkdirs(CLUSTER_PATH)

    if ATM_PATH:
        atm_file = open(ATM_PATH, "r")
        atm = atm_file.read().split(" ")
    else:
        atm = None
    cm = pd.read_csv(CM_PATH, header=0, sep='\t', index_col=0)
    cm = cm.transpose()
    # spots_center_gen = spot_gen(cm)

    ####### TILE FEATURE EXTRACT #########
    model = ResNet()
    tiles = tile_gen(TILE_PATH)
    features_gen(tiles, model, FEATURE_PATH)

    ####### CLUSTERING #########

    df_cm = pd.read_csv(CM_PATH, header=0, sep='\t',index_col=0)
    # df_cluster = pd.read_csv(os.path.join(CLUSTER_PATH, TEST+'.csv'), header=0, index_col =0,sep=',')
    df_tfv = pd.read_csv(os.path.join(FEATURE_PATH, BASENAME)+'.tsv', header=0, sep='\t')
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

    ### AUTOENCODER ###
    data_set = {"gene_expression_data": cm_val,
                "tile_feature_vector": tfv_val}
    model_dic = {"normal_autoencoder": autoencoder,
                 "combined_autoencoder": combine_ae}
    bottleneck_representation = None
    data = ''
    for model in model_dic:
        print("start training model {}".format(model))
        if model == "combined_autoencoder":
            data = 'both_dataset'
            print("using {}".format(data))
            data_list = list(data_set.values())
            ae, encoder = model_dic[model](data_list[0].shape[1], data_list[1].shape[1])
            ae.fit(data_list, data_list, batch_size=32, epochs=EPOCHS, validation_split=0.2, shuffle=True)
            bottleneck_representation = encoder.predict(data_list)
            cluster = k_means(bottleneck_representation, CLUSTER)
            print("generate scatter plot")
            scatter_plot(x_points, y_points, colors=cluster, alignment=atm, image=IMG_PATH,
                         output=os.path.join(CLUSTER_PATH, '{}_{}'.format(model, data)))
        else:
            for data in data_set:
                print("using {}".format(data))
                ae, encoder = model_dic[model](data_set[data].shape[1])
                ae.fit(data_set[data], data_set[data], batch_size=32, epochs=EPOCHS, validation_split=0.2, shuffle=True)
                bottleneck_representation = encoder.predict(data_set[data])
                cluster = k_means(bottleneck_representation, CLUSTER)
                print("generate scatter plot")
                scatter_plot(x_points, y_points, colors=cluster, alignment=atm, image=IMG_PATH,
                             output=os.path.join(CLUSTER_PATH, '{}_{}'.format(model, data)))