import os
import numpy as np
import collections
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import transforms
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
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


def parseAlignmentMatrix(alignment_file):
    alignment_matrix = np.identity(3)
    with open(alignment_file, "r") as filehandler:
        line = filehandler.readline()
        tokens = line.split()
        assert(len(tokens) == 9)
        alignment_matrix[0,0] = float(tokens[0])
        alignment_matrix[1,0] = float(tokens[1])
        alignment_matrix[2,0] = float(tokens[2])
        alignment_matrix[0,1] = float(tokens[3])
        alignment_matrix[1,1] = float(tokens[4])
        alignment_matrix[2,1] = float(tokens[5])
        alignment_matrix[0,2] = float(tokens[6])
        alignment_matrix[1,2] = float(tokens[7])
        alignment_matrix[2,2] = float(tokens[8])
    return alignment_matrix


def scatter_plot(x_points, y_points, output=None, colors=None,
                 alignment=None, cmap=None, title='Scatter', xlabel='X',
                 ylabel='Y', image=None, alpha=1.0, size=10, vmin=None, vmax=None):
    # Plot spots with the color class in the tissue image
    fig, a = plt.subplots()
    base_trans = a.transData
    # Extend (left, right, bottom, top)
    # The location, in data-coordinates, of the lower-left and upper-right corners.
    # If None, the image is positioned such that the pixel centers fall on zero-based (row, column) indices.
    extent_size = [1, 33, 35, 1]
    # If alignment is None we re-size the image to chip size (1,1,33,35)
    # Otherwise we keep the image intact and apply the 3x3 transformation
    if alignment is not None and not np.array_equal(alignment, np.identity(3)):
        base_trans = transforms.Affine2D(matrix=alignment) + base_trans
        extent_size = None

    # Create the scatter plot
    sc = a.scatter(x_points, y_points, c=colors, edgecolor="none",
                   cmap=cmap, s=size, transform=base_trans, alpha=alpha,
                   vmin=vmin, vmax=vmax)
    # Plot the image
    if image is not None and os.path.isfile(image):
        img = plt.imread(image)
        a.imshow(img, extent=extent_size)
    # Add labels and title
    a.set_xlabel(xlabel)
    a.set_ylabel(ylabel)
    a.set_title(title, size=10)
    if output is not None:
        fig.savefig("{}.pdf".format(output),
                    format='pdf', dpi=180)
    else:
        fig.show()


def plot_latent(bottleneck_representation, classes):
    plt.scatter(bottleneck_representation[:, 0], bottleneck_representation[:, 1], c = classes, cmap = 'tab20', s = 10)
    plt.legend()
    plt.title('Autoencoder')
    plt.xlabel("latnet_1")
    plt.ylabel("latnet_2")


def plot_tsne(bottleneck_representation, cluster_info):
    model_tsne_auto = TSNE(learning_rate = 200, n_components = 2, random_state = 123,
                       perplexity = 90, n_iter = 100, verbose = 1)
    tsne_auto = model_tsne_auto.fit_transform(bottleneck_representation)
    plt.scatter(tsne_auto[:, 0], tsne_auto[:, 1],c=cluster_info, cmap = 'tab20', s = 10)
    plt.title('tSNE on Autoencoder')
    plt.xlabel("tSNE1")
    plt.ylabel("tSNE2")


def pca_tsne_plot(input_x, classes, n_pc=20):
    pc = PCA(n_components=n_pc).fit_transform(input_x)
    model_tsne = TSNE(learning_rate=200, n_components=2, random_state=123,
                      perplexity=90, n_iter=1000, verbose=1)
    tsne = model_tsne.fit_transform(pc)
    plt.scatter(tsne[:, 0], tsne[:, 1], c=classes, cmap='Set1', s=10)
    plt.title('tSNE on PCA')
    plt.xlabel("tSNE1")
    plt.ylabel("tSNE2")


def k_means(input_x, n_cluster):
    y_pred = KMeans(init='k-means++', n_clusters=n_cluster, n_init=20, max_iter=1000).fit_predict(input_x)
    counter = collections.Counter(y_pred)
    sorted_counter = [i[0] for i in sorted(counter.items(), key=lambda x: x[1], reverse=True)]
    color = [color_map[j] for j in [sorted_counter.index(i) for i in y_pred]]
    return color


def tile_gen(tile_path):
    file_name = []
    for tile_root, _, tile_files in os.walk(tile_path):
        for tile_file in tile_files:
            if tile_file.endswith(".jpeg"):
                tile = Image.open(os.path.join(tile_root, tile_file))
                tile = np.asarray(tile, dtype="int32")
                tile = tile.astype(np.float32)
                tile = np.stack([tile])
                img_name, coordx, coordy = os.path.splitext(tile_file)[0].split("-")
                file_name.append((img_name, coordx, coordy))
                yield (tile, (img_name, coordx, coordy))





