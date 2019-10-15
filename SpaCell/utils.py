import os
import numpy as np
import collections
import matplotlib
from scipy import interp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
from matplotlib import transforms
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
from config import *
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input as preprocess_resnet
from tensorflow import keras
from tensorflow.keras.utils import to_categorical


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
    for i in range(img.size[0]):  # For every column
        for j in range(img.size[1]):  # For every row
            r, g, b = pixels[i, j]
            new_r, new_g, new_b = r + r_offset, g + g_offset, b + b_offset
            pixels[i, j] = int(new_r), int(new_g), int(new_b)
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
    red = source[R].point(lambda i: i * r_scale)
    green = source[G].point(lambda i: i * g_scale)
    blue = source[B].point(lambda i: i * b_scale)
    return Image.merge('RGB', [red, green, blue])


def remove_colour_cast(img):
    img = img.convert('RGB')
    img_array = np.array(img)
    # Calculate 99th percentile pixels values for each channel
    rp = np.percentile(img_array[:, :, 0].ravel(), q=99)
    gp = np.percentile(img_array[:, :, 1].ravel(), q=99)
    bp = np.percentile(img_array[:, :, 2].ravel(), q=99)
    # scale image based on percentile values
    return scale_rgb(img, 255 / rp, 255 / gp, 255 / bp)


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
        assert (len(tokens) == 9)
        alignment_matrix[0, 0] = float(tokens[0])
        alignment_matrix[1, 0] = float(tokens[1])
        alignment_matrix[2, 0] = float(tokens[2])
        alignment_matrix[0, 1] = float(tokens[3])
        alignment_matrix[1, 1] = float(tokens[4])
        alignment_matrix[2, 1] = float(tokens[5])
        alignment_matrix[0, 2] = float(tokens[6])
        alignment_matrix[1, 2] = float(tokens[7])
        alignment_matrix[2, 2] = float(tokens[8])
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
    plt.scatter(bottleneck_representation[:, 0], bottleneck_representation[:, 1], c=classes, cmap='tab20', s=10)
    plt.legend()
    plt.title('Autoencoder')
    plt.xlabel("latnet_1")
    plt.ylabel("latnet_2")


def plot_tsne(bottleneck_representation, cluster_info):
    model_tsne_auto = TSNE(learning_rate=200, n_components=2, random_state=123,
                           perplexity=90, n_iter=100, verbose=1)
    tsne_auto = model_tsne_auto.fit_transform(bottleneck_representation)
    plt.scatter(tsne_auto[:, 0], tsne_auto[:, 1], c=cluster_info, cmap='tab20', s=10)
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


def find_sample_name(pd_index):
    return "_".join(pd_index.split("_")[:-1])


def save_result(result_list, out_path):
    result_np = np.array(result_list)
    np.save(out_path + ".npy", result_np)
    return result_np


def cv_roc_plot(total_actual, total_predicted, class_list, prefix=""):
    tprs = []
    base_fpr = np.linspace(0, 1, 101)
    plt.figure(figsize=(12, 10))
    color = ['blue', 'green', 'red', 'cyan']
    for i, class_ in enumerate(class_list):
        for j in range(len(total_actual)):
            y_true = total_actual[j]
            y_pred = total_predicted[j]
            y_true_onehot = np.zeros((len(y_true), len(class_list)))
            y_true_onehot[np.arange(len(y_true)), y_true] = 1
            fpr, tpr, thresholds = roc_curve(y_true_onehot[:, i], y_pred[:, i])
            plt.plot(fpr, tpr, 'b', alpha=0.1, color=color[i])
            tpr = interp(base_fpr, fpr, tpr)
            tpr[0] = 0.0
            tprs.append(tpr)
        tprs = np.array(tprs)
        mean_tprs = tprs.mean(axis=0)
        std = tprs.std(axis=0)
        tprs_upper = np.minimum(mean_tprs + std, 1)
        tprs_lower = mean_tprs - std
        roc_auc = auc(base_fpr, mean_tprs)
        plt.plot(base_fpr, mean_tprs, color=color[i], label='%s (area = %0.2f)' % (class_list[i], roc_auc))
        plt.fill_between(base_fpr, tprs_lower, tprs_upper, color=color[i], alpha=0.1)
        tprs = []
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.ylabel('True Positive Rate', fontsize=20)
    plt.xlabel('False Positive Rate', fontsize=20)
    # plt.axes().set_aspect('equal', 'datalim')
    plt.legend(loc='lower right', fontsize=20)
    # plt.show()
    plt.savefig('{}_cv_roc.pdf'.format(prefix))


def plot_confusion_matrix_cv(cm_list, classes=None, prefix=""):
    cm = np.stack(cm_list)
    cm_mean = cm.mean(axis=(0))
    cm_std = cm.std(axis=(0))
    # print(cm.shape)
    # Normalise Confusion Matrix by dividing each value by the sum of that row
    cm_std = cm_std.astype('float') / cm_mean.sum(axis=1)[:, np.newaxis]
    cm_mean = cm_mean.astype('float') / cm_mean.sum(axis=1)[:, np.newaxis]
    labels = (np.asarray(["%.2f±%.2f" % (mean, std)
                          for mean, std in zip(cm_mean.flatten(),
                                               cm_std.flatten())])).reshape(4, 4)
    # Make DataFrame from Confusion Matrix and classes
    cm_df = pd.DataFrame(cm_mean, index=classes, columns=classes)
    # Display Confusion Matrix
    plt.figure(figsize=(4, 4), dpi=300)
    cm_plot = sns.heatmap(cm_df, vmin=0, vmax=1, annot=labels, fmt='s', cmap='Blues', square=True,
                          annot_kws={"size": 7})
    plt.title('Confusion Matrix', fontsize=12)
    # Display axes labels
    plt.ylabel('True label', fontsize=12)
    plt.xlabel('Predicted label', fontsize=12)
    plt.savefig('{}_cv_confusion_matrx.pdf'.format(prefix))
    plt.tight_layout()
    return cm_plot


def add_plot(np_array, shape, name, color):
    base = np.linspace(0, len(np_array[0]), len(np_array[0]))
    for i in range(len(np_array)):
        plt.plot(np_array[i], ls=shape, color=color, alpha=0.1)
    mean = np_array.mean(axis=0)
    std = np_array.std(axis=0)
    upper = mean + std
    lower = mean - std
    plt.plot(mean, ls=shape, color=color, label='mean %s' % (name))
    plt.fill_between(base, lower, upper, color=color, alpha=0.1)


def add_test_plot(np_array, shape, name, color, test_acc):
    base = np.linspace(0, len(np_array[0]), len(np_array[0]))
    for i in range(len(np_array)):
        plt.plot(np_array[i], ls=shape, color=color[i], alpha=0.2, label="round: %s val_acc: %0.2f test_acc: %0.2f"
                                                                         % (i + 1, np_array[i][-1], test_acc[i]))
    mean = np_array.mean(axis=0)
    std = np_array.std(axis=0)
    upper = mean + std
    lower = mean - std
    plt.plot(mean, ls=shape, color="red", label='overall %s: %0.2f±%0.2f \n test_acc: %0.2f±%0.2f'
                                                % (name, mean[-1], std[-1], np.array(test_acc).mean(),
                                                   np.array(test_acc).std()))
    plt.fill_between(base, lower, upper, color="red", alpha=0.1)


def calculate_accuracy(total_actual, total_predicted, class_list):
    acc_list = []
    for j in range(len(total_actual)):
        y_true = total_actual[j]
        y_pred = total_predicted[j]
        y_true_onehot = np.zeros((len(y_true), len(class_list)))
        y_true_onehot[np.arange(len(y_true)), y_true] = 1
        y_pred_int = np.argmax(y_pred, axis=1)
        acc = accuracy_score(y_true, y_pred_int)
        acc_list.append(acc)
    return acc_list


def calculate_cm(total_actual, total_predicted):
    confusion_matrix_list = []
    for j in range(len(total_actual)):
        y_true = total_actual[j]
        y_pred = total_predicted[j]
        y_pred_int = np.argmax(y_pred, axis=1)
        cm = confusion_matrix(y_true, y_pred_int)
        confusion_matrix_list.append(cm)
    return confusion_matrix_list


def learning_curve(total_train_accuracy, total_val_accuracy, acc_list, prefix=""):
    c = ['green', 'cyan', 'magenta', 'black', 'red']
    plt.figure(figsize=(15, 12))
    add_plot(total_train_accuracy, '-', 'train_accuracy', 'blue')
    add_test_plot(total_val_accuracy, '-', 'val_acc', c, acc_list)
    plt.ylabel('Accuracy', fontsize=20)
    plt.xlabel('Epoch', fontsize=20)
    plt.legend(loc='lower right', fontsize=15)
    # plt.show()
    plt.savefig('{}_cv_learning_curve.pdf'.format(prefix))


def loss_curve(total_val_loss, total_train_loss, prefix=""):
    plt.figure(figsize=(15, 12))
    add_plot(total_val_loss, '--', 'val_loss', 'red')
    add_plot(total_train_loss, '--', 'train_loss', 'blue')
    plt.ylabel('Loss', fontsize=20)
    plt.xlabel('Epoch', fontsize=20)
    plt.legend(loc='upper right', fontsize=20)
    # plt.show()
    plt.savefig('{}_cv_loss_curve.pdf'.format(prefix))


def save_cv_output(total_train_loss, total_val_loss, total_train_accuracy, total_val_accuracy,
                   total_predicted_unchanged_test, total_actual_unchange, class_list, out_path, prefix=""):
    total_train_loss = save_result(total_train_loss, os.path.join(out_path, "total_train_loss"))
    total_val_loss = save_result(total_val_loss, os.path.join(out_path, "total_val_loss"))
    total_train_accuracy = save_result(total_train_accuracy, os.path.join(out_path, "total_train_accuracy"))
    total_val_accuracy = save_result(total_val_accuracy, os.path.join(out_path, "total_val_accuracy"))
    total_predicted_unchanged_test = save_result(total_predicted_unchanged_test, os.path.join(out_path, "total_predicted_unchanged_test"))
    total_actual_unchange = save_result(total_actual_unchange, os.path.join(out_path, "total_actual_unchange"))
    cv_roc_plot(total_actual_unchange, total_predicted_unchanged_test, class_list,
                prefix=os.path.join(out_path, prefix))
    acc_list = calculate_accuracy(total_actual_unchange, total_predicted_unchanged_test, class_list)
    confusion_matrix_list = calculate_cm(total_actual_unchange, total_predicted_unchanged_test)
    plot_confusion_matrix_cv(confusion_matrix_list, class_list, prefix=os.path.join(out_path, prefix))
    learning_curve(total_train_accuracy, total_val_accuracy, acc_list, prefix=os.path.join(out_path, prefix))
    loss_curve(total_val_loss, total_train_loss, prefix=os.path.join(out_path, prefix))


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, df, cm_df, le, batch_size=batch_size, dim=(299, 299), n_channels=3,
                 cm_len=None, n_classes=n_classes, shuffle=True, is_train=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.df = df
        self.list_IDs = cm_df.index
        self.n_channels = n_channels
        self.cm_len = cm_len
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.cm_df = cm_df
        self.le = le
        self.is_train = is_train
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X_img, X_cm, y = self.__data_generation(list_IDs_temp)
        if self.is_train:
            return [X_img, X_cm], y
        else:
            return [X_img, X_cm]

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X_img = np.empty((self.batch_size, *self.dim, self.n_channels))
        X_cm = np.empty((self.batch_size, self.cm_len))
        y = np.empty((self.batch_size, self.n_classes), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store img
            X_img[i,] = self._load_img(ID)

            # Store cm
            X_cm[i,] = self._load_cm(ID)
            # Store class
            y[i,] = self._load_label(ID)

        return X_img, X_cm, y

    def _load_img(self, img_temp):
        img_path = self.df.loc[img_temp, 'img']
        X_img = image.load_img(img_path, target_size=self.dim)
        X_img = image.img_to_array(X_img)
        X_img = np.expand_dims(X_img, axis=0)
        n_rotate = np.random.randint(0, 4)
        X_img = np.rot90(X_img, k=n_rotate, axes=(1, 2))
        X_img = preprocess_resnet(X_img)
        return X_img

    def _load_cm(self, cm_temp):
        spot = self.df.loc[cm_temp, 'cm']
        X_cm = self.cm_df.ix[spot, :-ADDITIONAL_COLUMN].values
        return X_cm

    def _load_label(self, lable_temp):
        spot = self.df.loc[lable_temp, 'cm']
        y = self.cm_df.ix[spot, [-ADDITIONAL_COLUMN]].values
        y = self.le.transform(y)
        return to_categorical(y, num_classes=self.n_classes)

    def get_classes(self):
        if not self.is_train:
            y = self.cm_df.iloc[:, [-ADDITIONAL_COLUMN]].values
            y = self.le.transform(y)
        return y


class ImageGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, df, cm_df, le, batch_size=batch_size, dim=(299, 299), n_channels=3,
                 cm_len=12666, n_classes=n_classes, shuffle=True, is_train=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.df = df
        self.list_IDs = cm_df.index
        self.n_channels = n_channels
        self.cm_len = cm_len
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.cm_df = cm_df
        self.le = le
        self.is_train = is_train
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X_img, X_cm, y = self.__data_generation(list_IDs_temp)
        if self.is_train:
            return X_img, y
        else:
            return X_img

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X_img = np.empty((self.batch_size, *self.dim, self.n_channels))
        X_cm = np.empty((self.batch_size, self.cm_len))
        y = np.empty((self.batch_size, self.n_classes), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store img
            X_img[i,] = self._load_img(ID)

            # Store cm
            X_cm[i,] = self._load_cm(ID)
            # Store class
            y[i,] = self._load_label(ID)

        return X_img, X_cm, y

    def _load_img(self, img_temp):
        img_path = self.df.loc[img_temp, 'img']
        X_img = image.load_img(img_path, target_size=self.dim)
        X_img = image.img_to_array(X_img)
        X_img = np.expand_dims(X_img, axis=0)
        n_rotate = np.random.randint(0, 4)
        X_img = np.rot90(X_img, k=n_rotate, axes=(1, 2))
        X_img = preprocess_resnet(X_img)
        return X_img

    def _load_cm(self, cm_temp):
        spot = self.df.loc[cm_temp, 'cm']
        X_cm = self.cm_df.ix[spot, :-ADDITIONAL_COLUMN].values
        return X_cm

    def _load_label(self, lable_temp):
        spot = self.df.loc[lable_temp, 'cm']
        y = self.cm_df.ix[spot, [-ADDITIONAL_COLUMN]].values
        y = self.le.transform(y)
        return to_categorical(y, num_classes=self.n_classes)

    def get_classes(self):
        if not self.is_train:
            y = self.cm_df.iloc[:, [-ADDITIONAL_COLUMN]].values
            y = self.le.transform(y)
            return y
