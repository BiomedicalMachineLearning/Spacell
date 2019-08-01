import glob
import keras
import numpy as np
import pandas as pd
import os
import time 
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import xception
from keras.utils import to_categorical
from config import *


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, df, cm_df, le, batch_size=32, dim=(299,299), n_channels=3,
                 cm_len = None, n_classes=n_classes, shuffle=True, is_train=True):
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
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

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
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
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
        X_img = xception.preprocess_input(X_img)
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
    df.to_csv(os.path.join(DATASET_PATH, 'dataset_age.tsv'), sep='\t')
    cm.to_csv(os.path.join(DATASET_PATH, "cm_age.tsv"), sep='\t')
