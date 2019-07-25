from keras.layers import Dense, GlobalAveragePooling2D, Input, concatenate, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.applications import xception
from keras.utils import multi_gpu_model
from keras.applications.xception import Xception
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils import class_weight
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def st_comb_nn(tile_shape, cm_shape, output_shape):
    #### xception base for tile
    tile_input = Input(shape=tile_shape, name = "tile_input")
    xception_base = Xception(input_tensor=tile_input, weights='imagenet', include_top=False)
    x_tile = xception_base.output
    x_tile = GlobalAveragePooling2D()(x_tile)
    x_tile = Dense(512, activation='relu', name="tile_fc")(x_tile)
    #### NN for count matrix
    cm_input = Input(shape=cm_shape, name="count_matrix_input")
    x_cm = Dense(512, activation='relu', name="cm_fc")(cm_input)
    #### merge
    merge = concatenate([x_tile, x_cm], name="merge_tile_cm")
    merge = Dense(512, activation='relu', name="merge_fc_1")(merge)
    merge = Dense(128, activation='relu', name="merge_fc_2")(merge)
    preds = Dense(output_shape, activation='softmax')(merge)
    ##### compile model
    model = Model(inputs=[tile_input, cm_input], outputs=preds)
    try:
        parallel_model = multi_gpu_model(model, gpus=4, cpu_merge=False)
    except:
        parallel_model = model
    parallel_model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return parallel_model, model


def model_eval(y_pred, y_true, class_list):
    y_true_onehot = np.zeros((len(y_true), len(class_list)))
    y_true_onehot[np.arange(len(y_true)), y_true] = 1
    y_pred_int = np.argmax(y_pred, axis=1)
    confusion_matrix_age = confusion_matrix(y_true, y_pred_int)
    plt.figure()
    color = ['blue', 'green', 'red', 'cyan']
    for i in range(len(class_list)):
        fpr, tpr, thresholds = roc_curve(y_true_onehot[:,i], y_pred[:,i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color[i], lw=2, label='ROC %s curve (area = %0.2f)' % (class_list[i], roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('roc_auc')
    plt.legend(loc="lower right")
    plt.savefig('./age_roc_combine.pdf')
    cm_plot = plot_confusion_matrix(confusion_matrix_age, classes = class_list)


def plot_confusion_matrix(cm, classes=None):
    #Normalise Confusion Matrix by dividing each value by the sum of that row
    cm = cm.astype('float')/cm.sum(axis = 1)[:, np.newaxis]
    print(cm)
    #Make DataFrame from Confusion Matrix and classes
    cm_df = pd.DataFrame(cm, index = classes, columns = classes)
    #Display Confusion Matrix 
    plt.figure(figsize = (4,4), dpi = 300)
    cm_plot = sns.heatmap(cm_df, vmin = 0, vmax = 1, annot = True, fmt = '.2f', cmap = 'Blues', square = True)
    plt.title('Confusion Matrix', fontsize = 12)
    #Display axes labels
    plt.ylabel('True label', fontsize = 12)
    plt.xlabel('Predicted label', fontsize = 12)
    plt.savefig('./age_confusion_matrix_combine.pdf')
    plt.tight_layout()
    return cm_plot