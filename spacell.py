from config import *
from dataset_management import DataGenerator
from model import st_comb_nn, model_eval, plot_confusion_matrix
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split




if __name__ == '__main__':
	df = pd.read_csv(os.path.join(DATASET_PATH,"dataset_age.tsv"), header=0, sep='\t', index_col=0)
	sorted_cm_age_G63A = pd.read_csv(os.path.join(DATASET_PATH,"cm_age.tsv"), header=0, sep='\t', index_col=0)
	label_encoder = LabelEncoder()
	class_list = list(set(sorted_cm_age_G63A.loc[:,'age']))
	label_encoder.fit(class_list)
	train_index, test_index = train_test_split(sorted_cm_age_G63A.index, train_size = 0.7, shuffle=True)
	train_cm = sorted_cm_age_G63A.loc[train_index,:]
	test_cm = sorted_cm_age_G63A.loc[test_index,:]
	cm_shape = train_cm.shape[1]-2
	train_gen = DataGenerator(df=df, cm_df=train_cm, le=label_encoder, cm_len=cm_shape)
	test_gen = DataGenerator(df=df, cm_df=test_cm, le=label_encoder, shuffle=False, is_train=False, batch_size=1,
                        	cm_len=cm_shape)
	parallel_model_combine, model_ = st_comb_nn((299, 299, 3), (cm_shape,), n_classes)
	parallel_model_combine.fit_generator(generator=train_gen,
                                  		steps_per_epoch=len(train_gen),
                                  		epochs=1)
	y_pred = parallel_model_combine.predict_generator(generator=test_gen, verbose=1)
	y_true = sorted_cm_age_G63A.ix[test_index,[-2]].values
	y_true = label_encoder.transform(y_true)
	model_eval(y_pred, y_true, class_list=class_list)