from config import *

from model import st_comb_nn, model_eval, plot_confusion_matrix, st_nn, st_cnn
from tensorflow.keras.utils import to_categorical
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from utils import find_sample_name, save_result, mkdirs, save_cv_output, ImageGenerator, DataGenerator




def run_gene_model():
	train_cm_X = train_cm.iloc[:, 0:-ADDITIONAL_COLUMN].values
	train_cm_Y = train_cm.loc[:, LABEL_COLUMN].values
	train_cm_Y_one_hot = to_categorical(label_encoder.transform(train_cm_Y), num_classes=len(class_list))

	test_cm_X = test_cm.iloc[:, 0:-ADDITIONAL_COLUMN].values
	test_cm_Y = test_cm.loc[:, LABEL_COLUMN].values
	test_cm_Y_one_hot = to_categorical(label_encoder.transform(test_cm_Y), num_classes=len(class_list))

	parallel_model_gene = st_nn(train_cm_X.shape[1], train_cm_Y_one_hot.shape[1])
	parallel_model_gene.fit(train_cm_X, train_cm_Y_one_hot,
											batch_size=batch_size,
											epochs=epochs)
	y_pred = parallel_model_gene.predict(test_cm_X, verbose=1)
	# get true label (categorical)
	y_true = test_cm_Y
	# convert categorical string label to numerical label
	y_true = label_encoder.transform(y_true)
	# plot ROC and confusion matrix
	out_path = os.path.join(DATASET_PATH, "results_gene_model")
	mkdirs(out_path)
	model_eval(y_pred, y_true, class_list=class_list, prefix=os.path.join(out_path, "single_model_gene_count"))


def run_tile_model():
	# create input batch for training model. eg. (image, label) * batch_size
	train_gen = ImageGenerator(df=df, cm_df=train_cm, le=label_encoder, cm_len=cm_shape, batch_size=batch_size)
	# create test data generator. eg. image * 1
	test_gen = ImageGenerator(df=df, cm_df=test_cm, le=label_encoder, shuffle=False, is_train=False, batch_size=1,
							 cm_len=cm_shape)
	# compile model
	parallel_model_combine = st_cnn((SIZE[0], SIZE[1], N_CHANNEL), len(class_list))
	# training model
	parallel_model_combine.fit_generator(generator=train_gen, steps_per_epoch=len(train_gen), epochs=epochs)
	# predict on test dataset to get predict values with shape (n_classes,)
	y_pred = parallel_model_combine.predict_generator(generator=test_gen, verbose=1)
	# get true label (categorical)
	y_true = test_cm.loc[:, LABEL_COLUMN].values
	# convert categorical string label to numerical label
	y_true = label_encoder.transform(y_true)
	# plot ROC and confusion matrix
	out_path = os.path.join(DATASET_PATH, "results_tile_model")
	mkdirs(out_path)
	model_eval(y_pred, y_true, class_list=class_list, prefix=os.path.join(out_path, "single_model_tile"))


def run_combine_model():
	# create input batch for training model. eg. ([image, cm], label) * batch_size
	train_gen = DataGenerator(df=df, cm_df=train_cm, le=label_encoder, cm_len=cm_shape, batch_size=batch_size)
	# create test data generator. eg. [image, cm] * 1
	test_gen = DataGenerator(df=df, cm_df=test_cm, le=label_encoder, shuffle=False, is_train=False, batch_size=1,
							 cm_len=cm_shape)
	# compile model
	parallel_model_combine, model_ = st_comb_nn((SIZE[0], SIZE[1], N_CHANNEL), (cm_shape,), len(class_list))
	# training model
	parallel_model_combine.fit_generator(generator=train_gen, steps_per_epoch=len(train_gen), epochs=epochs)
	# predict on test dataset to get predict values with shape (n_classes,)
	y_pred = parallel_model_combine.predict_generator(generator=test_gen, verbose=1)
	# get true label (categorical)
	y_true = test_cm.loc[:, LABEL_COLUMN].values
	# convert categorical string label to numerical label
	y_true = label_encoder.transform(y_true)
	# plot ROC and confusion matrix
	out_path = os.path.join(DATASET_PATH, "results_combine_model")
	mkdirs(out_path)
	model_eval(y_pred, y_true, class_list=class_list, prefix=os.path.join(out_path, "combine_model"))


def run_gene_model_cross_validation():
	total_val_loss = []
	total_train_accuracy = []
	total_train_loss = []
	total_val_accuracy = []
	total_predicted_unchanged_test = []
	total_actual_unchange = []
	for i, (train_cv_index, test_cv_index) in enumerate(
			skf.split(df_sample_label.loc[train_index, ], df_sample_label.loc[train_index, ]["label"])):

		train_cv_cm = sorted_cm[df["sample"].isin(df_sample_label.index[train_cv_index])]
		test_cv_cm = sorted_cm[df["sample"].isin(df_sample_label.index[test_cv_index])]
		train_cm_X = train_cv_cm.iloc[:, 0:-ADDITIONAL_COLUMN].values
		train_cm_Y = train_cv_cm.loc[:, LABEL_COLUMN].values
		train_cm_Y_one_hot = to_categorical(label_encoder.transform(train_cm_Y), num_classes=len(class_list))

		test_cm_X = test_cv_cm.iloc[:, 0:-ADDITIONAL_COLUMN].values
		test_cm_Y = test_cv_cm.loc[:, LABEL_COLUMN].values
		test_cm_Y_one_hot = to_categorical(label_encoder.transform(test_cm_Y), num_classes=len(class_list))

		parallel_model_gene = st_nn(train_cm_X.shape[1], train_cm_Y_one_hot.shape[1])
		train_history = parallel_model_gene.fit(train_cm_X, train_cm_Y_one_hot,
												batch_size=batch_size,
												epochs=epochs,
												validation_data=(test_cm_X, test_cm_Y_one_hot))
		loss = train_history.history['loss']
		total_train_loss.append(loss)
		val_loss = train_history.history['val_loss']
		total_val_loss.append(val_loss)
		acc = train_history.history['acc']
		total_train_accuracy.append(acc)
		val_acc = train_history.history['val_acc']
		total_val_accuracy.append(val_acc)
		y_pred_unchange = parallel_model_gene.predict(test_cm.iloc[:, :-ADDITIONAL_COLUMN].values, verbose=1)
		total_predicted_unchanged_test.append(y_pred_unchange)
		y_true_unchange = test_cm.loc[:, LABEL_COLUMN].values
		y_true_unchange = label_encoder.transform(y_true_unchange)
		total_actual_unchange.append(y_true_unchange)

	out_path = os.path.join(DATASET_PATH, "{}_fold_results_gene_model".format(k_fold))
	mkdirs(out_path)
	save_cv_output(total_train_loss, total_val_loss, total_train_accuracy, total_val_accuracy,
				   total_predicted_unchanged_test, total_actual_unchange, class_list, out_path, prefix="single_model_gene_count")


def run_tile_model_cross_validation():
	total_val_loss = []
	total_train_accuracy = []
	total_train_loss = []
	total_val_accuracy = []
	total_predicted_unchanged_test = []
	total_actual_unchange = []
	for i, (train_cv_index, test_cv_index) in enumerate(
			skf.split(df_sample_label.loc[train_index,], df_sample_label.loc[train_index,]["label"])):
		train_cv_cm = sorted_cm[df["sample"].isin(df_sample_label.index[train_cv_index])]
		test_cv_cm = sorted_cm[df["sample"].isin(df_sample_label.index[test_cv_index])]
		cm_shape = train_cm.shape[1] - 2
		train_gen = ImageGenerator(df=df,
								  cm_df=train_cv_cm,
								  le=label_encoder,
								  cm_len=cm_shape)
		valid_gen = ImageGenerator(df=df,
								  cm_df=test_cv_cm,
								  le=label_encoder,
								  cm_len=cm_shape)
		test_unchange_gen = ImageGenerator(df=df,
										  cm_df=test_cm,
										  le=label_encoder,
										  shuffle=False,
										  is_train=False,
										  batch_size=1,
										  cm_len=cm_shape)

		parallel_model_tile = st_cnn((SIZE[0], SIZE[1], N_CHANNEL), len(class_list))
		train_history = parallel_model_tile.fit_generator(generator=train_gen,
														  steps_per_epoch=len(train_gen),
														  epochs=epochs,
														  validation_data=valid_gen,
														  validation_steps=len(valid_gen))
		loss = train_history.history['loss']
		total_train_loss.append(loss)
		val_loss = train_history.history['val_loss']
		total_val_loss.append(val_loss)
		acc = train_history.history['acc']
		total_train_accuracy.append(acc)
		val_acc = train_history.history['val_acc']
		total_val_accuracy.append(val_acc)
		y_pred_unchange = parallel_model_tile.predict_generator(generator=test_unchange_gen, verbose=1)
		total_predicted_unchanged_test.append(y_pred_unchange)
		y_true_unchange = test_cm.loc[:, LABEL_COLUMN].values
		y_true_unchange = label_encoder.transform(y_true_unchange)
		total_actual_unchange.append(y_true_unchange)

	out_path = os.path.join(DATASET_PATH, "{}_fold_result_tile_model".format(k_fold))
	mkdirs(out_path)
	save_cv_output(total_train_loss, total_val_loss, total_train_accuracy, total_val_accuracy,
				   total_predicted_unchanged_test, total_actual_unchange, class_list, out_path, prefix="single_model_tile")


def run_combine_model_cross_validation():
	total_val_loss = []
	total_train_accuracy = []
	total_train_loss = []
	total_val_accuracy = []
	total_predicted_unchanged_test = []
	total_actual_unchange = []
	for i, (train_cv_index, test_cv_index) in enumerate(
			skf.split(df_sample_label.loc[train_index,], df_sample_label.loc[train_index,]["label"])):
		train_cv_cm = sorted_cm[df["sample"].isin(df_sample_label.index[train_cv_index])]
		test_cv_cm = sorted_cm[df["sample"].isin(df_sample_label.index[test_cv_index])]
		cm_shape = train_cm.shape[1] - 2
		train_gen = DataGenerator(df=df,
								  cm_df=train_cv_cm,
								  le=label_encoder,
								  cm_len=cm_shape)
		valid_gen = DataGenerator(df=df,
								  cm_df=test_cv_cm,
								  le=label_encoder,
								  cm_len=cm_shape)
		test_unchange_gen = DataGenerator(df=df,
										  cm_df=test_cm,
										  le=label_encoder,
										  shuffle=False,
										  is_train=False,
										  batch_size=1,
										  cm_len=cm_shape)

		parallel_model_combine, model_ = st_comb_nn((SIZE[0], SIZE[1], N_CHANNEL), (cm_shape,), len(class_list))
		train_history = parallel_model_combine.fit_generator(generator=train_gen,
														  steps_per_epoch=len(train_gen),
														  epochs=epochs,
														  validation_data=valid_gen,
														  validation_steps=len(valid_gen))
		loss = train_history.history['loss']
		total_train_loss.append(loss)
		val_loss = train_history.history['val_loss']
		total_val_loss.append(val_loss)
		acc = train_history.history['acc']
		total_train_accuracy.append(acc)
		val_acc = train_history.history['val_acc']
		total_val_accuracy.append(val_acc)
		y_pred_unchange = parallel_model_combine.predict_generator(generator=test_unchange_gen, verbose=1)
		total_predicted_unchanged_test.append(y_pred_unchange)
		y_true_unchange = test_cm.loc[:, LABEL_COLUMN].values
		y_true_unchange = label_encoder.transform(y_true_unchange)
		total_actual_unchange.append(y_true_unchange)

	out_path = os.path.join(DATASET_PATH, "{}_fold_result_combine_model".format(k_fold))
	mkdirs(out_path)
	save_cv_output(total_train_loss, total_val_loss, total_train_accuracy, total_val_accuracy,
				total_predicted_unchanged_test, total_actual_unchange, class_list, out_path, prefix="combine_model")



if __name__ == '__main__':
	# load dataset management file and gene count matrix
	df = pd.read_csv(os.path.join(DATASET_PATH, "dataset.tsv"), header=0, sep='\t', index_col=0)
	sorted_cm = pd.read_csv(os.path.join(DATASET_PATH, "cm_final.tsv"), header=0, sep='\t', index_col=0)
	# convert categorical string label to numerical label
	label_encoder = LabelEncoder()
	label_encoder.fit(list(set(sorted_cm.loc[:, LABEL_COLUMN])))
	class_list = label_encoder.classes_
	# split training data and test data by individual samples
	df["sample"] = df.index.map(find_sample_name)
	df_sample_label = df.groupby("sample").first()
	train_index, test_index, _, _ = train_test_split(df_sample_label.index, df_sample_label.label, train_size=train_ratio, stratify=df_sample_label.label)
	# subset training and test gene count matrix
	train_cm = sorted_cm[df["sample"].isin(train_index)]
	test_cm = sorted_cm[df["sample"].isin(test_index)]
	# calculate number of genes
	cm_shape = train_cm.shape[1]-ADDITIONAL_COLUMN
	if cross_validation:
		skf = StratifiedKFold(n_splits=k_fold, shuffle=True)
		if "combine" in model:
			run_combine_model_cross_validation()
		if "gene_only" in model:
			run_gene_model_cross_validation()
		if "tile_only" in model:
			run_tile_model_cross_validation()
	else:
		if "combine" in model:
			run_combine_model()
		if "gene_only" in model:
			run_gene_model()
		if "tile_only" in model:
			run_tile_model()

