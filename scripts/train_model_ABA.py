import sys
import os
import csv
import math

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import pickle
import csv
import numpy as np
import pandas as pd
import random
import tensorflow as tf
from evoaug_tf import evoaug, augment
from src.diff_expression_model import get_model, get_siamese_model, post_hoc_conjoining, get_auroc
from Bio import SeqIO
from tensorflow import keras
from src.diff_expression_model import compile_model
from src.prepare_dataset import one_hot_encode_series, reverse_complement_series, reverse_complement_sequence
from src.prepare_dataset import grouped_shuffle_split
from src.prepare_dataset import one_hot_encode_series, reverse_complement_series, reverse_complement_sequence


def get_input_and_labels(df):
	ohe_np = np.stack(df["One_hot_encoded"])
	rc_np = np.stack(df["RC_one_hot_encoded"])

	x = np.append(ohe_np, rc_np, axis=0)
	x = x.astype('float32')
	y = np.append(df["Label"], df["Label"])
	return x, y

def main():
	# Prepare the data
	ppath = "data/dataset_solid_chrome.pkl"
	with open(ppath, 'rb') as f:
		data = pickle.load(f)

	dataset = data.reset_index()
	dataset = dataset[["geneID", "Category", "GeneFamily", "seqs"]]
	dataset.rename(columns={"geneID":"GeneID", "Category":"Label", "seqs": "Sequence"}, inplace=True)

	dataset["One_hot_encoded"] = one_hot_encode_series(dataset["Sequence"])
	dataset["RC_one_hot_encoded"] = reverse_complement_series(dataset["One_hot_encoded"])

	# Split train_df into train and validation
	train_df, validation_test_df = grouped_shuffle_split(dataset, dataset["GeneFamily"], 0.2)
	validation_df, test_df  = grouped_shuffle_split(validation_test_df, validation_test_df["GeneFamily"], 0.5)

	print(f"Length of training set: {train_df.shape[0]}")
	print(f"Length of validation set: {validation_df.shape[0]}")
	print(f"Length of test set: {test_df.shape[0]}")

	x_train, y_train = get_input_and_labels(train_df)
	x_validation, y_validation = get_input_and_labels(validation_df)

	augment_list = [
		augment.RandomRC(rc_prob=0.5),
		augment.RandomInsertionBatch(insert_min=0, insert_max=20),
		augment.RandomDeletion(delete_min=0, delete_max=30),
		augment.RandomTranslocationBatch(shift_min=0, shift_max=20),
		augment.RandomMutation(mutate_frac=0.05),
		augment.RandomNoise()
	]

	input_shape = train_df["One_hot_encoded"].iloc[0].shape

	f = open("/home/ubuntu/DeepDifE/results/test_gpu.csv", "a")
	f.write("loss,accurcay,auroc,auprc,TP,test_auroc,train_auroc,finetune_loss,finetune_accurcay,finetune_auroc,finetune_auprc,finetune_TP,finetune_test_auroc,finetune_train_auroc,post_hoc_validation")
	# f.write("loss,accurcay,auroc,auprc,TP,test_auroc,train_auroc,post_hoc_validation")
	f.close()
	
	
	for i in range(0,5):
		model = get_model(input_shape=input_shape, perform_evoaug=True, augment_list=augment_list,learning_rate=0.001)

		x_full_train = np.append(x_train, x_validation, axis=0)
		y_full_train = np.append(y_train, y_validation)


		early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
													patience=20,
													verbose=1,
													mode='min',
													restore_best_weights=True)
		# reduce learning rate callback
		reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
														factor=0.1,
														patience=5,
														min_lr=1e-7,
														mode='min',
														verbose=1)
		callbacks = [early_stopping_callback, reduce_lr]

		history = model.fit(x_train,
						y_train,
						epochs=100,
						batch_size=100,
						validation_data=(x_validation, y_validation),
						callbacks=callbacks
						)
		
		auroc_values = history.history['val_loss']

		# Find the index of the epoch where AUROC is the maximum
		max_auroc_epoch = np.argmin(auroc_values)

		# Extract all the metrics for the epoch with the maximum AUROC
		metrics_at_max_auroc = {metric: values[max_auroc_epoch] for metric, values in history.history.items()}
		metrics_at_max_auroc

		# Find training score
		train_auroc = metrics_at_max_auroc["auROC"]

		val_scores = model.evaluate(x_validation, y_validation, verbose=0)
		
		# Test set
		siamese_model = get_siamese_model(model.model)

		x_test = np.stack(test_df["One_hot_encoded"])
		x_test_rc = np.stack(test_df["RC_one_hot_encoded"])

		y_test = test_df["Label"].to_numpy()

		x_test = model._pad_end(x_test)
		x_test_rc = model._pad_end(x_test_rc)

		predictions_categories, predictions = post_hoc_conjoining(siamese_model, x_test, x_test_rc)

		test_auroc = get_auroc(y_test, predictions)

		# Post hoc conjoining
		siamese_model = get_siamese_model(model.model)

		x_val = np.stack(validation_df["One_hot_encoded"])
		x_val_rc = np.stack(validation_df["RC_one_hot_encoded"])

		y_val = validation_df["Label"].to_numpy()

		x_val = model._pad_end(x_val)
		x_val_rc = model._pad_end(x_val_rc)

		predictions_categories, predictions = post_hoc_conjoining(siamese_model, x_val, x_val_rc)

		post_hoc_validation = get_auroc(y_val, predictions)

		# Finetune
		finetune_optimizer = keras.optimizers.Adam(learning_rate=0.0001)
		model.finetune_mode(optimizer=finetune_optimizer)


		# set up callbacks
		early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
													patience=10,
													verbose=1,
													mode='min',
													restore_best_weights=True)
		# train model
		finetune_history = model.fit(x_train, y_train,
						epochs=20,
						batch_size=100,
						shuffle=True,
						validation_data=(x_validation, y_validation),
						callbacks=[early_stopping_callback])
		
		siamese_model = get_siamese_model(model.model)

		finetune_auroc_values = finetune_history.history['val_loss']

		# Find the index of the epoch where AUROC is the maximum
		finetune_max_auroc_epoch = np.argmin(finetune_auroc_values)

		# Extract all the metrics for the epoch with the maximum AUROC
		finetune_metrics_at_max_auroc = {metric: values[finetune_max_auroc_epoch] for metric, values in finetune_history.history.items()}
		finetune_metrics_at_max_auroc

		# Find training score
		finetune_train_auroc = finetune_metrics_at_max_auroc["auROC"]

		finetune_val_scores = model.evaluate(x_validation, y_validation, verbose=0)


		predictions_categories, predictions = post_hoc_conjoining(siamese_model, x_test, x_test_rc)

		finetune_test_auroc = get_auroc(y_test, predictions)
		f = open("/home/ubuntu/DeepDifE/results/test_gpu.csv", "a")
		f.write(f'{val_scores[0]},{val_scores[1]},{val_scores[2]},{val_scores[3]},{val_scores[4]},{test_auroc},{train_auroc},{finetune_val_scores[0]},{finetune_val_scores[1]},{finetune_val_scores[2]},{finetune_val_scores[3]},{finetune_val_scores[4]},{finetune_test_auroc},{finetune_train_auroc},{post_hoc_validation}\n')
		f.close()

if __name__=="__main__":
	main()