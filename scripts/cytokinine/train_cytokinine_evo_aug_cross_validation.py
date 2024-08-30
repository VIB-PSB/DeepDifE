import sys
import os
import csv
import math

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname((os.path.dirname(__file__))))))

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
from sklearn.model_selection import GroupKFold
from src.diff_expression_model import compile_model
from src.diff_expression_model import compile_model
from src.prepare_dataset import one_hot_encode_series, reverse_complement_series, reverse_complement_sequence
from src.prepare_dataset import grouped_shuffle_split
from src.prepare_dataset import one_hot_encode_series, reverse_complement_series, reverse_complement_sequence


def get_input_and_labels(df, reverse_complement=True):
	ohe_np = np.stack(df["One_hot_encoded"])
	rc_np = np.stack(df["RC_one_hot_encoded"])

	if reverse_complement:
		x = np.append(ohe_np, rc_np, axis=0)
		y = np.append(df["Label"], df["Label"])
	else:
		x = ohe_np
		y = df["Label"]
	
	x = x.astype('float32')
	y = y.astype('int64')
	return x, y

def main():
	# Prepare the data
	train_df = pd.read_csv("cytokinine_data/train.csv")
	test_df = pd.read_csv("cytokinine_data/test.csv")

	train_df.rename(columns={"gene_id": "GeneID", "class": "Label", "gene_family":"GeneFamily", "sequence":"Sequence"}, inplace=True)
	test_df.rename(columns={"gene_id": "GeneID", "class": "Label", "gene_family":"GeneFamily", "sequence":"Sequence"}, inplace=True)

	# One hot encoding
	train_df["One_hot_encoded"] = one_hot_encode_series(train_df["Sequence"])
	test_df["One_hot_encoded"] = one_hot_encode_series(test_df["Sequence"])

	train_df["RC_one_hot_encoded"] = reverse_complement_series(train_df["One_hot_encoded"])
	test_df["RC_one_hot_encoded"] = reverse_complement_series(test_df["One_hot_encoded"])

	X, Y = get_input_and_labels(train_df, reverse_complement=False)
	groups = train_df["GeneFamily"]

	input_shape = train_df["One_hot_encoded"].iloc[0].shape
	
	group_kfold = GroupKFold(n_splits=5)

	# OPTIMIZED AUGMENTATION SETTINGS
	# augment_list = [
	# 	augment.RandomInsertionBatch(insert_min=0, insert_max=30),
	# 	augment.RandomDeletion(delete_min=0, delete_max=10),
	# 	augment.RandomMutation(mutate_frac=0.05),
	# 	augment.RandomTranslocationBatch(shift_min=0, shift_max=240),
	# 	augment.RandomNoise()
	# 	]
	
	augment_list = [
		augment.RandomInsertionBatch(insert_min=0, insert_max=20),
		augment.RandomDeletion(delete_min=0, delete_max=30),
		augment.RandomTranslocationBatch(shift_min=0, shift_max=20),
		augment.RandomMutation(mutate_frac=0.05),
		augment.RandomNoise()
	]
	
	for i in range(0,10):

		# Define metric containers
		loss_list = []
		accuracy_list = []
		auroc_list = []
		auprc_list = []
		true_positive_list = []
		
		for i, (train_index, validation_index) in enumerate(group_kfold.split(X, Y, groups)):
			train_df_fold = train_df.iloc[list(train_index)]
			validation_df_fold = train_df.iloc[validation_index]

			x_train, y_train = get_input_and_labels(train_df_fold, reverse_complement=True)
			x_validation, y_validation = get_input_and_labels(validation_df_fold, reverse_complement=True)
		
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
		

			scores = model.evaluate(x_validation, y_validation, verbose=0)
			print(f'Score for fold {i}: Loss of {scores[0]}; accuracy of {scores[1]}; auroc of {scores[2]}; auprc of {scores[3]}; TP of {scores[4]}')
			
			loss_list.append(scores[0])
			accuracy_list.append(scores[1])
			auroc_list.append(scores[2])
			auprc_list.append(scores[3])
			true_positive_list.append(scores[4])

		print('------------------------------------------------------------------------')
		print('Score per fold')
		for i in range(0, len(loss_list)):
			print('------------------------------------------------------------------------')
			print(f'Score for fold {i}: Loss of {loss_list[i]}; accuracy of {accuracy_list[i]}; AUROC of {auroc_list[i]}; AUPRC of {auprc_list[i]}; TP of {true_positive_list[i]}')
			print('------------------------------------------------------------------------')
		print('Average scores for all folds:')
		print(f'> Loss: {np.mean(loss_list)} (+- {np.std(loss_list)})')
		print(f'> Auroc: {np.mean(auroc_list)}')
		f = open("/home/ubuntu/DeepDifE/cytokinine_data/results/result_evoaug_cv.csv", "a")
		f.write(f'{np.mean(auroc_list)}\n')
		f.close()

		print('------------------------------------------------------------------------')



if __name__=="__main__":
	main()