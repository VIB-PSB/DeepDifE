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
import optuna
import tensorflow as tf
from evoaug_tf import evoaug, augment
from src.diff_expression_model import get_model, get_siamese_model, post_hoc_conjoining, get_auroc
from Bio import SeqIO
from tensorflow import keras
from src.diff_expression_model import compile_model
from src.prepare_dataset import one_hot_encode_series, reverse_complement_series, reverse_complement_sequence
from src.prepare_dataset import grouped_shuffle_split

def get_input_and_labels(df):
	ohe_np = np.stack(df["One_hot_encoded"])
	rc_np = np.stack(df["RC_one_hot_encoded"])

	x = np.append(ohe_np, rc_np, axis=0)
	x = x.astype('float32')
	y = np.append(df["Label"], df["Label"])
	return x, y

def main():
	ppath = "data/dataset_solid_chrome.pkl"
	with open(ppath, 'rb') as f:
		data = pickle.load(f)
	dataset = data.reset_index()
	dataset = dataset[["geneID", "Category", "GeneFamily", "seqs"]]
	dataset.rename(columns={"geneID":"GeneID", "Category":"Label", "seqs": "Sequence"}, inplace=True)

	from src.prepare_dataset import one_hot_encode_series, reverse_complement_series, reverse_complement_sequence
	dataset["One_hot_encoded"] = one_hot_encode_series(dataset["Sequence"])
	dataset["RC_one_hot_encoded"] = reverse_complement_series(dataset["One_hot_encoded"])

	from src.prepare_dataset import grouped_shuffle_split
	train_df, test_df = grouped_shuffle_split(dataset, dataset["GeneFamily"], 0.2)

	print(f"Length of training set: {train_df.shape[0]}")
	print(f"Length of test set: {test_df.shape[0]}")


	import tensorflow as tf

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

	input_shape = train_df["One_hot_encoded"].iloc[0].shape

	X, Y = get_input_and_labels(train_df)
	groups = pd.concat([train_df["GeneFamily"], train_df["GeneFamily"]], axis = 0) 

	def objective(trial, x_train, y_train, x_val, y_val, i):
		insert_max = trial.suggest_int("insert_max", 10, 200, 20)
		delete_max = trial.suggest_int("delete_max", 10, 200, 20)
		max_translocation = trial.suggest_int("max_translocation", 10, 300, 10)
		mutation_frac = trial.suggest_float("mutate_frac", 0.05, 0.3, step=0.05)


		augment_list = [
			augment.RandomInsertionBatch(insert_min=0, insert_max=insert_max),
			augment.RandomDeletion(delete_min=0, delete_max=delete_max),
			augment.RandomTranslocationBatch(shift_min=0, shift_max=max_translocation),
			augment.RandomMutation(mutate_frac=mutation_frac),
			augment.RandomNoise()
		]

		model = get_model(input_shape=input_shape, 
					perform_evoaug=True, 
					augment_list=augment_list, 
					learning_rate=0.001)

		# We add validation here, because one of the callbacks relies on val_loss metric
		model.fit(x_train,
				y_train,
				epochs=100,	
				batch_size=100,
				validation_data=(x_val, y_val),
				callbacks=callbacks
				)
		score = model.evaluate(x_val, y_val, verbose=0)
		return score[0] 
	
	from sklearn.model_selection import GroupKFold

	def objective_cv(trial):

		# Get the MNIST dataset.
		group_kfold = GroupKFold(n_splits=5)
		
		scores = []
		for i, (train_index, validation_index) in enumerate(group_kfold.split(X, Y, groups)):
			x_train = X[train_index]
			y_train = Y[train_index]

			x_val = X[validation_index]
			y_val = Y[validation_index]

			loss = objective(trial, x_train, y_train, x_val, y_val, i)
			scores.append(loss)
		return np.mean(scores)

	study = optuna.create_study(direction='minimize')
	study.optimize(objective_cv, n_trials=150)
	best_params = study.best_params
	print(best_params)



if __name__=="__main__":
	main()