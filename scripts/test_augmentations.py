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

def get_input_and_labels(df):
	ohe_np = np.stack(df["One_hot_encoded"])
	rc_np = np.stack(df["RC_one_hot_encoded"])

	x = np.append(ohe_np, rc_np, axis=0)
	x = x.astype('float32')
	y = np.append(df["Label"], df["Label"])
	return x, y


augment_list = [
	augment.RandomRC(rc_prob=0.5),
	augment.RandomInsertionBatch(insert_min=0, insert_max=20),
	augment.RandomDeletion(delete_min=0, delete_max=30),
	augment.RandomTranslocationBatch(shift_min=0, shift_max=20),
	augment.RandomMutation(mutate_frac=0.05),
	augment.RandomNoise()
]


def main():
	dataset = pd.read_csv("/home/ubuntu/DeepDifE/diatoms_data/high_light_30min_genes_lfc_cutoff_2_500up.csv")
	dataset[~dataset.Sequence.str.contains("N")]

	dataset["One_hot_encoded"] = one_hot_encode_series(dataset["Sequence"])
	dataset["RC_one_hot_encoded"] = reverse_complement_series(dataset["One_hot_encoded"])
	dataset = dataset.fillna("no_gf")
	train_df, validation_test_df = grouped_shuffle_split(dataset, dataset["GeneFamily"], 0.4)
	validation_df, test_df  = grouped_shuffle_split(validation_test_df, validation_test_df["GeneFamily"], 0.5)
	x_train, y_train = get_input_and_labels(train_df)
	x_validation, y_validation = get_input_and_labels(validation_df)
	input_shape = train_df["One_hot_encoded"].iloc[0].shape
	model = get_model(input_shape=input_shape, perform_evoaug=True, augment_list=augment_list,learning_rate=0.001)

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

	siamese_model = get_siamese_model(model.model)


	x_test = np.stack(test_df["One_hot_encoded"])
	x_test_rc = np.stack(test_df["RC_one_hot_encoded"])

	x_test = model._pad_end(x_test)
	x_test_rc = model._pad_end(x_test_rc)

	y_test = test_df["Label"].to_numpy()


	predictions_categories, predictions = post_hoc_conjoining(siamese_model, x_test, x_test_rc)

	get_auroc(y_test, predictions)

	

if __name__=="__main__":
	main()