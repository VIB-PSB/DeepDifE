

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

import random

def get_batch(data, phylo_aug_data, indices, perform_phylo_aug=True, phylo_aug_rate=1.0):
	"""
	Creates a batch of the input and one-hot encodes the sequences
	"""
	seqs = []
	labels = []
	gene_ids = []
	for ii in indices:
		gene_id = data.iloc[ii].GeneID
		sample = data.iloc[ii]
		if random.choices(population=[0, 1], weights=[1 - phylo_aug_rate, phylo_aug_rate])[0] == 1:
			orthologs = phylo_aug_data[phylo_aug_data.Ortholog == gene_id]
			if not orthologs.empty:
				sample = phylo_aug_data[phylo_aug_data.Ortholog == gene_id].sample().iloc[0]
			# Randomly pick the forward or reverse complement strand
		if random.randint(0, 1) == 0:
			gene_ids.append(sample.GeneID)
			seqs.append(sample.RC_one_hot_encoded)
		else:
			gene_ids.append(sample.GeneID)
			seqs.append(sample.One_hot_encoded)
			
		labels.append(sample.Label)

	X_batch = np.array(seqs)
	X_batch = X_batch.astype('float32')

	Y_batch = np.array(labels)
	Y_batch = Y_batch.astype('int64')

	return X_batch, Y_batch

def data_gen(data, phylo_aug_data, batch_size, perform_phylo_aug=True, phylo_aug_rate=1.0):
	"""
	Generator function for loading input data in batches
	"""
	num_samples = data.shape[0]
	# indices = list(range(num_samples))
	indices = np.random.choice(list(range(num_samples)), num_samples, replace=False)
	# indices = np.random.choice(list(range(num_samples)), num_samples, replace=False)

	ii = 0
	while True:
		yield get_batch(data, phylo_aug_data, indices[ii:ii + batch_size], perform_phylo_aug, phylo_aug_rate)
		ii += batch_size
		if ii >= num_samples:
			ii = 0
			indices = np.random.choice(list(range(num_samples)), num_samples, replace=False)
			# indices = list(range(num_samples))

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
	ppath = "data/dataset_solid_chrome.pkl"
	with open(ppath, 'rb') as f:
		data = pickle.load(f)

	dataset = data.reset_index()
	dataset = dataset[["geneID", "Category", "GeneFamily", "seqs"]]
	dataset.rename(columns={"geneID":"GeneID", "Category":"Label", "seqs": "Sequence"}, inplace=True)

	phylo_augmentation_base_path = "data/phylo_aug/fasta/"
	species_fasta_files = {
				"aly":"aly.tss.fasta",
				"chi":"chi.tss.fasta",
				"cru":"cru.tss.fasta",
				"aar":"aar.tss.fasta",
				"ath-an1":"ath-an1.tss.fasta",
				"ath-c24":"ath-c24.tss.fasta",
				"ath-cvi":"ath-cvi.tss.fasta",
				"ath-eri":"ath-eri.tss.fasta",
				"ath-kyo":"ath-kyo.tss.fasta",
				"ath-ler":"ath-ler.tss.fasta",
				"ath-sha":"ath-sha.tss.fasta",
				"ath":"ath.tss.fasta",
				"chi":"chi.tss.fasta",
				"cpa":"cpa.tss.fasta",
				"cru":"cru.tss.fasta",
				"esa":"esa.tss.fasta",
				"spa":"spa.tss.fasta",
				"tha":"tha.tss.fasta"
	}

	species_to_include = ["ath-an1","ath-c24","ath-cvi","ath-eri","ath-kyo","ath-ler","ath-sha","aly","chi","cru"]

	from Bio import SeqIO

	sequence_list = []
	gene_id_list = []
	label_list = []
	species_list = []
	loss = []
	for species in species_to_include:
		species_fasta = species_fasta_files[species]
		fasta_path = f"{phylo_augmentation_base_path}{species_fasta}"
		fasta_sequences = SeqIO.parse(open(fasta_path),'fasta')
		print(f"File: {fasta_path} done!")
		for fasta in fasta_sequences:
			name, sequence = fasta.id, str(fasta.seq)
			
			# The label is stored inside the id name (e.g. >ATERI-4G42060_1), split these
			gene_id = name.split("_")[0]
			label = name.split("_")[1]
			species = species_fasta.split(".")[0]

			gene_id_list.append(gene_id)
			sequence_list.append(sequence)
			label_list.append(label)
			species_list.append(species)
	
	phylo_aug_df = pd.DataFrame({"GeneID": gene_id_list, "Label": label_list, "Sequence": sequence_list, "Species": species_list})
	base_dir_ortholog = "data/phylo_aug/orthologs/"
	ortholog_files = {
				"ath-aly":"ath_aly.txt",
				"ath-chi":"ath_chi.txt",
				"ath-cru":"ath_cru.txt",
				"ath-aar":"ath_aar.txt",
				"ath-aly":"ath_aly.txt",
				"ath-ath-an1":"ath_ath-an1.txt",
				"ath-ath-c24":"ath_ath-c24.txt",
				"ath-ath-cvi":"ath_ath-cvi.txt",
				"ath-ath-eri":"ath_ath-eri.txt",
				"ath-ath-kyo":"ath_ath-kyo.txt",
				"ath-ath-ler":"ath_ath-ler.txt",
				"ath-ath-sha":"ath_ath-sha.txt",
				"ath-ath":"ath_ath.txt",
				"ath-chi":"ath_chi.txt",
				"ath-cpa":"ath_cpa.txt",
				"ath-cru":"ath_cru.txt",
				"ath-esa":"ath_esa.txt",
				"ath-spa":"ath_spa.txt",
				"ath-tha":"ath_tha.txt"
				}
	
	lookup_table = {}
	for orthologs in ortholog_files.values():
		with open(f"{base_dir_ortholog}{orthologs}", 'r') as file:
			for line in file:
				# Split each line by tab
				columns = line.strip().split('\t')
				if len(columns) >= 2:  # Ensure there are at least two columns
					key = columns[1]
					value = columns[0]
					lookup_table[key] = value
	
	phylo_aug_df["Ortholog"] = phylo_aug_df["GeneID"].map(lookup_table)
	phylo_aug_df = pd.merge(phylo_aug_df, dataset[["GeneID", "GeneFamily"]], left_on="Ortholog", right_on="GeneID", how="left")
	phylo_aug_df.drop("GeneID_y", axis=1, inplace=True)
	phylo_aug_df.rename(columns={"GeneID_x": "GeneID"}, inplace=True)

	from src.prepare_dataset import one_hot_encode_series, reverse_complement_series, reverse_complement_sequence
	dataset["One_hot_encoded"] = one_hot_encode_series(dataset["Sequence"])
	phylo_aug_df["One_hot_encoded"] = one_hot_encode_series(phylo_aug_df["Sequence"])

	dataset["RC_one_hot_encoded"] = reverse_complement_series(dataset["One_hot_encoded"])
	phylo_aug_df["RC_one_hot_encoded"] = reverse_complement_series(phylo_aug_df["One_hot_encoded"])
	phylo_aug_df = phylo_aug_df[phylo_aug_df["GeneFamily"].notna()]

	from src.prepare_dataset import grouped_shuffle_split
	train_df, validation_test_df = grouped_shuffle_split(dataset, dataset["GeneFamily"], 0.2)
	validation_df, test_df  = grouped_shuffle_split(validation_test_df, validation_test_df["GeneFamily"], 0.5)
	train_df = train_df.append(validation_df)
	X, Y = get_input_and_labels(train_df, reverse_complement=False)
	groups = train_df["GeneFamily"]
	
	# early stopping callback
	import tensorflow as tf

	early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
												patience=40,
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
	
	
	from sklearn.model_selection import GroupKFold
	from src.diff_expression_model import compile_model

	BATCH_SIZE = 100
	group_kfold = GroupKFold(n_splits=5)


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

			x_validation, y_validation = get_input_and_labels(validation_df_fold, reverse_complement=True)
			train_data_gen = data_gen(train_df_fold,phylo_aug_df, BATCH_SIZE, True, 0.5)
			input_shape = train_df_fold["One_hot_encoded"].iloc[0].shape
	
			model = get_model(input_shape=input_shape, perform_evoaug=False, augment_list=[], learning_rate=0.001)

			# We add validation here, because one of the callbacks relies on val_loss metric
			import math
			history = model.fit(train_data_gen,
								steps_per_epoch=math.ceil(
									train_df_fold.shape[0] / BATCH_SIZE),
								epochs=100,
								validation_data=(x_validation, y_validation),
								callbacks=callbacks
								)
			
			# Finetuning the model
			x_train_pure, y_train_pure = get_input_and_labels(train_df, reverse_complement=True)
			model = compile_model(model, learning_rate=0.0001)

			# train model
			finetune_history = model.fit(x_train_pure, y_train_pure,
									epochs=10,
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
		f = open("/home/ubuntu/DeepDifE/results/result_phylo_cross_validation_all.csv", "a")
		f.write(f'{np.mean(auroc_list)}\n')
		f.close()

		print('------------------------------------------------------------------------')


if __name__=="__main__":
	main()