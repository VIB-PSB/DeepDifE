
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


def get_input_and_labels(df):
	ohe_np = np.stack(df["One_hot_encoded"])
	rc_np = np.stack(df["RC_one_hot_encoded"])

	x = np.append(ohe_np, rc_np, axis=0)
	x = x.astype('float32')
	y = np.append(df["Label"], df["Label"])
	y = y.astype('int64')
	return x, y


def create_ortholog_lookup_table():
	# To include the gene family to perform family-wise train-test split, we use the ortholog data
	base_dir_ortholog = "/home/ubuntu/DeepDifE/data/phylo_aug/orthologs/"
	ortholog_files = {
					"ath-aly":"ath_aly.txt",
					"ath-chi":"ath_chi.txt",
					"ath-cru":"ath_cru.txt"
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
	return lookup_table


def get_phylo_aug_data(dataset, max_distance=1000, include_accessions=True):
	phylo_augmentation_base_path = "/home/ubuntu/DeepDifE/data/phylo_aug/fasta/"

	accessions_fasta_files = {
				"ath-an1":"ath-an1.tss.fasta",
				"ath-c24":"ath-c24.tss.fasta",
				"ath-cvi":"ath-cvi.tss.fasta",
				"ath-eri":"ath-eri.tss.fasta",
				"ath-kyo":"ath-kyo.tss.fasta",
				"ath-ler":"ath-ler.tss.fasta",
				"ath-sha":"ath-sha.tss.fasta",
	}

	species_fasta_files = {
				"aly":"aly.tss.fasta",
				"chi":"chi.tss.fasta",
				"cru":"cru.tss.fasta"
	}

	if(include_accessions):
		species_fasta_files.update(accessions_fasta_files)
	
	# Check which species to include based on the max distance
	phylo_distance_path = "/home/ubuntu/DeepDifE/data/phylo_aug/phylo_distances.csv"

	with open(phylo_distance_path, mode='r') as phylo_distances:
		reader = csv.reader(phylo_distances)
		phylo_distance_dict = {rows[1]:int(rows[2]) for rows in reader}
	
	species_to_include = []
	for species in phylo_distance_dict:
		phylo_distance = phylo_distance_dict[species]
		if phylo_distance <= max_distance:
			species_to_include.append(species)
	
	# Based in this species selection parse the fasta files
	sequence_list = []
	gene_id_list = []
	label_list = []
	species_list = []
	for species in species_to_include:
		if not species in species_fasta_files:
			continue
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
	lookup_table = create_ortholog_lookup_table()
	phylo_aug_df["Ortholog"] = phylo_aug_df["GeneID"].map(lookup_table)
	phylo_aug_df = pd.merge(phylo_aug_df, dataset[["GeneID", "GeneFamily"]], left_on="Ortholog", right_on="GeneID", how="left")
	phylo_aug_df.drop("GeneID_y", axis=1, inplace=True)
	phylo_aug_df.rename(columns={"GeneID_x": "GeneID"}, inplace=True)
	return phylo_aug_df


def main():
	print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

	ppath = "/home/ubuntu/DeepDifE/data/dataset_solid_chrome.pkl"
	with open(ppath, 'rb') as f:
		data = pickle.load(f)

	dataset = data.reset_index()
	dataset = dataset[["geneID", "Category", "GeneFamily", "seqs"]]
	dataset.rename(columns={"geneID":"GeneID", "Category":"Label", "seqs": "Sequence"}, inplace=True)
	import time
	start_time = time.time()
	for max_phylo_distance in range(5, 6):
			for rate in range(0, 3):
				augmentation_rate = rate / 10
				BATCH_SIZE = 100

				phylo_aug_df = get_phylo_aug_data(dataset, max_phylo_distance, include_accessions=False)

				dataset["One_hot_encoded"] = one_hot_encode_series(dataset["Sequence"])
				phylo_aug_df["One_hot_encoded"] = one_hot_encode_series(phylo_aug_df["Sequence"])

				dataset["RC_one_hot_encoded"] = reverse_complement_series(dataset["One_hot_encoded"])
				phylo_aug_df["RC_one_hot_encoded"] = reverse_complement_series(phylo_aug_df["One_hot_encoded"])

				phylo_aug_df = phylo_aug_df[phylo_aug_df["GeneFamily"].notna()]

				train_df, validation_test_df = grouped_shuffle_split(dataset, dataset["GeneFamily"], 0.2)
				validation_df, test_df  = grouped_shuffle_split(validation_test_df, validation_test_df["GeneFamily"], 0.5)

				print(f"Length of training set: {train_df.shape[0]}")
				print(f"Length of validation set: {validation_df.shape[0]}")
				print(f"Length of test set: {test_df.shape[0]}")

				filtered_phylo_aug_df = phylo_aug_df[~phylo_aug_df["Ortholog"].isin(validation_test_df["GeneID"])]

				x_validation, y_validation = get_input_and_labels(validation_df)

				train_data_gen = data_gen(train_df,filtered_phylo_aug_df, BATCH_SIZE, True, augmentation_rate)

				input_shape = train_df["One_hot_encoded"].iloc[0].shape
				model = get_model(input_shape=input_shape, perform_evoaug=False,learning_rate=0.001)

				# early stopping callback

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

				model.fit(train_data_gen,
							steps_per_epoch=math.ceil(
								train_df.shape[0] / BATCH_SIZE),
							epochs=100,
							validation_data=(x_validation, y_validation),
							callbacks=callbacks
							)
				scores = model.evaluate(x_validation, y_validation, verbose=0)

				x_train_pure, y_train_pure = get_input_and_labels(train_df)

				# set up callbacks
				model = compile_model(model, learning_rate=0.0001)

				# train model
				model.fit(x_train_pure, y_train_pure,
								epochs=10,
								batch_size=100,
								shuffle=True,
								validation_data=(x_validation, y_validation),
								callbacks=[early_stopping_callback])

				scores = model.evaluate(x_validation, y_validation, verbose=0)
				f = open("/home/ubuntu/DeepDifE/data/phylo_aug/results/result_distribution_rate_distance.csv", "a")
				f.write(f'{max_phylo_distance},{augmentation_rate},{scores[0]},{scores[1]},{scores[2]},{scores[3]},{scores[4]}\n')
				f.close()
	print("--- %s seconds ---" % (time.time() - start_time))
if __name__=="__main__":
	main()