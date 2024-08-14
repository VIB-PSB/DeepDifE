
import sys
import os
import csv

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import pickle
import importlib
import esparto
import optuna
from Bio import SeqIO
import tensorflow as tf
import numpy as np
import pandas as pd
from evoaug_tf import evoaug, augment
from src.diff_expression_model import get_model, get_siamese_model, post_hoc_conjoining, get_auroc
from src.prepare_dataset import one_hot_encode_series, reverse_complement_series, reverse_complement_sequence
from src.prepare_dataset import grouped_shuffle_split
from skopt.utils import use_named_args


def get_augmented_samples(base_path, fasta_files):
	# Process the phylo aug fasta files and get the labels
	sequence_list = []
	gene_id_list = []
	label_list = []
	species_list = []
	for species_fasta in fasta_files:
		fasta_path = f"{base_path}{species_fasta}"
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

	# Create phylo aug dataframe
	phylo_aug_df = pd.DataFrame({"GeneID": gene_id_list, "Label": label_list, "Sequence": sequence_list, "Species": species_list})
	return phylo_aug_df


def get_orthology_lookup_table(base_path, orthology_files):
	# Create ortholog lookup table
	lookup_table = {}
	for orthologs in orthology_files.values():
		with open(f"{base_path}{orthologs}", 'r') as file:
			for line in file:
				# Split each line by tab
				columns = line.strip().split('\t')
				if len(columns) >= 2:  # Ensure there are at least two columns
					key = columns[1]
					value = columns[0]
					lookup_table[key] = value
	return lookup_table


def main():
	# Load the dataset
	ppath = "/home/ubuntu/DeepDifE/data/dataset_solid_chrome.pkl"
	with open(ppath, 'rb') as f:
		data = pickle.load(f)


	dataset = data.reset_index()
	dataset = dataset[["geneID", "Category", "GeneFamily", "seqs"]]
	dataset.rename(columns={"geneID":"GeneID", "Category":"Label", "seqs": "Sequence"}, inplace=True)

	# Parse the phylo augmentation samples
	phylo_augmentation_base_path = "/home/ubuntu/DeepDifE/data/phylo_aug/fastas/"
	species_fasta_files = [
				"ath-an1.tss.fasta",
				"ath-c24.tss.fasta",
				"ath-cvi.tss.fasta",
				"ath-eri.tss.fasta",
				"ath-kyo.tss.fasta",
				"ath-ler.tss.fasta",
				"ath-sha.tss.fasta"
	]
	phylo_aug_df = get_augmented_samples(phylo_augmentation_base_path, species_fasta_files)

	# We need to know from which gene each ortholog originates to get the gene families and to avoid leakage
	base_dir_ortholog = "/home/ubuntu/DeepDifE/data/phylo_aug/orthologs/"
	ortholog_files = {
				"ath-aar":"ath_aar.txt",
				"ath-aly":"ath_aly.txt",
				"ath-an1":"ath_ath-an1.txt",
				"ath-c24":"ath_ath-c24.txt",
				"ath-cvi":"ath_ath-cvi.txt",
				"ath-eri":"ath_ath-eri.txt",
				"ath-kyo":"ath_ath-kyo.txt",
				"ath-ler":"ath_ath-ler.txt",
				"ath-sha":"ath_ath-sha.txt",
				"ath-ath":"ath_ath.txt",
				"ath-chi":"ath_chi.txt",
				"ath-cpa":"ath_cpa.txt",
				"ath-cru":"ath_cru.txt",
				"ath-esa":"ath_esa.txt",
				"ath-spa":"ath_spa.txt",
				"ath-tha":"ath_tha.txt"
				}

	lookup_table = get_orthology_lookup_table(base_dir_ortholog, ortholog_files)

	# Create ortholog column
	phylo_aug_df["Ortholog"] = phylo_aug_df["GeneID"].map(lookup_table)

	# Based on these ortholgs we can set the gene family
	phylo_aug_df = pd.merge(phylo_aug_df, dataset[["GeneID", "GeneFamily"]], left_on="Ortholog", right_on="GeneID", how="left")
	phylo_aug_df.drop("GeneID_y", axis=1, inplace=True)
	phylo_aug_df.rename(columns={"GeneID_x": "GeneID"}, inplace=True)

	# One hot encoding (and reverse complement) of both pure dataset and the phylo augmentation dataframe
	dataset["One_hot_encoded"] = one_hot_encode_series(dataset["Sequence"])
	phylo_aug_df["One_hot_encoded"] = one_hot_encode_series(phylo_aug_df["Sequence"])

	dataset["RC_one_hot_encoded"] = reverse_complement_series(dataset["One_hot_encoded"])
	phylo_aug_df["RC_one_hot_encoded"] = reverse_complement_series(phylo_aug_df["One_hot_encoded"])

	# Remove the rows where no gene family is known
	phylo_aug_df = phylo_aug_df[phylo_aug_df["GeneFamily"].notna()]

	# We split the pure dataset (test set will be the same as in previous experiment)
	train_df, validation_test_df = grouped_shuffle_split(dataset, dataset["GeneFamily"], 0.2)
	validation_df, test_df  = grouped_shuffle_split(validation_test_df, validation_test_df["GeneFamily"], 0.5)

	# We will only add phylo aug samples which are NOT orthologs of genes in the validation or test set, to avoid leakage
	filtered_phylo_aug_df = phylo_aug_df[~phylo_aug_df["Ortholog"].isin(validation_test_df["GeneID"])]
	training_phylo_aug_df = filtered_phylo_aug_df.drop(["Species", "Ortholog"], axis=1)
	
	result_scores = []
	# Limit the amount of samples 
	for i in range(0, 60, 5):
		augmentation_rate  = i / 10
		sampled_phylo_aug = training_phylo_aug_df.sample(int(train_df.shape[0]*augmentation_rate), replace=False, random_state=1)
		augmented_train = pd.concat([sampled_phylo_aug, train_df])
		
		print(f"Length of training set: {augmented_train.shape[0]}")
		print(f"Length of validation set: {validation_df.shape[0]}")
		print(f"Length of test set: {test_df.shape[0]}")

		# Prepare model input
		def get_input_and_labels(df):
			ohe_np = np.stack(df["One_hot_encoded"])
			rc_np = np.stack(df["RC_one_hot_encoded"])

			x = np.append(ohe_np, rc_np, axis=0)
			x = x.astype('float32')
			y = np.append(df["Label"], df["Label"])
			y = y.astype('int64')
			return x, y

		x_train, y_train = get_input_and_labels(augmented_train)
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

		model = get_model(input_shape=input_shape, perform_evoaug=False, augment_list=augment_list,learning_rate=0.001)

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
							epochs=10,
							batch_size=100,
							validation_data=(x_validation, y_validation),
							callbacks=callbacks
							)
		scores = model.evaluate(x_validation, y_validation, verbose=0)

		print(f'Score for augemnation rate {augmentation_rate}: Loss of {scores[0]}; accuracy of {scores[1]}; auroc of {scores[2]}; auprc of {scores[3]}; TP of {scores[4]}')
		result_scores.append({'augmentation_rate': augmentation_rate,'loss': scores[0],'accuracy': scores[1],'auroc': scores[2],'auprc': scores[3],'tp': scores[4]})
	model.save('/home/ubuntu/DeepDifE/data/models/phylo_aug_accession.h5')


	with open('/home/ubuntu/DeepDifE/data/phylo_aug/phylo_aug_rate_results2.csv', 'w', newline='') as csvfile:
		fieldnames = ['augmentation_rate','loss','accuracy','auroc','auprc','tp']
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(result_scores)

if __name__=="__main__":
	main()