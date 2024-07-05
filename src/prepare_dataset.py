import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

def __one_hot_encode_nucleotide(nucleotide):
	"""One-hot encode a single nucleotide."""
	mapping = {'A': [1, 0, 0, 0],
			   'C': [0, 1, 0, 0],
			   'G': [0, 0, 1, 0],
			   'T': [0, 0, 0, 1]}
	return mapping.get(nucleotide, [0, 0, 0, 0])


def one_hot_encode_sequence(sequence):
	"""One-hot encode a sequence of nucleotides."""
	return np.array([__one_hot_encode_nucleotide(nuc) for nuc in sequence])


def one_hot_encode_series(data_series):
	""""
	Add a one-hot-encoded version of the column defined by the column_name to the dataset object 

	Parameters:
	- dataSerie (pandas DataSerie): Dataframe containing containing the geneIDs and its sequences

	Returns:
	- pandas DataSeries: one-hot-encoded version of the input dataSeries
	"""
	one_hot_encoded = data_series.apply(one_hot_encode_sequence)
	return one_hot_encoded


def reverse_complement_sequence(sequence):
	complement_map = np.array([
		[0, 0, 0, 1],  # A -> T
		[0, 0, 1, 0],  # C -> G
		[0, 1, 0, 0],  # G -> C
		[1, 0, 0, 0],  # T -> A
	])
	reverse_complement_one_hot = complement_map[np.argmax(sequence, axis=1)][::-1]
	return reverse_complement_one_hot


def reverse_complement_series(data_series):
	""""
	Add a one-hot-encoded version of the column defined by the column_name to the dataset object 

	Parameters:
	- dataset (pandas DataFrame): Dataframe containing containing the geneIDs and its sequences

	Returns:
	- pandas DataSeries: reverse complement of the input dataSeries
	"""
	reverse_complement = data_series.apply(reverse_complement_sequence)
	return reverse_complement


def grouped_shuffle_split(dataset, groups, test_train_ratio):
	groupSplitter = GroupShuffleSplit(test_size=test_train_ratio, n_splits=1, random_state = 42)
	train_idx = list(groupSplitter.split(dataset, groups=groups))[0][0]
	test_idx = list(groupSplitter.split(dataset, groups=groups))[0][1]

	return dataset.iloc[train_idx], dataset.iloc[test_idx]



