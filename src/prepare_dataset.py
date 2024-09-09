import pandas as pd
import numpy as np
from Bio import SeqIO
from sklearn.model_selection import GroupShuffleSplit

def __one_hot_encode_nucleotide(nucleotide):
	"""
	One-hot encode a single nucleotide. This one-hot-encoding can not be changed as 
	it will disturb the symmetrie needed by the reverse complement creation
	"""
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
	"""
	Flip the array over both axes. This highly depends on the symmetry of the ACGT one-hot-encoding
	"""
	reverse_complement_one_hot = np.flip(sequence)
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


def prepare_dataset_from_fasta(fasta_path, dataset, output_path=None):
	""""
	Based on a dataset containing the gene ids, labels and genefamilies, extract the sequences from a fasta file.
	The gene id's of the fasta file are expected to be in the following format: <GeneID>::<additional info>
	In the resulting dataframe, the gene id from the fasta file is used.

	Parameters:
	- dataset (pandas DataFrame): Dataframe containing containing the following columns GeneID, Label and GeneFamily
	- fasta_url (String): Path to the fasta file containing the sequences
	- output_path (String): Path where the resulting dataframe will be saved as a CSV file. If None no file will be saved.

	Returns:
	- pandas DataFrame: Dataframe containing the gene id, label, gene family and sequence
	"""

	# Check if dataset contains the necessary columns
	required_columns = ["GeneID", "Label", "GeneFamily"]
	missing_columns = [col for col in required_columns if col not in dataset.columns]

	if missing_columns:
		print(f"Missing columns: {', '.join(missing_columns)}")
	else:
		print("All required columns are present.")

	# Create the dicts for easy lookup
	label_dict = dataset.set_index('GeneID')['Label'].to_dict()
	gf_dict = dataset.set_index('GeneID')['GeneFamily'].to_dict()

	gene_id_list	= []
	sequence_list	= []
	label_list		= []
	gf_list			= []

	genes = list(dataset.GeneID)
	
	fasta_sequences = SeqIO.parse(open(fasta_path),'fasta')

	for fasta in fasta_sequences:
		name, sequence = fasta.id, str(fasta.seq)
		gene_id = name.split(":")[0]
		if gene_id in genes:
			gene_id_list.append(name)
			sequence_list.append(sequence)
			label_list.append(label_dict[gene_id])
			gf_list.append(gf_dict[gene_id])
	
	complete_dataset = pd.DataFrame({"GeneID":gene_id_list, "Label":label_list, "GeneFamily": gf_list, "Sequence": sequence_list})

	if output_path:
		complete_dataset.to_csv(output_path, index=False)

	return complete_dataset




