import numpy as np

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


def get_input_and_labels(df):
    """
    Helper function for preparing data for model

    Args:
        df (pd.DataFrame): DataFrame that has one hot encoded sequences (and rc)

    Returns:
        two np.ndarray: (1) One hot encoded and reverse sequence tensor,
                        (2) Label tensor corresponding to (1)
    """
    ohe_np = np.stack(df["One_hot_encoded"])
    rc_np = np.stack(df["RC_one_hot_encoded"])
    x = np.append(ohe_np, rc_np, axis=0)
    x = x.astype('float32')
    y = np.append(df["Label"], df["Label"])
    y = y.astype('int')

    return x, y




