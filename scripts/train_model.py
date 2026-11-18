import argparse
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense, Flatten, Dropout, MaxPooling1D, BatchNormalization, ReLU, Conv1D
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC, Recall, Precision

# Arguments given by flags
parser = argparse.ArgumentParser(description='model info')
parser.add_argument('--data', action="store", dest='data')
parser.add_argument('--analysis', action="store", dest='analysis')
parser.add_argument('--splits', action="store", dest='splits', default=5)
parser.add_argument('--shape', action="store", dest='shape', default=600)
parser.add_argument('--name', action="store", dest='name')
args = parser.parse_args()
# Seed for reproducibility
SEED = 42

convolutions = [4,4,3,3,4]
filters_1 = [88,68,100,84,76]
filters_2 = [44,50,42,24,22]
strides = [5,3,4,4,3]
kernel = [8,4,8,6,8]

def main():

    full_data, folds = read_data()
    input_shape = int(args.shape)
    # Training setup

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=20,
                                                    verbose=1,
                                                    mode='min',
                                                    restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                    factor=0.1,
                                                    patience=5,
                                                    min_lr=1e-7,
                                                    mode='min',
                                                    verbose=1)

    callbacks = [reduce_lr, early_stopping_callback]

    for i in range(int(args.splits)):
        split = folds[i]
        train_idx = split[0]
        holdout_idx = split[1]
        full_train = full_data.iloc[train_idx]
        holdout = full_data.iloc[holdout_idx]

        splitter = GroupShuffleSplit(test_size=0.5, n_splits=1, random_state=SEED)
        val_idx = list(splitter.split(holdout, groups=holdout["gene_family"]))[0][0]
        test_idx = list(splitter.split(holdout, groups=holdout["gene_family"]))[0][1]
        validation = holdout.iloc[val_idx]
        test = holdout.iloc[test_idx]

        # get fold specific variables
        number_of_convolutions = convolutions[i]
        number_of_filters_1 = filters_1[i]
        number_of_filters_2 = filters_2[i]
        conv_strides = strides[i]
        kernel_size = kernel[i]

        # Create model
        model_opt = create_model(input_shape, kernel_size, number_of_convolutions, number_of_filters_1, number_of_filters_2, conv_strides)
        x_train, y_train = get_input_and_labels(full_train)
        x_validation, y_validation = get_input_and_labels(validation)

        # Train model
        model_opt.fit(x_train, y_train,
                        epochs = 100,
                        shuffle=True,
                        validation_data=(x_validation, y_validation),
                        verbose=0,
                        callbacks=callbacks)
        
        print(model_opt.evaluate(x_validation, y_validation, verbose=0))
        # save model and test/validation sets
        model_opt.save("../data/models/" + str(args.name) + "_fold_" + str(i) + ".keras")
        test[["GeneID", "Label", "gene_family", "Sequence"]].to_csv("../data/models/" + str(args.name) + "_test_set_fold_" + str(i) + ".csv")
        validation[["GeneID", "Label", "gene_family", "Sequence"]].to_csv("../data/models/" + str(args.name) + "_val_set_fold_" + str(i) + ".csv")
    return 0


def __one_hot_encode_nucleotide(nucleotide):
	"""
	One-hot encode a single nucleotide. This one-hot-encoding can not be changed as 
	it will disturb the symmetry needed by the reverse complement creation
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
	"""
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
	"""
	Add a one-hot-encoded version of the column defined by the column_name to the dataset object 

	Parameters:
	- dataset (pandas DataFrame): Dataframe containing containing the geneIDs and its sequences

	Returns:
	- pandas DataSeries: reverse complement of the input dataSeries
	"""
	reverse_complement = data_series.apply(reverse_complement_sequence)
	return reverse_complement


def complement_sequence(sequence):
    """
    Flip the array over the second axis. This highly depends on the symmetry of the ACGT one-hot-encoding
    """
    complement_one_hot = np.flip(sequence, axis=1)
    return complement_one_hot


def complement_series(data_series):
    """
    Add a one-hot-encoded version of the column defined by the column_name to the dataset object 

    Parameters:
    - dataset (pandas DataFrame): Dataframe containing containing the geneIDs and its sequences

    Returns:
    - pandas DataSeries: complement of the input dataSeries
    """
    complement = data_series.apply(complement_sequence)
    return complement


def read_data():

    full_data = pd.read_csv(args.data)
    full_data = full_data[full_data["Sequence"].str.len() == int(args.shape)]
    full_data["One_hot_encoded"] = one_hot_encode_series(full_data["Sequence"])
    full_data["RC_one_hot_encoded"] = reverse_complement_series(full_data["One_hot_encoded"])
    splitter = GroupKFold(n_splits=int(args.splits))
    folds = list(splitter.split(full_data, full_data["Label"], groups=full_data["gene_family"]))
    return full_data, folds


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

    return x, y


def create_model(input_shape,kernel_size_1,number_of_convolutions,number_of_filters_1,number_of_filters_2,conv_strides):

    dropout_f = 0.3
    kernel_size_2 = 6
    l2_reg = 1e-06
    maxpool_size = 6
    maxpool_stride = 1
    learning_rate = 0.001

    len_seq = input_shape

    model = Sequential()
    model.add(Conv1D(filters = number_of_filters_1,
                            kernel_size = kernel_size_1,
                            kernel_regularizer = regularizers.l2(l2_reg),
                            input_shape = [len_seq, 4],
                            padding = 'same'))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPooling1D(pool_size=maxpool_size, strides=maxpool_stride, padding='same'))

    model.add(Dropout(dropout_f))

    for _ in range(1, number_of_convolutions):
        model.add(Conv1D(filters = number_of_filters_2,
                                kernel_size=kernel_size_2,
                                kernel_regularizer=regularizers.l2(l2_reg),
                                padding='same',
                                strides=conv_strides))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(MaxPooling1D(pool_size=maxpool_size, strides=maxpool_stride, padding='same'))

    model.add(Dropout(dropout_f))

    model.add(Flatten())

    model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(l2_reg)))

    model.add(Dropout(dropout_f))

    model.add(Dense(256/2, activation='relu', kernel_regularizer=regularizers.l2(l2_reg)))

    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer=Adam(learning_rate=learning_rate, weight_decay=None),
                loss='binary_crossentropy',
                metrics=[
                    'acc',
                    AUC(name= "auROC", curve="ROC"),
                    AUC(name = "auPRC", curve="PR"),
                    tf.keras.metrics.TruePositives(),
                    Precision(),
                    Recall()
                ]
    )

    return model


if __name__=="__main__":
    main()
