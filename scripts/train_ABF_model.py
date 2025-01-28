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
from src.prepare_dataset import one_hot_encode_series, reverse_complement_series, reverse_complement_sequence

def get_input_and_labels(df):
	ohe_np = np.stack(df["One_hot_encoded"])
	rc_np = np.stack(df["RC_one_hot_encoded"])

	x = np.append(ohe_np, rc_np, axis=0)
	x = x.astype('float32')
	y = np.append(df["Label"], df["Label"])
	return x, y

def main():
    # Prepare dataset

    dataset_1 = pd.read_csv("/home/ubuntu/DeepDifE/abf_data/variable_peak_data/ABF1_train_shift_peaks.csv")
    dataset_2 = pd.read_csv("/home/ubuntu/DeepDifE/abf_data/variable_peak_data/ABF2_train_shift_peaks.csv")
    dataset_3 = pd.read_csv("/home/ubuntu/DeepDifE/abf_data/variable_peak_data/ABF3_train_shift_peaks.csv")
    dataset_4 = pd.read_csv("/home/ubuntu/DeepDifE/abf_data/variable_peak_data/ABF4_train_shift_peaks.csv")

    dataset = pd.concat([dataset_1, dataset_2, dataset_3, dataset_4], ignore_index=True, axis=0)

    from src.prepare_dataset import one_hot_encode_series, reverse_complement_series, reverse_complement_sequence
    dataset["One_hot_encoded"] = one_hot_encode_series(dataset["Sequence"])
    dataset["RC_one_hot_encoded"] = reverse_complement_series(dataset["One_hot_encoded"])

    from src.prepare_dataset import grouped_shuffle_split
    train_df, validation_test_df = grouped_shuffle_split(dataset, dataset["GeneFamily"], 0.2)
    validation_df, test_df  = grouped_shuffle_split(validation_test_df, validation_test_df["GeneFamily"], 0.5)

    x_train, y_train = get_input_and_labels(train_df)
    x_validation, y_validation = get_input_and_labels(validation_df)


    input_shape = train_df["One_hot_encoded"].iloc[0].shape
    for i in range(0, 10):
        model = get_model(input_shape=input_shape, perform_evoaug=False ,learning_rate=0.001)

        # early stopping callback
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

        history = model.fit(x_train,
                        y_train,
                        epochs=100,
                        batch_size=100,
                        validation_data=(x_validation, y_validation),
                        callbacks=callbacks
                        )
        
        siamese_model = get_siamese_model(model)
        x_test = np.stack(test_df["One_hot_encoded"])
        x_test_rc = np.stack(test_df["RC_one_hot_encoded"])
        y_test = test_df["Label"].to_numpy()

        predictions_categories, predictions = post_hoc_conjoining(siamese_model, x_test, x_test_rc)
        print(get_auroc(y_test, predictions))

        # Test on DE dataset
        ppath = "/home/ubuntu/DeepDifE/data/dataset_solid_chrome.pkl"
        with open(ppath, 'rb') as f:
            data = pickle.load(f)
        dataset = data.reset_index()
        test_df_helder = dataset[dataset.set == "test"]
        print(test_df_helder.columns)
        x_test = np.stack(test_df_helder["ohs"])
        x_test_rc = np.stack(test_df_helder["rcohs"])
        y_test = test_df_helder["Category"].to_numpy()
        predictions_categories, predictions = post_hoc_conjoining(siamese_model, x_test, x_test_rc)
        test_auroc = get_auroc(y_test, predictions)
        f = open("/home/ubuntu/DeepDifE/abf_data/results/result_ABF_DE_10_runs", "a")
        f.write(f'{test_auroc}\n')
        f.close()

if __name__=="__main__":
    main()