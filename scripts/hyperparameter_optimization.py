import argparse
import sys
from pathlib import Path
import tensorflow as tf
import optuna
import numpy as np
import statistics
import pandas as pd
from tensorflow.keras.layers import Dense, Flatten, Dropout, MaxPooling1D, BatchNormalization, ReLU, Conv1D
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from sklearn.model_selection import GroupKFold
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC, Recall, Precision

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.prepare_dataset import one_hot_encode_series, reverse_complement_series

# Arguments given by flags
parser = argparse.ArgumentParser(description='model info')
parser.add_argument('--data', action="store", dest='data')
parser.add_argument('--analysis', action="store", dest='analysis')
parser.add_argument('--trials', action="store", dest='trials', default=25)
parser.add_argument('--splits', action="store", dest='splits', default=5)
parser.add_argument('--shape', action="store", dest='shape', default=600)
args = parser.parse_args()


def main():

    full_data, folds = read_data()

    for i in range(int(args.splits)):
        split = folds[i]
        train_idx = split[0]
        test_idx = split[1]
        full_train = full_data.iloc[train_idx]
        holdout = full_data.iloc[test_idx]

        study = optuna.create_study(direction="maximize", study_name="model optimization_" + str(i))
        study.optimize(Objective(full_train), n_trials=int(args.trials), n_jobs=2)
        
        print(optuna.importance.get_param_importances(study))

        best = study.best_trial
        info_array = [str(i), str(best.value), str(study.best_params)]
        with open(args.analysis, 'a', encoding="utf-8") as wr_file:
            wr_file.write('\t'.join(info_array))
            wr_file.write('\n')
    return 0


class Objective:
    # Define class to add parameters to optuna objective
    def __init__(self, full_train):
        
        self.data = full_train
        self.splitter = GroupKFold(int(args.splits))
        self.input_shape = int(args.shape)


    def __call__(self, trial):
        # Train model

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

        # Save metrics
        auroc_score = []

        # Make trial variables before the inner cv loop
        number_of_convolutions = trial.suggest_int("number_of_convolutions", 2, 4, 1)
        number_of_filters_1 = trial.suggest_int("number_of_filters_1", 52, 100, 4)
        number_of_filters_2 = trial.suggest_int("number_of_filters_2", 16, 64, 2)
        conv_strides = trial.suggest_int("conv_strides", 1, 6, 1)
        kernel_size_1 = trial.suggest_int("kernel_size_1", 4, 14, 1)

        # Inner fold
        for _, (train_index, val_index) in enumerate(self.splitter.split(self.data, self.data["Label"], self.data["gene_family"])):
            # Create model
            model_opt = create_model(self.input_shape,kernel_size_1,number_of_convolutions,number_of_filters_1,number_of_filters_2,conv_strides)
            # Data inner fold
            train = self.data.iloc[train_index]
            val = self.data.iloc[val_index]
            x_train, y_train = get_input_and_labels(train)
            x_validation, y_validation = get_input_and_labels(val)

            # Train model
            model_opt.fit(x_train, y_train,
                            epochs = 100,
                            shuffle=True,
                            validation_data=(x_validation, y_validation),
                            verbose=0,
                            callbacks=callbacks)
            
            # calculate auroc score
            auroc_score.append(model_opt.evaluate(x_validation, y_validation, verbose=0)[2])
            print(model_opt.evaluate(x_validation, y_validation, verbose=0))
        return statistics.mean(auroc_score)


def read_data():

    full_data = pd.read_csv(args.data)
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
