from evoaug_tf import evoaug, augment

# Tensorflow imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Average, Maximum
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling1D, MaxPooling2D, AveragePooling2D, Activation, concatenate, BatchNormalization, maximum, Lambda
from tensorflow.keras import regularizers 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from tensorflow.keras.utils import to_categorical  #alternative for the above line
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import model_from_json

import numpy as np
from sklearn import metrics



def get_tf_model(input_shape, kernel_size, number_of_convolutions, perform_max_pooling, second_convolution):

	len_seq = input_shape[0]
	prm = {
	"batch": 256,
	"conv": [
		64,
		64,
		32
	],
	"dense": 256,
	"dropout": 0.25,
	"epochs": 100,
	"maxPool": [
		6,
		6,
		6
	],
	"num_bloks": 3,
	"regularizer": 1e-06,
	"wind_size": [
		12,
		6,
		6
	]
	}
	
	model = Sequential()
	model.add(Conv2D(filters = prm['conv'][0],
								kernel_size = kernel_size, #"wind_size": [12, 6, 6] (shrikumar onehot order reverses this)
								activation = 'relu',
								kernel_regularizer = regularizers.l2(prm['regularizer']),
								input_shape = [len_seq, 4, 1],
								padding = 'valid'))

	model.add(MaxPooling2D(pool_size=(prm['maxPool'][0],1), strides=(prm['maxPool'][0],1)))
	model.add(Dropout(prm['dropout']))

	for i in range(1, number_of_convolutions):
		model.add(Conv2D(filters = prm['conv'][i],
							kernel_size=(prm['wind_size'][i],1),
							activation='relu',
							kernel_regularizer=regularizers.l2(prm['regularizer']),
							strides=(prm['wind_size'][i],1),
							padding='same'))
		model.add(Dropout(prm['dropout']))

	model.add(Flatten())
	model.add(Dense(prm['dense'], activation='relu', kernel_regularizer=regularizers.l2(prm['regularizer'])))
	model.add(Dropout(prm['dropout']))
	model.add(Dense(prm['dense']/2, activation='relu', kernel_regularizer=regularizers.l2(prm['regularizer'])))
	model.add(Dense(1, activation='sigmoid')) # 'sigmoid' if you don't encode the classes to categorical, None (without "") for conjoining

	return model    


def get_model(input_shape=(600,4),
				perform_evoaug=True,
				augment_list=[],
				finetune=False,
				learning_rate=0.001,
				kernel_size=(12,4),
				number_of_convolutions=3,
				perform_max_pooling=False, 
				second_convolution=6):


	if perform_evoaug:
		optimizer = Adam(learning_rate=learning_rate)

		if finetune:
			optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

	else:
		pass


	if perform_evoaug:
		model = evoaug.RobustModel(get_tf_model, 
					kernel_size=kernel_size, 
					number_of_convolutions=number_of_convolutions, 
					input_shape=input_shape, 
					perform_max_pooling=perform_max_pooling, 
					second_convolution=second_convolution, 
					augment_list=augment_list, 
					max_augs_per_seq=2, 
					hard_aug=True)
		if finetune:
			model = evoaug.RobustModel(get_tf_model, 
							kernel_size=kernel_size, 
							number_of_convolutions=number_of_convolutions, 
							input_shape=input_shape, 
							perform_max_pooling=perform_max_pooling, 
							second_convolution=second_convolution, 
							augment_list=augment_list, 
							max_augs_per_seq=1, 
							hard_aug=True)
	else:
		model = get_tf_model(input_shape)

	# Compile model
	model.compile(optimizer = optimizer,
				loss = 'binary_crossentropy',
				metrics=[
					'acc',
					AUC(name= "auROC", curve="ROC"),
					AUC(name = "auPRC", curve="PR"),
					tf.keras.metrics.TruePositives(),
					tf.keras.metrics.AUC(curve='ROC', name='auroc'),
				]
	)

	return model

def load_model(model_path, weights_path):
	'''
	model_kind can be "single" or "posthoc"
	Also gets used in deepExplainer_tools.py
	'''

	json_file = open(model_path, 'r') 
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	loaded_model.load_weights(weights_path)
	print("\nLoaded model from disk")
	return loaded_model


def get_siamese_model(model):
	binary_model_getlogits = keras.models.Model(inputs=model.inputs,
												outputs=model.layers[-1].output)

	first_dimension = model.inputs[0].shape[1]

	fwd_sequence_input = keras.layers.Input(shape=(first_dimension, 4))
	rev_sequence_input = keras.layers.Input(shape=(first_dimension, 4))
	fwd_logit_output = binary_model_getlogits(fwd_sequence_input)
	rev_logit_output = binary_model_getlogits(rev_sequence_input)
	average_logits = keras.layers.Average()([fwd_logit_output, rev_logit_output])
	sigmoid_out = keras.layers.Activation("linear")(average_logits)

	siamese_model = keras.models.Model(inputs=[fwd_sequence_input,rev_sequence_input],
											outputs=[sigmoid_out])
	return siamese_model


def post_hoc_conjoining(siamese_model, x_fw, x_rc):

	prediction = siamese_model.predict([x_fw, x_rc])
	predicted_categories = []

	for i, pr in enumerate(prediction):
		if pr > 0.5: 
			predicted_categories.append(1)
		else:
			predicted_categories.append(0)

	return predicted_categories, prediction.squeeze()


def get_auroc(y_labels, prediction):
	fpr, tpr, thresholds = metrics.roc_curve(y_labels, prediction, pos_label=1)
	auROC = metrics.auc(fpr, tpr)
	return auROC