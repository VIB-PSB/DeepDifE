

import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib
import numpy as np
import pandas as pd
import os
import re
import csv
import shap
import importlib
import pathlib
import pickle
from random import sample
from random import sample
from evoaug_tf import evoaug
from src.logo_plot_utils import plot_weights_modified
from matplotlib.ticker import FuncFormatter


def getDeepExplainerBackground(background_samples, shuffle, post_hoc_conjoining):
	"""
	Prepare the background for deepExplainer

	Parameters:
	- background_samples (numpy array): Array of dimension (2, #samples, #sequence length, 4) that will be used as the background
	- shuffle (bool): Shuffle the background
	- post_hoc_conjoining (bool): Prepare a background to get SHAP values for a post-hoc conjoining model

	Returns:
	- numpy array: Array of dimension (2, #samples, #sequence length, 4)

	"""
	fw = background_samples[0]
	rv = background_samples[1]
	if shuffle == True:
		try:
			rng = np.random.default_rng()
			
			fw = rng.permuted(fw, axis = 0)
			rv = rng.permuted(rv, axis = 0)
		except:
			print("Failed to sample randomly")
	
	if post_hoc_conjoining:
		bg = np.stack((fw, rv))
	else:
		bg = fw		

	return bg


def deepExplain(samples, loaded_model, bg, evo_aug=False, post_hoc_conjoining=False, remove_evo_aug_padding=False, augment_list=[], pad_samples=False, pad_background=False):
	"""
	Run deepexplainer based on the provided sequences based on a Tensorflow model and a background
	To determine if the samples or background need to be padded follow this rule of thumb:
	If your model was trained using evo aug make sure both background and samples are padded.

	Parameters:
	- samples (numpy array): Array of dimension (2, #samples, #sequence length, 4)
	- loaded_model (keras/TF): Trained model, either siamese for post-hoc conjoinging of single
	- evo_aug (bool): Indicate if the loaded model performs evo aug
	- post_hoc_conjoining (bool): Besides the forward strand also print the shap values for the reverse compliment
	- remove_evo_aug_padding (bool): If evo aug padding is done, either before or in this function, this padding can be ignored in the generated logo's
	- augment_list (list): List of possible augmentation needed for evo aug
	- pad_samples (bool): Pad the samples with evo-aug padding
	- pad_background (bool): Pad the background with evo-aug padding

	Returns:
	- int: Shap values of size (#samples, 2, #sequence length, 4)

	"""
	if pad_background or pad_samples or remove_evo_aug_padding:
		robust_model = evoaug.RobustModel(loaded_model, augment_list=augment_list)

	fw = samples[0]
	rv = samples[1]

	# Perform padding where necessary
	if pad_samples:
		fw = robust_model._pad_end(fw)
		rv = robust_model._pad_end(rv)

		fw = np.array(fw)
		rv = np.array(rv)

	# Prepare background
	if pad_background:
		if post_hoc_conjoining:
			bg = [robust_model._pad_end(dir) for dir in bg]
			bg = [np.array(dir) for dir in bg]
		else:
			bg = robust_model._pad_end(bg)
	elif post_hoc_conjoining:
		bg = [np.array(dir) for dir in bg]

	e = shap.DeepExplainer((loaded_model.input,loaded_model.layers[-1].output), bg)

	if post_hoc_conjoining:
		"""
		NOTE
		shap has to call tf.compat.v1.disable_v2_behavior() at import of tf, or it wont work. 
		if you call this though before dl, dl wont work
		the way my code is structured is that i cannot switch between v2 (for learning) and v1 (for interpreting)
		this AddV2 snippet in the line under this makes the shap explainer ok to work with v2 ?? 
		"""
		shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough #https://github.com/slundberg/shap/issues/1110
		shap_values = e.shap_values([fw, rv])
		npshap = np.array(shap_values)
	else:
		shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough #https://github.com/slundberg/shap/issues/1110

		shap_values = e.shap_values(fw)
		npshap = np.array(shap_values)

		# Make it again according to the original npshap shape in order not to have to change anything. fw and rv are now a copy of each other.
		npshap = np.tile(npshap, (2, 1, 1, 1, 1))
		npshap = np.expand_dims(npshap, axis=0)

	
	if remove_evo_aug_padding:

		# We only want to display real nucleotides, so we calculate this range 
		totalpadding = robust_model.insert_max
		assert totalpadding%2 == 0, "totalpadding should be even"
		print("removing padding")
		half = int(totalpadding/2)
		unpaddedlen = fw[0].shape[0]-totalpadding

		# Evo aug pads on both sides
		shap_result = [[fwshap[half:unpaddedlen+half], revshap[half:unpaddedlen+half]] for fwshap, revshap in zip(npshap[0][0], npshap[0][1])] #each npshap duo will be of shape (2(list), 300, 4, 1)
	else:
		shap_result = [[fwshap, revshap] for fwshap, revshap in zip(npshap[0][0], npshap[0][1])] #each npshap duo will be of shape (2(list), 300, 4, 1)

	return shap_result


def plotResults(shap_values, samples, post_hoc_conjoining, gene_ids=[], fig_path="", in_silico_mut=False, model=None, plot_title_prefix="Gene "):
	"""
	Plot the SHAP values for a list DNA sequence strands.

	Parameters:
	- shap_values (numpy array): Array of dimension ( #samples, 2, #sequence length, 4) containing the deepexplainer SHAP values
	- samples (numpy array): Array of dimension ( 2, #samples, #sequence length, 4) containing the samples on which the SHAP values are based
	- post_hoc_conjoining (bool): Besides the forward strand also print the shap values for the reverse compliment
	- gene_ids (list): List of Gene ID's to use as plot titles
	- fig_path (String): Path of the directory where the plots should be stored, if empty no plots will be saved.
	- in_silico_mut (bool): Aside from the SHAP values also plot the in silico mutagenesis (this will increase the runtime drastically)
	- model (Keras/TF): Trained model needed for the in silico mutagenesis
	- plot_title_prefix: Prefix added to the plot title (default: "Gene ")

	"""
	# As the shap values are of shape (# samples, 2, #sequencelength, 4), we need to change the shape of the samples
	samples = np.moveaxis(samples, 0, 1)

	for i, (shap_result, sequence) in enumerate(zip(shap_values, samples)):
		if len(gene_ids):
			gene_and_coordinates = gene_ids[i]
		else:
			gene_and_coordinates = ""
		
		if fig_path:
			os.makedirs(fig_path, exist_ok=True)
			cleaned_string = re.sub(r'[^a-zA-Z0-9]', '_', gene_and_coordinates)
			full_path = f"{fig_path}/{cleaned_string}_deepexplainer.svg"
		else:
			full_path = fig_path
		
		# Find indices of padding
		matching_indices = np.where(np.all(sequence[0] == [0, 0, 0, 0], axis=1))[0]

		if len(matching_indices) > 0:
			if matching_indices[0] > 0:
				# In case the padding is on the right
				start = 0
				stop = matching_indices[0]
			else:
				# In case the padding is on the left
				start = matching_indices[-1] + 1
				stop = len(sequence[0])
		else:
			# In case the padding there is no padding
			start = 0
			stop = len(sequence[0])
		
		__plot_saliency_map(shap_result=shap_result, 
					  		sequence=sequence, 
							start_offset=start, 
							stop_offset=stop, 
							post_hoc_conjoining=post_hoc_conjoining, 
							gene_and_coordinates=gene_and_coordinates, 
							full_path=full_path, 
							in_silico_mut=in_silico_mut, 
							model=model,
							plot_title_prefix=plot_title_prefix)

def plotChunkedResults(shap_values, samples, post_hoc_conjoining, gene_ids=[], fig_path="", in_silico_mut=False, model=None, plot_title_prefix="Gene ", stride=500):
	"""
	Plot the SHAP values for a list of DNA sequence strands. For every sequence, both a complete saliency map will be plot as also 
	a series of sliding window saliency maps. For larger sequences this provides a way to have a more detailed overview of the SHAP values.

	Parameters:
	- shap_values (numpy array): Array of dimension ( #samples, 2, #sequence length, 4) containing the deepexplainer SHAP values
	- samples (numpy array): Array of dimension ( 2, #samples, #sequence length, 4) containing the samples on which the SHAP values are based
	- post_hoc_conjoining (bool): Besides the forward strand also print the shap values for the reverse compliment
	- gene_ids (list): List of Gene ID's to use as plot titles
	- fig_path (String): Path of the directory where the plots should be stored, if empty no plots will be saved.
	- in_silico_mut (bool): Aside from the SHAP values also plot the in silico mutagenesis (this will increase the runtime drastically)
	- model (Keras/TF): Trained model needed for the in silico mutagenesis
	- plot_title_prefix (String): Prefix added to the sample id in the plot title (default: 'Gene ')
	- string (int): Stride used for the sliding window.

	"""
	# As the shap values are of shape (# samples, 2, #sequencelength, 4), we need to change the shape of the samples
	samples = np.moveaxis(samples, 0, 1)
	
	for i, (shap_result, sequence) in enumerate(zip(shap_values, samples)):

		# Find indices of padding
		matching_indices = np.where(np.all(sequence[0] == [0, 0, 0, 0], axis=1))[0]

		if len(matching_indices) > 0:
			if matching_indices[0] > 0:
				# In case the padding is on the right
				start_offset = 0
				stop_offset = matching_indices[0]
			else:
				# In case the padding is on the left
				start_offset = matching_indices[-1] + 1
				stop_offset = len(sequence[0])
		else:
			# In case the padding there is no padding
			start_offset = 0
			stop_offset = len(sequence[0])
		
		coordinates_included = False
		gene_id_short = ""
		chromosome = ""
		strand = ""
		coordinate_start = 0

		if len(gene_ids):
			pattern = re.compile('([^::]*)::([^:]*):([^-]*)-([^\(]*)(.*)')
			gene_id = gene_ids[i]

			if(pattern.match(gene_id)):
				title_search = pattern.search(gene_id)
				gene_id_short = title_search.group(1)
				chromosome = title_search.group(2)
				coordinate_start = int(title_search.group(3))
				strand = title_search.group(5)
				coordinates_included = True
			else:
				gene_id_short = re.sub(r'[^a-zA-Z0-9]', '_', gene_id)
				coordinates_included = False

		
		# Create subdirectory
		os.mkdir(f"{fig_path}/{gene_id_short}")
		fig_subpath = f"{fig_path}/{gene_id_short}"

		# Plot whole saliency map
		full_path = __get_filename(gene_ids[i], fig_subpath, add_full_postfix=True)
		__plot_saliency_map(shap_result=shap_result,
					  		sequence=sequence,
							start_offset=start_offset,
							stop_offset=stop_offset,
							post_hoc_conjoining=post_hoc_conjoining,
							gene_and_coordinates=gene_ids[i],
							full_path=full_path,
							in_silico_mut=in_silico_mut,
							model=model,
							plot_title_prefix=plot_title_prefix)

		# Plot in chunked saliency map
		while(start_offset + stride < stop_offset):
			gene_and_coordinates = __get_gene_and_coordinate_name(
										coordinates_included=coordinates_included, 
										gene_id_short=gene_id_short, 
										chromosome=chromosome, 
										coordinate_start=coordinate_start, 
										start_offset=start_offset, 
										stop_offset=start_offset+stride, 
										strand=strand)
									
			full_path = __get_filename(gene_and_coordinates, fig_subpath)
			__plot_saliency_map(shap_result=shap_result,
					   			sequence=sequence,
								start_offset=start_offset,
								stop_offset=start_offset + stride,
								post_hoc_conjoining=post_hoc_conjoining,
								gene_and_coordinates=gene_and_coordinates,
								full_path=full_path,
								in_silico_mut=in_silico_mut, 
								model=model,
								plot_title_prefix=plot_title_prefix)
			start_offset = start_offset + stride

		# Plot the remaining part of the sequence
		gene_and_coordinates = __get_gene_and_coordinate_name(
										coordinates_included=coordinates_included, 
										gene_id_short=gene_id_short, 
										chromosome=chromosome, 
										coordinate_start=coordinate_start, 
										start_offset=start_offset, 
										stop_offset=stop_offset, 
										strand=strand)
		full_path = __get_filename(gene_and_coordinates, fig_subpath)
		__plot_saliency_map(shap_result=shap_result, 
					  		sequence=sequence, 
							start_offset=start_offset, 
							stop_offset=stop_offset, 
							post_hoc_conjoining=post_hoc_conjoining, 
							gene_and_coordinates=gene_and_coordinates, 
							full_path=full_path, 
							in_silico_mut=in_silico_mut, 
							model=model,
							plot_title_prefix=plot_title_prefix)


def __get_gene_and_coordinate_name(coordinates_included, gene_id_short, chromosome="", coordinate_start=0, start_offset=0, stop_offset=1, strand=""):
	"""
	Create gene id string with coordinates. 
	"""
	if coordinates_included:
		gene_and_coordinates = f"{gene_id_short}::{chromosome}:{coordinate_start + start_offset}-{coordinate_start + stop_offset}{strand}"
	else:
		gene_and_coordinates = f"{gene_id_short}_{str(start_offset)}_{str(stop_offset)}"
	return gene_and_coordinates


def __get_filename(gene_and_coordinates, fig_path, add_full_postfix=False):
	"""
	Create filename if necessary. 
	"""
	if fig_path:
		if(add_full_postfix):
			full_path = f"{fig_path}/{gene_and_coordinates}_full.png"
		else:
			full_path = f"{fig_path}/{gene_and_coordinates}.png"
	else:
		full_path = fig_path

	return full_path

def __create_bp_x_axis_formatter(ax):
	def x_axis_formatter(x, pos):
		# Get the tick locations and find the last tick
		ticks = ax.get_xticks()
		last_tick = ticks[-1] if len(ticks) > 0 else None
		
		# Add "bp" only to the last tick shown
		if x == last_tick:
			return f'{int(x)}bp'
		return f'{int(x)}'
	return FuncFormatter(x_axis_formatter)

def __plot_saliency_map(shap_result, sequence, start_offset, stop_offset, post_hoc_conjoining, gene_and_coordinates, full_path, in_silico_mut, model, plot_title_prefix):
	ntrack = 3 if in_silico_mut else 2
	fig = plt.figure(figsize=(32,8))
	
	if gene_and_coordinates:
		fig.suptitle(f"{plot_title_prefix}{gene_and_coordinates}", x=0.51)


	_, ax1 =plot_weights_modified((shap_result[0]*sequence[0])[start_offset:stop_offset,:],
								fig,
								ntrack,
								1,
								1,
								title="Input sequence", 
								subticks_frequency=10,
								ylab="Attribution scores",
								)#highlight=coords #titleDictList[i]["startstop"] #,highlight={"black":[info[3] for info in titleDictList[i]["startstops"]]}

	ax1.xaxis.set_major_formatter(__create_bp_x_axis_formatter(ax1))

	if post_hoc_conjoining:
		_, ax2 =plot_weights_modified(((shap_result[1]*sequence[1])[::-1,:][start_offset:stop_offset,:]),
									fig,
									ntrack,
									1,
									2,
									title="Reverse complement",
									subticks_frequency=10,
									ylab="Attribution scores",
									)
	
		min_y = min(ax1.get_ylim()[0], ax2.get_ylim()[0])
		max_y = max(ax1.get_ylim()[1], ax2.get_ylim()[1])

		ax1.set_ylim(min_y, max_y)
		ax2.set_ylim(min_y, max_y)

		ax2.xaxis.set_major_formatter(__create_bp_x_axis_formatter(ax2))

	# In silico mutagenesis
	if in_silico_mut:
		selected_classes = ["up"]
		
		# Create empty numpy to contain the prediction delta for every nucleotide position
		arrr_A = np.zeros((len(selected_classes),sequence[0].shape[0]))
		arrr_C = np.zeros((len(selected_classes),sequence[0].shape[0]))
		arrr_G = np.zeros((len(selected_classes),sequence[0].shape[0]))
		arrr_T = np.zeros((len(selected_classes),sequence[0].shape[0]))

		# Perform a baseline prediction for the sequence without any mutations
		duo = [[np.expand_dims(sequence[0],2)], [np.expand_dims(sequence[1],2)]]
		duo = [np.expand_dims(np.expand_dims(sequence[0],0),3), np.expand_dims(np.expand_dims(sequence[1],0),3)]
		real_score = model.predict(duo)[0]			
		
		for mutloc in range(sequence[0].shape[0]):
			rowohs = np.copy(sequence[0])

			new_X_,new_X_RC = __mutate(rowohs, "A", mutloc)
			#removing the lists around the input to the model, because it apparently now expects real arrays and not lists of arrays... :/
			preda = model.predict([new_X_,new_X_RC], verbose=0)
			arrr_A[:,mutloc]=(real_score - preda[0][0])

			new_X_,new_X_RC = __mutate(rowohs, "C", mutloc)
			arrr_C[:,mutloc]=(real_score - model.predict([new_X_,new_X_RC], verbose=0)[0])

			new_X_,new_X_RC = __mutate(rowohs, "G", mutloc)
			arrr_G[:,mutloc]=(real_score - model.predict([new_X_,new_X_RC], verbose=0)[0])

			new_X_,new_X_RC = __mutate(rowohs, "T", mutloc)
			arrr_T[:,mutloc]=(real_score - model.predict([new_X_,new_X_RC], verbose=0)[0])

		# We only want to plot the prediction delta of possbile mutations, 
		# we set the delta 0 to None as these are the original nucleotides present in the sequence
		arrr_A[arrr_A==0]=None
		arrr_C[arrr_C==0]=None
		arrr_G[arrr_G==0]=None
		arrr_T[arrr_T==0]=None

		# Plotting for every nucleotide mutation
		ax = fig.add_subplot(ntrack,1,3)
		ax.scatter(range(rowohs.shape[0]),-1*arrr_A[[0]],label='A',color='green')
		ax.scatter(range(rowohs.shape[0]),-1*arrr_C[[0]],label='C',color='blue')
		ax.scatter(range(rowohs.shape[0]),-1*arrr_G[[0]],label='G',color='orange')
		ax.scatter(range(rowohs.shape[0]),-1*arrr_T[[0]],label='T',color='red')
		ax.legend()
		ax.axhline(y=0,linestyle='--',color='gray')

	if full_path:
		fig.savefig(full_path, facecolor='white')
	
	plt.show()


def __mutate(new_X_, base, mutloc):
	'''
	squeezes, mutates and reexpands, makes rc
	'''

	# new_X_dm =  np.squeeze(new_X_) #(300, 4, 1) to (300, 4)
	if base == "A":
		new_X_[mutloc,:] = np.array([1, 0, 0, 0], dtype='int8')
	elif base == "C":
		new_X_[mutloc,:] = np.array([0, 1, 0, 0], dtype='int8')
	elif base == "G":
		new_X_[mutloc,:] = np.array([0, 0, 1, 0], dtype='int8')
	elif base == "T":
		new_X_[mutloc,:] = np.array([0, 0, 0, 1], dtype='int8')

	new_X_dm_expand = np.expand_dims(new_X_, axis=2) #(300, 4) to (300, 4, 1) 

	new_X_dm_expand_RC = new_X_dm_expand[::-1,::-1,:] #(300, 4, 1) 

	#expand once more, after not working with the v1 compatibility anymore.... 
	new_X_dm_expand_expand = np.expand_dims(new_X_dm_expand, axis=0) #(300, 4, 1) to (1, 300, 4, 1)
	new_X_dm_expand_RC_expand = np.expand_dims(new_X_dm_expand_RC, axis=0) #(300, 4, 1) to (1, 300, 4, 1)

	return new_X_dm_expand_expand, new_X_dm_expand_RC_expand

